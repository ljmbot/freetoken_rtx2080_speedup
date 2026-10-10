from __future__ import annotations

import functools
import glob
import json
import os
import re
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    detect_compressed_tensors_nvfp4,
)


def _quant_accessor(hf_config: Any):
    """A ``get(key, default=None)`` accessor over the HF ``quantization_config`` (dict or
    object), or ``None`` when the model has no quant config."""
    quant = getattr(hf_config, "quantization_config", None)
    if quant is None:
        return None
    return quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))


def _fp8_block_quant(hf_config: Any) -> tuple[str, tuple[int, int] | None]:
    """Detect DeepSeek-V3-style 128x128 block-fp8 from HF ``quantization_config``.

    Returns ``("fp8_block", (block_n, block_k))`` for a block-fp8 checkpoint (weights
    fp8-e4m3 + per-block ``weight_scale_inv``, dynamic activation), else ``("none", None)``.
    The quantization_config sits on the top-level hf_config (not ``text_config``).
    """
    get = _quant_accessor(hf_config)
    if get is None:
        return "none", None
    method = str(get("quant_method") or get("quant_algo") or "").lower()
    block = get("weight_block_size")
    if method == "fp8" and block:
        bs = tuple(int(x) for x in block)
        assert bs == (128, 128), f"only 128x128 block-fp8 is supported, got {bs}"
        return "fp8_block", bs
    return "none", None


def _expert_quant(hf_config: Any) -> str:
    """Quantization format of the *routed* experts (the only weights served from the
    offload cache). The nvidia/modelopt checkpoints are either plain NVFP4 (``quant_algo``
    ``NVFP4``) or ``MIXED_PRECISION`` (per-layer ``quantized_layers`` map); in the mixed
    case the routed experts carry their own ``W4A16_NVFP4``/``FP8`` algo. Dense quantized
    weights (attention/shared-expert/lm_head) are handled separately by dequant-at-load."""
    get = _quant_accessor(hf_config)
    if get is None:
        return "none"
    algo = str(get("quant_algo") or get("quant_method") or "").lower()
    if "fp4" in algo:
        return "nvfp4"
    if "mixed" in algo:
        layers = get("quantized_layers") or {}
        for name, spec in (layers.items() if isinstance(layers, dict) else []):
            if name.endswith(".mlp.experts") or ".mlp.experts." in name:
                expert_algo = str((spec or {}).get("quant_algo", "")).lower()
                if "fp4" in expert_algo:
                    return "nvfp4"
                if "fp8" in expert_algo:
                    return "fp8"
    return "none"


# Detection now lives in models/config.py (shared with muse_glimmer); weight.py imports
# it under this name.
_compressed_tensors_nvfp4 = detect_compressed_tensors_nvfp4


# Routed-expert NVFP4 tensor in a compressed-tensors checkpoint (per-expert, un-fused,
# ``model[.language_model].layers.N.mlp.experts.E.{gate,up,down}_proj.weight_packed``).
# The optional ``language_model.`` infix covers both the VLM wrapper (Ornith:
# ``model.language_model.…``) and the flat text-only config (Coder-Next: ``model.…``);
# the leading ``^model\.`` anchor still excludes the MTP head's ``mtp.layers.*`` tensors.
_CT_EXPERT_NVFP4_RE = re.compile(
    r"^model\.(?:language_model\.)?layers\.\d+\.mlp\.experts\.\d+\.(?:gate|up|down)_proj\.weight_packed$"
)


def _ct_checkpoint_path(hf_config: Any) -> str | None:
    """Local checkpoint folder for a config (``_name_or_path`` when it is an existing
    directory), else ``None``. ``cached_load_hf_config`` rebuilds the config via
    ``type(config)(**config.to_dict())`` which preserves ``_name_or_path`` for a local
    directory; hub ids resolve to a cache dir and are not inspected (they carry an index)."""
    p = getattr(hf_config, "_name_or_path", None)
    if isinstance(p, str) and p and os.path.isdir(p):
        return p
    return None


@functools.lru_cache(maxsize=8)
def _ct_nvfp4_module_map(model_path: str) -> dict[str, bool]:
    """Inspect the checkpoint's safetensors header: which modules are stored as NVFP4
    (``weight_packed``) vs bf16. The ``quantization_config`` flag cannot express "only the
    MLP is NVFP4, attention is bf16" and carries a stale ``ignore`` list, so the tensor
    layout is the ground truth. Returns ``{"has_moe_experts_nvfp4", "self_attn_nvfp4",
    "gdn_outproj_nvfp4", "shared_expert_nvfp4", "dense_mlp_nvfp4"}``; all False if the
    layout can't be read.

    * ``has_moe_experts_nvfp4`` -- any ``.mlp.experts.<e>.{gate,up,down}_proj.weight_packed``
    * ``self_attn_nvfp4`` -- any full-attention ``.self_attn.{q,k,v,o}_proj.weight_packed``
      (the full-attention projections the model runs through the W4A16 kernels when
      ``attn_quant=="nvfp4"``)
    * ``gdn_outproj_nvfp4`` -- any GDN ``.linear_attn.out_proj.weight_packed`` (tracked
      separately: a CT build can quantize self_attn while leaving GDN ``out_proj`` dense bf16,
      e.g. the plain Ornith NVFP4)
    * ``shared_expert_nvfp4`` -- any ``.mlp.shared_expert.{gate,up,down}_proj.weight_packed``
    * ``dense_mlp_nvfp4`` -- any bare ``.mlp.{gate,up,down}_proj.weight_packed`` (dense
      non-MoE MLP; the MoE shared_expert/experts infixes do not end with these)
    """
    result = {
        "has_moe_experts_nvfp4": False,
        "self_attn_nvfp4": False,
        "gdn_outproj_nvfp4": False,
        "shared_expert_nvfp4": False,
        "dense_mlp_nvfp4": False,
    }
    index = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index):
        try:
            with open(index, encoding="utf-8") as f:
                names = list(json.load(f)["weight_map"].keys())
        except (OSError, ValueError, KeyError):
            return result
    else:  # single-file / no-index checkpoint: scan each shard's header
        names = []
        for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
            if shard.endswith("consolidated.safetensors"):
                continue
            try:
                with open(shard, "rb") as fh:
                    n = int.from_bytes(fh.read(8), "little")
                    hdr = json.loads(fh.read(n))
            except (OSError, ValueError):
                continue
            names.extend(name for name in hdr if name != "__metadata__")
    for name in names:
        _ct_module_signal(name, result)
    return result


def _ct_module_signal(name: str, result: dict[str, bool]) -> None:
    """Update ``result`` in place from one tensor name (no-op for non-``weight_packed``)."""
    if not name.endswith(".weight_packed"):
        return
    if _CT_EXPERT_NVFP4_RE.match(name):
        result["has_moe_experts_nvfp4"] = True
    elif ".mlp.shared_expert." in name and (
        name.endswith((".gate_proj.weight_packed", ".up_proj.weight_packed", ".down_proj.weight_packed"))
    ):
        result["shared_expert_nvfp4"] = True
    elif name.endswith((".self_attn.q_proj.weight_packed", ".self_attn.k_proj.weight_packed",
                        ".self_attn.v_proj.weight_packed", ".self_attn.o_proj.weight_packed")):
        result["self_attn_nvfp4"] = True
    elif name.endswith(".linear_attn.out_proj.weight_packed"):
        result["gdn_outproj_nvfp4"] = True
    elif name.endswith((".mlp.gate_proj.weight_packed", ".mlp.up_proj.weight_packed",
                        ".mlp.down_proj.weight_packed")):
        result["dense_mlp_nvfp4"] = True


def _lm_head_quant(hf_config: Any) -> str:
    """Whether the checkpoint stores ``lm_head`` as NVFP4. modelopt MIXED_PRECISION lists it in
    the per-layer ``quantized_layers`` map (``W4A16_NVFP4``); pure-NVFP4 checkpoints have no
    per-layer map and leave lm_head bf16. Returns ``"nvfp4"`` or ``"none"``."""
    get = _quant_accessor(hf_config)
    if get is None:
        return "none"
    layers = get("quantized_layers") or {}
    if not isinstance(layers, dict):
        return "none"
    for name, spec in layers.items():
        if name == "lm_head" or name.endswith(".lm_head"):
            if "fp4" in str((spec or {}).get("quant_algo", "")).lower():
                return "nvfp4"
    return "none"


def _dense_mlp_quant(hf_config: Any) -> str:
    """NVFP4 on the *dense* (non-MoE) decoder MLP. modelopt MIXED_PRECISION dense checkpoints
    (e.g. Qwen3.6-27B-NVFP4) list ``.mlp.{gate,up,down}_proj`` as ``W4A16_NVFP4`` in
    ``quantized_layers``; MoE checkpoints have ``.mlp.experts.*`` / ``.mlp.shared_expert.*``
    instead (covered by ``expert_quant``). ``endswith(".mlp.gate_proj")`` matches only the bare
    dense MLP -- not ``.mlp.shared_expert.gate_proj`` nor ``.mlp.experts.N.gate_proj``."""
    get = _quant_accessor(hf_config)
    if get is None:
        return "none"
    layers = get("quantized_layers") or {}
    if not isinstance(layers, dict):
        return "none"
    for name, spec in layers.items():
        if name.endswith((".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj")):
            if "fp4" in str((spec or {}).get("quant_algo", "")).lower():
                return "nvfp4"
    return "none"


def _attn_quant(hf_config: Any) -> str:
    """Per-tensor FP8 on the *dense* attention/GDN projections. The modelopt
    ``MIXED_PRECISION`` checkpoints tag ``self_attn.{q,k,v,o}_proj`` and
    ``linear_attn.{in_proj_qkv,in_proj_z,out_proj}`` with ``quant_algo`` ``FP8`` (fp8-e4m3
    weight + a scalar ``weight_scale``; W8A16). Returns ``"fp8_pertensor"`` when present,
    else ``"none"`` (NVFP4 dense weights -- shared_expert/lm_head -- stay dequant-at-load)."""
    get = _quant_accessor(hf_config)
    if get is None:
        return "none"
    layers = get("quantized_layers") or {}
    if not isinstance(layers, dict):
        return "none"
    for name, spec in layers.items():
        algo = str((spec or {}).get("quant_algo", "")).lower()
        if algo == "fp8" and (".self_attn." in name or ".linear_attn." in name):
            return "fp8_pertensor"
    return "none"


def _layer_types(text: Any) -> list[str]:
    layer_types = getattr(text, "layer_types", None)
    if layer_types is not None:
        return list(layer_types)
    # Fall back to full_attention_interval: every Nth layer (1-indexed) is full.
    interval = int(getattr(text, "full_attention_interval", 4))
    n = int(text.num_hidden_layers)
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n)
    ]


def parse_config(hf_config: Any) -> ModelConfig:
    text = getattr(hf_config, "text_config", hf_config)

    head_dim = (
        getattr(text, "head_dim", None)
        or text.hidden_size // text.num_attention_heads
    )
    num_kv_heads = getattr(text, "num_key_value_heads", text.num_attention_heads)

    rope_params = getattr(text, "rope_parameters", None) or {}
    rope_theta = rope_params.get("rope_theta", getattr(text, "rope_theta", None))
    partial = (
        rope_params.get("partial_rotary_factor")
        or getattr(text, "partial_rotary_factor", None)
        or 1.0
    )
    rotary_dim = round(head_dim * partial)

    # For text-only with the default rope type, partial NeoX rope needs no scaling dict
    # (the mRoPE params reduce to standard partial rope for text). Avoid carrying the
    # unhashable ``mrope_section`` list into get_rope's cache key.
    rope_type = rope_params.get("rope_type", "default")
    rope_scaling = (
        None
        if rope_type in (None, "default")
        else {k: v for k, v in rope_params.items() if not isinstance(v, (list, dict))}
    )

    expert_quant, weight_block_size = _fp8_block_quant(hf_config)
    if expert_quant == "none":
        expert_quant = _expert_quant(hf_config)  # nvfp4 / mixed-precision modelopt
    # Dense attention/GDN quant is independent of the routed experts (block-fp8 already
    # quantizes both, so only probe for per-tensor FP8 when experts aren't block-fp8).
    attn_quant = "none" if expert_quant == "fp8_block" else _attn_quant(hf_config)
    # GDN projections mirror the full-attention quant for modelopt / block-fp8 checkpoints
    # (both quantize self_attn and GDN together); compressed-tensors checkpoints may differ
    # (self_attn NVFP4 but GDN out_proj dense bf16) and override this below.
    gdn_quant = attn_quant
    # NVFP4 checkpoints store the dense MLP projections (shared_expert; dense non-MoE MLP) as
    # packed FP4 exactly like the routed experts -- independent of whether attention is FP8
    # (mixed) or bf16 (pure NVFP4). Keep them native FP4 (W4A16) whenever the experts are
    # NVFP4. The lm_head is detected separately (only the mixed checkpoint quantizes it).
    # MoE-NVFP4 keeps the shared_expert dense MLP native FP4 (expert_quant=="nvfp4"); a dense
    # (non-MoE) modelopt checkpoint instead tags the bare .mlp.{gate,up,down}_proj as NVFP4.
    dense_quant = "nvfp4" if expert_quant == "nvfp4" else _dense_mlp_quant(hf_config)
    lm_head_quant = _lm_head_quant(hf_config)

    # compressed-tensors NVFP4: the quantization_config flag cannot express *which* modules
    # are NVFP4 (and carries a stale ``ignore`` list), so inspect the tensor layout. Dense
    # Qwen3.6-27B has attention + dense MLP as W4A16 NVFP4 (GDN in_proj_*/lm_head/norms bf16);
    # the AEON MoE Ornith build has only the MLP (routed experts + shared_expert) as NVFP4 with
    # attention/GDN bf16; the plain MoE Ornith build quantizes self_attn too but leaves GDN
    # out_proj dense bf16. Wire the W4A16 kernels per module accordingly (attn_quant=="nvfp4"
    # routes the full-attention linears through them; gdn_quant=="nvfp4" routes GDN out_proj).
    if _compressed_tensors_nvfp4(hf_config):
        m = _ct_nvfp4_module_map(_ct_checkpoint_path(hf_config) or "")
        if m["has_moe_experts_nvfp4"]:
            expert_quant = "nvfp4"  # MoE: routed experts are NVFP4 (offload cache)
        attn_quant = "nvfp4" if m["self_attn_nvfp4"] else "none"
        gdn_quant = "nvfp4" if m["gdn_outproj_nvfp4"] else "none"
        dense_quant = "nvfp4" if (m["shared_expert_nvfp4"] or m["dense_mlp_nvfp4"]) else "none"
        lm_head_quant = "none"

    # Dense variants (e.g. Qwen3.6-27B) report num_experts==0: route the decoder MLP through
    # the dense Qwen3_5DenseMLP instead of the MoE block.
    num_experts = getattr(text, "num_experts", 0) or 0
    moe_enabled = num_experts > 0

    layer_types = _layer_types(text)
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    linear_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")

    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=text.max_position_embeddings,
        base=rope_theta,
        scaling=rope_scaling,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=full_rotary,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=text.linear_num_key_heads,
        num_value_heads=text.linear_num_value_heads,
        key_head_dim=text.linear_key_head_dim,
        value_head_dim=text.linear_value_head_dim,
        conv_kernel_dim=text.linear_conv_kernel_dim,
        output_gate=True,
    )
    # Order groups by their first layer id for deterministic iteration.
    groups = tuple(
        sorted(
            (full_group, linear_group),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    return ModelConfig(
        num_layers=text.num_hidden_layers,
        num_qo_heads=text.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=getattr(text, "intermediate_size", 0),
        hidden_act=text.hidden_act,
        rms_norm_eps=text.rms_norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=full_rotary,
        num_experts=num_experts,
        num_experts_per_tok=getattr(text, "num_experts_per_tok", 0),
        moe_intermediate_size=getattr(text, "moe_intermediate_size", 0),
        shared_expert_intermediate_size=getattr(text, "shared_expert_intermediate_size", 0),
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", False)),
        moe_enabled=moe_enabled,
        use_qk_norm=True,
        model_type=getattr(hf_config, "model_type", "qwen3_5_moe"),
        architectures=getattr(hf_config, "architectures", ["Qwen3_5MoeForConditionalGeneration"]),
        vision_config=None,  # text-only milestone
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        expert_quant=expert_quant,
        weight_block_size=weight_block_size,
        attn_quant=attn_quant,
        gdn_quant=gdn_quant,
        dense_quant=dense_quant,
        lm_head_quant=lm_head_quant,
    )


__all__ = ["parse_config"]
