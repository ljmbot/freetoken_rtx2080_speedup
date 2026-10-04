"""Qwen3.5-MoE Multi-Token-Prediction (MTP / "nextn") head.

The head predicts token t+2 from (hidden_t, embed(token_{t+1})):

    e  = pre_fc_norm_embedding(embed(next_id))     # Gemma RMSNorm
    h  = pre_fc_norm_hidden(prev_hidden)           # Gemma RMSNorm
    x  = fc(cat([e, h], dim=-1))                    # Linear 2H -> H  (mtp.fc)
    x  = x + attn(input_layernorm(x))              # one full-attn block
    x  = x + mlp(post_attention_layernorm(x))      # routed + shared MoE
    x  = norm(x)                                    # Gemma RMSNorm   (mtp.norm)
    logits = lm_head(x)                             # SHARED lm_head

State-dict keys match the checkpoint's 19 ``mtp.*`` tensors exactly (see
``load_mtp_weights``). Embedding and lm_head are SHARED with the base model and
passed into ``forward``.

k=1 design (Phase 4): the head sees exactly ONE token per draft, so its inner
attention is self-contained (causal over the single new token) and needs NO
extra KV-pool layer. The 256 routed experts stay GPU-resident in nvfp4 and run
through the SAME fused Triton kernel the base model uses for decode
(``fused_experts_decode_nvfp4_marlin``), indexed by raw expert id -- no PCIe, no
contention with the base model's offload cache. This is what makes the head
~1 base-layer cost in production (the A2 harness's dequant-then-einsum MoE was
a 3.5x slowdown artifact).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, GemmaRMSNorm, LinearReplicated
from freetoken.layers.rotary import get_rope
from freetoken.moe.fused_nvfp4 import fused_experts_decode_nvfp4_marlin

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


# ---------------------------------------------------------------------------
# bf16 -> nvfp4 quantization (identical to the Phase 3 / A2-validated harness).
# Packed e2m1 codes [S, OUT, IN//2] uint8 + fp8-e4m3 per-16 block scales
# [S, OUT, IN//16] + fp16 per-output-row global scales [S, OUT].
# ---------------------------------------------------------------------------
_E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _e2m1_round(x: torch.Tensor) -> torch.Tensor:
    grid = _E2M1_GRID
    ax = x.abs()
    d = (ax.unsqueeze(-1) - grid).abs()
    dmin = d.min(-1, keepdim=True).values
    eq = (d <= dmin)
    even = (torch.arange(8, device=x.device) % 2 == 0).view(1, 1, 1, 8)
    cand = torch.where(eq & even, torch.arange(8, device=x.device).view(1, 1, 1, 8), 8)
    tie_even = cand.min(-1).values
    tie_any = torch.where(eq, torch.arange(8, device=x.device).view(1, 1, 1, 8), 8).min(-1).values
    idx = torch.where(tie_even <= 7, tie_even, tie_any)
    return torch.sign(x) * grid[idx]


def quantize_nvfp4(w_cpu: torch.Tensor, dev: torch.device, chunk: int = 32):
    """Quantize a [S, OUT, IN] bf16 expert stack to packed nvfp4 (on ``dev``)."""
    S, OUT, IN = w_cpu.shape
    assert IN % 16 == 0
    packed = torch.empty((S, OUT, IN // 2), dtype=torch.uint8)
    scale = torch.empty((S, OUT, IN // 16), dtype=torch.uint8)
    g16 = torch.empty((S, OUT), dtype=torch.float16)
    for s0 in range(0, S, chunk):
        s1 = min(s0 + chunk, S)
        wf = w_cpu[s0:s1].float()
        row_max = wf.abs().amax(dim=-1)
        g = (row_max / (448.0 * 6.0)).clamp_min(1e-12)
        wb = wf.view(s1 - s0, OUT, IN // 16, 16)
        bmax = wb.abs().amax(dim=-1)
        bs = (bmax / g.unsqueeze(-1)).clamp(max=448.0)
        bs_e4m3 = bs.to(torch.float8_e4m3fn)
        denorm = bs_e4m3.float().unsqueeze(-1) * g.view(s1 - s0, OUT, 1, 1)
        q = (wb / denorm).clamp(-6.0, 6.0)
        q = _e2m1_round(q).view(s1 - s0, OUT, IN)
        ax = q.abs()
        gidx = (ax.unsqueeze(-1) - _E2M1_GRID.to(ax.device)).abs().argmin(-1)
        code = (8 * (q < 0).to(torch.int32) + gidx).to(torch.uint8)
        lo = code[..., 0::2]
        hi = code[..., 1::2]
        packed[s0:s1] = (lo | (hi << 4))
        scale[s0:s1] = bs_e4m3.view(torch.uint8)
        g16[s0:s1] = g.to(torch.float16)
        del wf, wb, q
    return (
        packed.to(dev).contiguous(),
        scale.to(dev).contiguous().view(torch.float8_e4m3fn),
        g16.to(dev).contiguous(),
    )


# The 17 non-expert head tensors (the 2 expert tensors are quantized separately).
_MTP_OTHER_KEYS = [
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
]


def load_mtp_weights(
    head: Qwen3_5MTP, safetensors_path: str, dtype: torch.dtype, dev: torch.device
) -> None:
    """Load the 19 ``mtp.*`` tensors from a safetensors file into ``head``.

    The non-expert tensors load straight into the module's weights (GemmaRMSNorm
    weights bake the +1.0, since GemmaRMSNorm scales by the raw stored weight);
    the two expert tensors are quantized bf16 -> nvfp4 (GPU-resident on ``dev``)
    and stored on ``head`` for the fused decode kernel. ``dev`` is passed
    explicitly because ``BaseOP`` allocates weights on CPU.
    """
    from safetensors.torch import load_file

    W = load_file(safetensors_path)

    def put_norm(module, wname):
        module.weight.data = (W[wname].float() + 1.0).to(dtype).to(dev)

    def put_linear(module, wname):
        module.weight.data = W[wname].to(dtype).to(dev).contiguous()

    # Entry norms + fc.
    put_norm(head.pre_fc_norm_embedding, "mtp.pre_fc_norm_embedding.weight")
    put_norm(head.pre_fc_norm_hidden, "mtp.pre_fc_norm_hidden.weight")
    put_linear(head.fc, "mtp.fc.weight")
    # Block norms.
    put_norm(head.input_layernorm, "mtp.layers.0.input_layernorm.weight")
    put_norm(head.post_attention_layernorm, "mtp.layers.0.post_attention_layernorm.weight")
    # Attention.
    put_linear(head.q_proj, "mtp.layers.0.self_attn.q_proj.weight")
    put_linear(head.k_proj, "mtp.layers.0.self_attn.k_proj.weight")
    put_linear(head.v_proj, "mtp.layers.0.self_attn.v_proj.weight")
    put_linear(head.o_proj, "mtp.layers.0.self_attn.o_proj.weight")
    put_norm(head.q_norm, "mtp.layers.0.self_attn.q_norm.weight")
    put_norm(head.k_norm, "mtp.layers.0.self_attn.k_norm.weight")
    # Routed router + shared expert.
    put_linear(head.gate, "mtp.layers.0.mlp.gate.weight")
    put_linear(head.sh_gate, "mtp.layers.0.mlp.shared_expert_gate.weight")
    put_linear(head.sh_gate_proj, "mtp.layers.0.mlp.shared_expert.gate_proj.weight")
    put_linear(head.sh_up_proj, "mtp.layers.0.mlp.shared_expert.up_proj.weight")
    put_linear(head.sh_down_proj, "mtp.layers.0.mlp.shared_expert.down_proj.weight")
    # Exit norm.
    put_norm(head.norm, "mtp.norm.weight")
    # Experts: bf16 -> nvfp4, GPU-resident.
    head.gu_packed, head.gu_scale, head.gu_global = quantize_nvfp4(
        W["mtp.layers.0.mlp.experts.gate_up_proj"], dev
    )
    head.dn_packed, head.dn_scale, head.dn_global = quantize_nvfp4(
        W["mtp.layers.0.mlp.experts.down_proj"], dev
    )


class Qwen3_5MTP(BaseOP):
    """The nextn / MTP head. One full-attention + MoE block plus the entry/exit
    norms and the 2H->H fc. k=1: self-contained attention (no KV pool)."""

    def __init__(self, config: "ModelConfig"):
        super().__init__()
        h = config.hidden_size
        eps = config.rms_norm_eps
        self.hidden_size = h
        self.num_q = config.num_qo_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = config.head_dim
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.moe_inter = config.moe_intermediate_size

        # Entry: norm the embedding and the previous hidden, project 2H -> H.
        self.pre_fc_norm_embedding = GemmaRMSNorm(h, eps=eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(h, eps=eps)
        self.fc = LinearReplicated(2 * h, h, has_bias=False)

        # Block norms (Qwen3.5 style: plain forward, residual added in forward).
        self.input_layernorm = GemmaRMSNorm(h, eps=eps)
        self.post_attention_layernorm = GemmaRMSNorm(h, eps=eps)

        # Full attention (gated: q_proj emits [q; gate]).
        hd = self.head_dim
        self.q_proj = LinearReplicated(h, self.num_q * 2 * hd, has_bias=False)
        self.k_proj = LinearReplicated(h, self.num_kv * hd, has_bias=False)
        self.v_proj = LinearReplicated(h, self.num_kv * hd, has_bias=False)
        self.o_proj = LinearReplicated(self.num_q * hd, h, has_bias=False)
        self.q_norm = GemmaRMSNorm(hd, eps=eps)
        self.k_norm = GemmaRMSNorm(hd, eps=eps)
        self.rotary = get_rope(
            head_dim=hd,
            rotary_dim=config.rotary_config.rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
            rope_scaling=(
                tuple(config.rotary_config.scaling.items())
                if config.rotary_config.scaling
                else None
            ),
        )

        # Routed MoE (nvfp4, GPU-resident) + shared expert (bf16 dense).
        self.gate = LinearReplicated(h, self.num_experts, has_bias=False)
        self.sh_gate = LinearReplicated(h, 1, has_bias=False)
        self.sh_gate_proj = LinearReplicated(h, self.moe_inter, has_bias=False)
        self.sh_up_proj = LinearReplicated(h, self.moe_inter, has_bias=False)
        self.sh_down_proj = LinearReplicated(self.moe_inter, h, has_bias=False)
        # nvfp4 expert banks, populated by load_mtp_weights ([num_experts, ...]).
        self.gu_packed: torch.Tensor | None = None
        self.gu_scale: torch.Tensor | None = None
        self.gu_global: torch.Tensor | None = None
        self.dn_packed: torch.Tensor | None = None
        self.dn_scale: torch.Tensor | None = None
        self.dn_global: torch.Tensor | None = None

        # Exit norm.
        self.norm = GemmaRMSNorm(h, eps=eps)

    # ------------------------------------------------------------------
    def _attn(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Gated full attention, self-contained (causal over the new tokens).
        Mirrors the A2-validated harness head_attn exactly."""
        L = x.shape[0]
        nq, nkv, hd = self.num_q, self.num_kv, self.head_dim
        qg = self.q_proj.forward(x).view(L, nq, 2 * hd)
        q = qg[..., :hd].contiguous()
        gate = qg[..., hd:].reshape(L, nq * hd)
        k = self.k_proj.forward(x).view(L, nkv, hd).contiguous()
        v = self.v_proj.forward(x).view(L, nkv, hd).contiguous()
        q = self.q_norm.forward(q)
        k = self.k_norm.forward(k)
        qf, kf = self.rotary.forward(
            positions, q.reshape(L, nq * hd), k.reshape(L, nkv * hd)
        )
        q = qf.view(L, nq, hd)
        k = kf.view(L, nkv, hd)
        rep = nq // nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            is_causal=True,
        )
        o = o.squeeze(0).transpose(0, 1).reshape(L, nq * hd)
        o = o * torch.sigmoid(gate)
        return self.o_proj.forward(o)

    def _moe(self, x: torch.Tensor) -> torch.Tensor:
        """Routed nvfp4 experts (fused decode kernel) + shared expert."""
        L = x.shape[0]
        logits = self.gate.forward(x)
        probs = torch.softmax(logits.float(), dim=-1)
        tw, ti = torch.topk(probs, self.top_k, dim=-1)
        tw = (tw / tw.sum(-1, keepdim=True)).to(x.dtype)
        ti = ti.to(torch.int32)
        routed = fused_experts_decode_nvfp4_marlin(
            x,
            self.gu_packed, self.gu_scale, self.gu_global,
            self.dn_packed, self.dn_scale, self.dn_global,
            tw, ti,
            activation="silu",
            apply_router_weight_on_input=False,
        )
        # Shared expert (dense bf16): silu(gate) * up -> down, gated by sigmoid.
        sa = F.silu(self.sh_gate_proj.forward(x)) * self.sh_up_proj.forward(x)
        shared = self.sh_down_proj.forward(sa)
        shared = shared * torch.sigmoid(self.sh_gate.forward(x))
        return routed + shared

    def forward(
        self,
        prev_hidden: torch.Tensor,
        next_ids: torch.Tensor,
        positions: torch.Tensor,
        embed_tokens,
        lm_head,
    ) -> torch.Tensor:
        """logits for the token AFTER ``next_ids`` given the base model's
        ``prev_hidden`` (pre-final-norm hidden at the position that produced
        ``prev_hidden``). k=1: ``prev_hidden`` is [1, H], ``next_ids`` is [1].
        ``embed_tokens``/``lm_head`` may be callables or modules (``.forward``)."""
        emb = embed_tokens if callable(embed_tokens) else embed_tokens.forward
        lm = lm_head if callable(lm_head) else lm_head.forward
        e = self.pre_fc_norm_embedding.forward(emb(next_ids))
        h = self.pre_fc_norm_hidden.forward(prev_hidden)
        x = self.fc.forward(torch.cat([e, h], dim=-1))
        x = x + self._attn(self.input_layernorm.forward(x), positions)
        x = x + self._moe(self.post_attention_layernorm.forward(x))
        x = self.norm.forward(x)
        return lm(x)


__all__ = ["Qwen3_5MTP"]
