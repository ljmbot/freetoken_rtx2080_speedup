# ft_ornith_plan.md — Serve `Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4` on FreeToken

Status: **PLAN (not yet implemented).** Everything below is grounded in the live checkpoint
(`~/Models/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4`, 23.7 GB single-file
`model.safetensors`) and the current loader source. **Reviewed for accuracy 2026-10-07**: all
file:line references, the tensor inventory, the detection root-cause, and the bank-reader
convention were re-verified against the live code + checkpoint. Run `parse_config` on the real
config as the acceptance gate for Step 1.

## TL;DR

- **Architecture: fully supported.** Ornith is byte-for-byte the same `Qwen3_5MoeForConditionalGeneration`
  class as `Qwen3.6-35B-A3B-NVFP4`: 40 layers (30 linear-attention GDN + 10 full-attention),
  256 routed experts / top-8, `hidden=2048`, `moe_intermediate=512`, all 40 layers MoE
  (no dense prefix). The model graph, kernels, and offload machinery already exist.
- **Quantization: NOT yet supported.** Ornith is a **compressed-tensors** NVFP4 build
  (`quant_method: compressed-tensors`, `format: nvfp4-pack-quantized`), and — unlike Qwen3.6's
  modelopt build — **only the MLP is NVFP4; attention/GDN/router/norms/lm_head are BF16.**
  The current loader mis-detects this and will build the wrong layer types and drop the
  routed experts.
- **Six bounded gaps** must be closed (detection ×2, dense-pass expert skip, dense-pass
  fusions — shared-expert gate|up *and* attention q/k/v, expert-bank reader naming +
  reciprocal-global, missing index file). Estimated **1–2 focused days.**
- **VRAM is tight but feasible** on the 8 GB RTX 2080 (byte-accurate, vision off): dense
  (resident) weights are **~4.4 GiB (4.7 GB)** — bf16 attention/GDN + untied bf16
  embed/lm_head at 248K vocab — and the expert cache is NVFP4 at **1.693 MB/expert**. At
  64K with 8-bit KV (0.63 GiB over the 10 full-attn layers): **C=512 experts → ~5.9 GiB total
  (~1.7 GiB headroom)**, C=1000 → ~6.7 GiB (~0.9 GiB), C=1350 → ~7.3 GiB (~0.3 GiB, ceiling).
  Start from the Qwen3.6 prod config (`qwen_q8.sh`) and tune `--moe-cache-size` / context down
  if OOM. Serve **text-only** (`FREETOKEN_LOAD_VISION=0`, the default) — it drops the 333
  vision tensors (**0.83 GiB** saved).

---

## 1. What the model actually is (verified from the safetensors header)

Total tensors: **93,346**. Dtype split by module:

| module | tensors | dtype | notes |
|---|---|---|---|
| routed experts | 92,160 | U8 / F8_E4M3 / F32 | 256 experts × 40 layers × 3 proj × 3 kinds (`weight_packed`/`weight_scale`/`weight_global_scale`) |
| shared_expert | 360 | U8 / F8_E4M3 / F32 | 40 layers × 3 proj × 3 kinds — **NVFP4** |
| linear_attn (GDN) | 270 | **BF16** | in_proj_{qkv,z,b,a}, out_proj, conv1d, A_log, dt_bias, norm |
| self_attn | 60 | **BF16** | q/k/v/o + q_norm/k_norm (10 full-attn layers) |
| norm | 81 | BF16 | |
| router_gate (`mlp.gate`) | 40 | BF16 | 256×2048 |
| shared_gate (`mlp.shared_expert_gate`) | 40 | BF16 | 1×2048 |
| vision | 333 | BF16 | multimodal (image/video) — **dropped when text-only** |
| embed / lm_head | 2 | BF16 | |

Per-expert NVFP4 layout (layer 0, expert 0, `moe_intermediate I=512`, `hidden H=2048`):

```
gate_proj.weight_packed   U8       [512, 1024]   = [I, H//2]
gate_proj.weight_scale    F8_E4M3  [512, 128]    = [I, H//16]
gate_proj.weight_global_scale F32  [1]           (quant-side scalar)
up_proj.*                 (same as gate_proj)
down_proj.weight_packed   U8       [2048, 256]   = [H, I//2]
down_proj.weight_scale    F8_E4M3  [2048, 32]    = [H, I//16]
down_proj.weight_global_scale F32  [1]
```

This is the **exact same NVFP4 geometry** the Qwen3.6 modelopt experts use — only the tensor
*naming* differs (compressed-tensors vs modelopt).

### ⚠️ The config's `ignore` list is stale/misleading

`quantization_config.ignore` lists **10,240** `...mlp.experts.<e>` entries (implying the
routed experts are *unquantized*). The safetensors header proves otherwise: **all 92,160
expert tensors are NVFP4.** Do **not** trust the `ignore` list — the ground truth is the
tensor layout. (This is why detection must inspect the header, not the config.)

### ⚠️ No `model.safetensors.index.json`

Ornith ships a **single** `model.safetensors` (23.7 GB) with **no** index file. The NVFP4
expert-bank reader (`models/nvfp4_banks.py:90`) hard-requires the index and will
`FileNotFoundError`. The dense pass (`models/loader.py:ShardReader`) already falls back to
scanning the single file, so only the bank reader is affected — but the clean fix is to
generate the standard HF index (Step 5).

---

## 2. What `parse_config` currently produces on the real Ornith config (the smoking gun)

Ran `parse_config(cached_load_hf_config(<Ornith>))` on the live checkpoint:

```
compressed_tensors_nvfp4 detected: True
num_layers=40  num_experts=256  num_experts_per_tok=8  moe_intermediate=512  hidden=2048
num_moe_layers=40  moe_enabled=True
expert_quant  = none        ❌  should be "nvfp4"   (256 routed NVFP4 experts exist)
attn_quant    = nvfp4       ❌  should be "none"    (attention/GDN are BF16, not NVFP4)
dense_quant   = nvfp4       ✅  (shared_expert IS NVFP4)
lm_head_quant = none        ✅
weight_block_size = None
```

Two detection bugs, both in `python/freetoken/models/qwen3_5_moe/config.py`:

- **`expert_quant="none"`** — `_expert_quant()` (config.py:42) reads
  `quant_algo or quant_method`. For Ornith that's `None or "compressed-tensors"` →
  `"compressed-tensors"`, which contains no `"fp4"` → returns `"none"`. Consequence: the MoE
  offload path routes experts to the **bf16 provider** (`_PROVIDERS["none"]`), and the model
  builds bf16 expert buffers — the routed experts are effectively dropped / wrong.
- **`attn_quant="nvfp4"`** — the blanket block at config.py:185-188
  (`if _compressed_tensors_nvfp4: attn_quant="nvfp4"; dense_quant="nvfp4"`) was written for the
  **dense** Qwen3.6-27B (where attention *is* NVFP4). For Ornith it wrongly marks
  attention/GDN as NVFP4, so the model instantiates **W4A16 NVFP4 attention linears** that the
  loader will feed **BF16** tensors to → load mismatch / silent corruption.

**Root cause:** detection is driven by the `quantization_config` flag, which (a) can't express
"only the MLP is NVFP4, attention is BF16" and (b) carries a stale `ignore` list. The fix is to
**inspect the actual tensor layout** to decide, per module, NVFP4 vs BF16.

---

## 3. The six gaps (exact locations)

| # | gap | file:line | severity |
|---|---|---|---|
| 1 | `expert_quant` mis-detected as `"none"` (should be `"nvfp4"`) | `config.py:42` `_expert_quant`, `config.py:167-169` | **root cause** |
| 2 | `attn_quant` blanket-set to `"nvfp4"` (should be `"none"` — attn is BF16) | `config.py:185-188` | **root cause** |
| 3 | Dense pass does **not skip routed experts** → per-expert `weight_packed` is processed as a dense weight → unexpected-key error | `weight.py:565` `_iter_weights_compressed_tensors` (no `.mlp.experts.<int>.` skip; cf. modelopt's `weight.py:236`) | high |
| 4 | Dense-pass **fusions missing** in the CT path: (a) shared-expert gate\|up → `gate_up_proj` (NVFP4 path); (b) attention q/k/v → `qkv_proj` (BF16 path — Ornith's attention is BF16, so it's yielded as separate `q_proj`/`k_proj`/`v_proj`, but the model's `qkv_proj` is a merged buffer) | `weight.py:544` `_CT_NVFP4_FUSE` (no shared_expert group), `weight.py:548` `_CT_BF16_FUSE` (no `.self_attn.qkv_proj` group) | high |
| 5 | NVFP4 expert-bank reader only recognizes **modelopt** naming (`weight`/`weight_scale`/`weight_scale_2`), not compressed-tensors (`weight_packed`/`weight_scale`/`weight_global_scale`), and stores the global raw (CT needs the **reciprocal**) | `weight.py:40` `_NVFP4_EXPERT_KEY_RE`, `weight.py:44` `_NVFP4_SOURCE_SPEC`, `nvfp4_banks.py:104-121,159-180` (kind handling) | high |
| 6 | No `model.safetensors.index.json` → bank reader `FileNotFoundError` | checkpoint packaging / `nvfp4_banks.py:90` | high |

Gaps 1–2 are the *detection* root cause; 3–6 are the *loading* work that becomes reachable
once detection is correct.

---

## 4. Implementation plan

Order matters: do **5 (index)** and **1–2 (detection)** first so you can iterate with a real
`parse_config` gate, then **3–4 (dense pass)**, then **6 (bank reader)** — the bank reader is
the largest piece and the last to be exercised end-to-end.

### Step 0 — Generate the HF index (one-time, unblocks everything)

Write a small script that reads the single `model.safetensors` header and emits the standard
`model.safetensors.index.json` (`{"metadata": {"total_size": N}, "weight_map": {name: "model.safetensors"}}`).
This is the HF-standard artifact, helps every reader, and is harmless to the dense pass.
(Alternative: teach `nvfp4_banks.py` to fall back to single-file mode like `ShardReader` does —
more robust, but more code. Prefer generating the index.)

### Step 1 — Fix detection (`config.py`)

Add a header-inspecting helper and use it for compressed-tensors checkpoints. Sketch:

```python
def _ct_nvfp4_module_map(hf_config, model_path) -> dict:
    """Inspect the safetensors header: which modules are NVFP4 (weight_packed) vs BF16.
    Returns {"has_moe_experts_nvfp4": bool, "attn_nvfp4": bool, "shared_expert_nvfp4": bool}."""
    # read model.safetensors.index.json (or the single file header) weight_map;
    # has_moe_experts_nvfp4 = any key ends with ".mlp.experts.<int>.<proj>.weight_packed"
    # attn_nvfp4            = any key ends with ".self_attn.q_proj.weight_packed" (or GDN out_proj)
    # shared_expert_nvfp4   = any key ends with ".mlp.shared_expert.gate_proj.weight_packed"
```

Then in `parse_config`, replace the blanket block (config.py:185-188) with a layout-driven one:

```python
if _compressed_tensors_nvfp4(hf_config):
    m = _ct_nvfp4_module_map(hf_config, model_path)
    expert_quant  = "nvfp4" if m["has_moe_experts_nvfp4"] else expert_quant   # MoE: nvfp4
    attn_quant    = "nvfp4" if m["attn_nvfp4"] else "none"                     # Ornith: none
    dense_quant   = "nvfp4" if (m["shared_expert_nvfp4"] or expert_quant == "nvfp4") else dense_quant
    lm_head_quant = "none"
```

**Regression guard:** the dense Qwen3.6-27B-NVFP4 (where attention *is* NVFP4) must still
resolve to `attn_quant="nvfp4"`. Add a unit test asserting both checkpoints parse to the
expected `(expert_quant, attn_quant, dense_quant)`.

### Step 2 — Dense pass: skip routed experts (`weight.py`)

In `_iter_weights_compressed_tensors` (weight.py:607 loop), add a skip mirroring the modelopt
path's `weight.py:236`:

```python
if re.search(r"\.mlp\.experts\.\d+\.", raw_name):
    continue  # routed experts -> NVFP4 bank reader (offload path)
```

(Production uses `--moe-backend offload`, so experts are loaded by the bank reader, not the
dense pass. The resident path `include_moe_experts=True` is out of scope for NVFP4.)

### Step 3 — Dense-pass fusions (`weight.py`)

Two fusions are missing in the CT path:

**(a) Shared-expert gate|up → `gate_up_proj` (NVFP4 path).** Add a shared-expert group to
`_CT_NVFP4_FUSE` (weight.py:544) so the NVFP4 shared expert fuses to the `gate_up_proj`
buffer the model expects (`_SharedExpert` → `Nvfp4DenseColMerged`):

```python
_CT_NVFP4_FUSE = {
    ".self_attn.qkv_proj": (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"),
    ".mlp.gate_up_proj": (".mlp.gate_proj", ".mlp.up_proj"),
    ".mlp.shared_expert.gate_up_proj": (".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj"),
}
```

**(b) Attention q/k/v → `qkv_proj` (BF16 path).** Ornith's attention is BF16 (not NVFP4),
so its q/k/v take the plain-bf16 path (`_emit_bf16_weight` → `_ct_bf16_fuse` with
`_CT_BF16_FUSE`), which currently only fuses the GDN `in_proj`. The model's
`Qwen3_5Attention.qkv_proj` is a **merged** buffer (`make_col_merged`), so the bf16 path
must also fuse q/k/v. Add to `_CT_BF16_FUSE` (weight.py:548):

```python
_CT_BF16_FUSE = {
    ".linear_attn.in_proj": (
        ".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z",
        ".linear_attn.in_proj_b", ".linear_attn.in_proj_a",
    ),
    ".self_attn.qkv_proj": (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"),
}
```

`down_proj` stays standalone (`Nvfp4DenseLinear`). The `endswith` matching is unambiguous:
`...shared_expert.gate_proj` matches only the new NVFP4 group, not the dense `.mlp.gate_proj`
group; the bf16 qkv group only matches `...self_attn.{q,k,v}_proj`. Adding qkv to the bf16
table is safe for dense Qwen3.6-27B (its attention is NVFP4, so it never reaches the bf16 path).

### Step 4 — Generalize the NVFP4 expert-bank reader (`nvfp4_banks.py` + `weight.py`)

The reader currently hardcodes modelopt kinds. Generalize via the spec so **both** the serial
(`load_nvfp4_expert_source_banks`) and parallel (`..._parallel`) readers work for either naming.

Add to `Nvfp4ExpertSourceSpec` (nvfp4_banks.py:20):

```python
@dataclass(frozen=True)
class Nvfp4ExpertSourceSpec:
    key_pattern: re.Pattern[str]
    proj_to_role: dict[str, str]
    layer_to_bank: LayerToBank
    desc: str
    kind_map: dict[str, str] = field(default_factory=lambda: {
        "weight": "packed", "weight_scale": "scale", "weight_scale_2": "global"})
    global_is_quant_side: bool = False   # True for compressed-tensors (store 1/stored)
```

Define the CT spec in `weight.py` next to `_NVFP4_SOURCE_SPEC` (weight.py:44):

```python
_NVFP4_CT_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight_packed|weight_scale|weight_global_scale)$")
_NVFP4_CT_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_NVFP4_CT_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,   # all 40 layers are MoE
    desc="Ornith compressed-tensors NVFP4 experts",
    kind_map={"weight_packed": "packed", "weight_scale": "scale", "weight_global_scale": "global"},
    global_is_quant_side=True,
)
```

In the reader, replace the hardcoded kind branches (nvfp4_banks.py:116-121, 159-180) with
`role = spec.kind_map[kind]` (raising on an unknown kind), and apply the global convention:

```python
g = f.get_tensor(name)
g = (1.0 / g).to(torch.float16) if spec.global_is_quant_side else g.to(torch.float16)
```

The bank **layout is unchanged** — the CT shapes map exactly onto the existing banks
(`gate_up_packed [E,2I,H//2]`, `gate_up_scale [E,2I,H//16]`, `gate_up_global [E,2I]`,
`down_packed [E,H,I//2]`, `down_scale [E,H,I//16]`, `down_global [E,H]`). Only the *source
naming* and the *global reciprocal* differ.

Finally, select the spec by checkpoint type in `load_nvfp4_expert_sources` /
`load_nvfp4_expert_sources_parallel` (weight.py:1056, 1071): use `_NVFP4_CT_SOURCE_SPEC` when
`_compressed_tensors_nvfp4(hf_config)` else `_NVFP4_SOURCE_SPEC`.

### Step 5 — Wire `expert_quant` → provider (no change needed)

Once `expert_quant="nvfp4"`, the offload path already routes to `_PROVIDERS["nvfp4"]`
(`moe/expert_banks.py:299`) → `_nvfp4_banks` → `load_nvfp4_moe_expert_sources` → the model's
`load_nvfp4_expert_sources(_parallel)` → the spec from Step 4. No provider change required.

### Step 6 — Serving config (`ornith_q8.sh`)

Clone `qwen_q8.sh`, point `--model` at Ornith, and keep the prod tuning. Expected deltas:

```bash
--model /home/lmeek/Models/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4
--served-model-name Ornith-1.0-35B-AEON-NVFP4
--dtype float16
--attention-backend cutlass
--moe-backend offload
--nvfp4-backend triton
--expert-load serial
--moe-cache-size 1350        # start here; drop to ~1000-1150 if OOM (bf16 attn is denser than Qwen3.6's fp8)
--num-tokens 64000 --max-seq-len-override 64000
--max-prefill-length 4096 --max-output-tokens 4096
--max-running-requests 1 --cuda-graph-max-bs 1
--cache-type radix --enable-cache-report --kv-q8
--sampling-defaults model --temperature 0.5 --reasoning-parser auto
# FREETOKEN_LOAD_VISION=0 (default) — text-only, drops the 333 vision tensors
```

VRAM estimate (resident/dense, excl. offloaded experts & vision): full-attn ~0.55 GB +
GDN ~2.0 GB + router ~0.04 GB + shared_expert (NVFP4) ~0.07 GB + embed/lm_head (bf16,
**untied**, 248K vocab) ~2.03 GB + norms ≈ **~4.4 GiB (4.7 GB)**. Adding the expert cache
C=1350 (NVFP4, ~2.23 GiB) brings the pre-KV total to ~6.6 GiB, leaving only **~1.0 GiB**
for KV + activations on an 8 GB card (0.95 usable = 7.6 GiB). So C=1350 + 64K context is
**tight** — expect to lower `--moe-cache-size` and/or `--num-tokens` first (the user prefers
more resident experts, so trade context down first).

---

## 5. Verification plan ("check it is working correctly")

Run these in order; each has a hard pass/fail gate.

1. **Detection gate (Step 1).** `parse_config` on Ornith asserts
   `expert_quant=="nvfp4"`, `attn_quant=="none"`, `dense_quant=="nvfp4"`,
   `lm_head_quant=="none"`, `num_experts==256`, `moe_enabled==True`.
   **Regression:** Qwen3.6-35B-A3B-NVFP4 (modelopt) and Qwen3.6-27B-NVFP4 (dense CT) parse to
   their *previous* values — the detection change must not move them.

2. **Tensor inventory (ground truth).** Re-run the header scan: assert 92,160 expert tensors
   (256×40×3×3), 360 shared_expert, correct dtypes, and that **no** attention/GDN/router tensor
   is `weight_packed` (all BF16). (Already done — this is the reference the code must match.)

3. **Bank-reader unit test (Step 4).** Build the NVFP4 expert source banks for layer 0 (and one
   full-attn layer) from the real checkpoint. Assert bank shapes
   (`gate_up_packed [256,1024,1024]` uint8, `gate_up_scale [256,1024,128]` fp8,
   `gate_up_global [256,1024]` fp16, `down_*` analogues) and `placed == 40*256*6` for a full load.

4. **Dequant correctness (the critical one).** Take one expert's `gate_proj`, dequant with the
   engine kernel (`dequant_nvfp4`) to bf16, and compare against an independent reference
   (manually unpack the FP4 nibbles × fp8 block-scale × **reciprocal** global). This verifies
   the packing order, the block-scale layout, and the `1/weight_global_scale` convention.
   Assert finite, non-zero, and matching to bf16 tolerance. **A wrong global convention here is
   the most likely silent-corruption bug — this test catches it.**

5. **Full-load smoke test.** `ft serve --model <Ornith>` (text-only). Gate: clean load with
   **no missing/unexpected state-dict keys**, no OOM, and a short completion that produces
   coherent (not garbage/repeating) text. Watch the CUTLASS guard log for bails.

6. **Reasoning-quality check (the user's actual goal).** Re-run the π-100-digit task
   (`pi_repro.py`) against Ornith at the new default `temperature 0.5`, plus 2–3 harder
   reasoning prompts. Compare convergence rate to Qwen3.6 (which was 1/~9 on π). The claim is
   "higher precision reasoning" — measure whether Ornith converges more reliably, not just
   that it loads.

7. **End-to-end regression.** Qwen3.6-35B-A3B-NVFP4 still serves and produces the same output
   on a fixed prompt after the `config.py` / `weight.py` / `nvfp4_banks.py` changes.

Artifacts: put test scripts in `/home/lmeek/.hermes/cache/scratch/q8run/` (extend
`inspect_ornith.py`, `cmp_expert_layout.py`; add `test_ornith_detect.py`,
`test_ornith_banks.py`, `test_ornith_dequant.py`, `serve_ornith.sh`).

---

## 6. Risks / open questions

- **VRAM headroom is the main practical risk.** bf16 attention is denser than Qwen3.6's fp8.
  If 64K context + C=1350 OOMs, lower `--moe-cache-size` (fewer resident experts) and/or
  `--num-tokens`. The user prefers more resident experts over context, so trade context down first.
- **The `ignore` list is a trap.** Any future detection that reads `quantization_config.ignore`
  will wrongly conclude the experts are unquantized. Detection must inspect the header.
- **Multimodal is out of scope.** Ornith has a vision tower (333 tensors, image/video token ids).
  Serve text-only (`FREETOKEN_LOAD_VISION=0`). Vision support would be a separate effort.
- **Global-scale convention.** The CT `weight_global_scale` is the *quant-side* scale; the bank
  must store its reciprocal. This is the single most likely silent-corruption source — Step 5
  (dequant test) is mandatory, not optional.
- **Resident (non-offload) expert path** for NVFP4 is not covered; production uses offload.
  If a resident path is ever needed, it's additional work.

## 7. Effort estimate

| step | est |
|---|---|
| 0 — generate index | ~0.5 h |
| 1 — detection + regression test | ~2 h |
| 2 — dense-pass expert skip | ~0.25 h |
| 3 — dense-pass fusions (shared-expert + attention qkv) | ~0.75 h |
| 4 — bank-reader generalization + dequant test | ~4 h (largest) |
| 6 — serving config + VRAM tuning | ~1 h |
| verification (smoke, π re-run, regression) | ~3 h |
| **total** | **~1–2 focused days** |
