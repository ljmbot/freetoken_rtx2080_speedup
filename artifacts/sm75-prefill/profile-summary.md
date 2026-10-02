# Phase B — Prefill Profile Summary

**Status: GATE PASSED.** The dominant cost is identified, root-caused to a specific
codegen defect, and the next experiment is chosen. This document is the required
gate before any large kernel work (plan §Phase B).

- Date: 2026-10-02
- Machine: RTX 2080, 8 GB, **SM75 (Turing)**
- Project: `~/src/freetoken-sm75-native-prefill`, branch `perf/sm75-prefill`
- Baseline commit: `70a7888` (tag `baseline`); original tree untouched
- Server: loopback:1920, baseline flags (`--max-prefill-length 2048 --moe-backend
  offload --memory-ratio 0.95 --moe-cache-size 1500 --cache-type radix`)

---

## 1. What was profiled

The engine runs in a spawned subprocess, so profiling is armed at launch via env
(`FREETOKEN_PROFILE=1`, `FREETOKEN_PROFILE_AFTER=3`, `FREETOKEN_PROFILE_COUNT=2`,
`FREETOKEN_PROFILE_OUT=artifacts/sm75-prefill/profile`). The engine hook
(`maybe_profile_prefill` in `engine/engine.py`) wraps the eager prefill forward in
`torch.profiler` (CPU+CUDA) and exports a chrome trace + a per-op device-time table.

Two **uncached** prefills were captured (each request used a unique token sequence
so the radix cache could not match — verified `cached_tokens: null`):

| capture | tokens | elapsed | notes |
|---|---|---|---|
| `prefill-04-n523` | 523 (full) | 10,903 ms | non-chunked, single forward |
| `prefill-05-n2048` | 2048 (chunk) | 36,637 ms | dominant 2048 chunk of the 2058-token prefill-2048 workload |

The first 3 prefill forwards were skipped (`AFTER=3`) so Triton autotune / cuBLAS
heuristics for the 512 and 2048 shapes were warm — the captured traces reflect
steady-state prefill, not one-off compilation.

### Runtime backend selection (from `backend-report.json`)

| knob | value |
|---|---|
| GPU | RTX 2080, cap (7,5), 7784 MiB |
| dtype | float16 |
| attention_backend | **triton** |
| moe_backend | **offload** (NVFP4 experts, CPU→GPU H2D overlap) |
| nvfp4_backend | triton |
| moe_prefill_overlap | true |
| cache_type | hybrid_radix |
| max_extend_tokens | 2048 |
| CUDA graphs | enabled for decode only — **prefill is always eager** (so every prefill kernel is visible) |
| MoE router | **pure-torch fallback** (`triton_kernels` not installed) |

---

## 2. The evidence — per-op device time

### prefill-2048 (36,637 ms elapsed; total device time 41,574 ms across 2616 kernels)

| op | count | device ms | % of elapsed |
|---|---|---|---|
| **`_gemm_kernel`** (W8A16 dense-proj GEMM) | 80 | **27,936** | **76.5%** |
| `_prefill_nvfp4_moe_kernel` (NVFP4 routed experts) | 80 | 6,355 | 17.4% |
| `aten::copy_` | 745 | 1,705 | 4.7% |
| `ChunkGatedDeltaRuleFunction` (GDN recurrence) | 30 | 1,672 | 4.6% |
| `Memcpy HtoD (Pinned→Device)` (MoE offload H2D) | 240 | 1,665 | 4.5% |
| `chunk_fwd_kernel_o` (GDN chunk output) | 30 | 1,136 | 3.1% |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 30 | 453 | 1.2% |
| `_extend_attention_split_kernel` (full attention) | 10 | 410 | 1.1% |
| everything else (norms, topk, conv, act, …) | — | ~1,240 | 3.4% |

### prefill-512 (10,903 ms elapsed)

| op | count | device ms | % of elapsed |
|---|---|---|---|
| **`_gemm_kernel`** | 80 | **7,818** | **71.7%** |
| `_prefill_nvfp4_moe_kernel` | 80 | 2,351 | 21.6% |
| (rest) | — | ~734 | 6.7% |

**The same kernel dominates at both sizes: ~72–76% of prefill time is
`_gemm_kernel`.** It scales with token count (3.57× for 3.92× tokens), i.e. it is
the bulk of the actual prefill compute, not fixed overhead.

### What `_gemm_kernel` computes

It is the **W8A16 dense-projection GEMM** in
`kernel/triton/fp8_pertensor_linear.py`. The modelopt MIXED_PRECISION checkpoint
stores the *dense* projections (attention qkv/o, GDN in_proj_qkvz/out_proj) as
per-tensor FP8 (e4m3); only the routed experts are NVFP4. On SM75 (no native FP8)
these run through `_gemm_kernel`, which e4m3-decodes the FP8 weight in-kernel and
accumulates with `tl.dot`.

The 80 GEMM launches at M=2048 decompose (from grid shapes) to exactly the dense
projections:

| projection | shape [M,K]×[N,K] | launches | GFLOP each |
|---|---|---|---|
| out_proj (all 40 layers) | 2048×2048 | 40 | 17.2 |
| GDN in_proj_qkvz (30 GDN layers) | 2048×2048, N=12288 | 30 | 103.1 |
| full-attn qkv (10 attn layers) | 2048×2048, N=9216 | 10 | 77.3 |

Total dense GEMM work = **4.55 TFLOP**, and it takes **27.9 s** → an effective
**0.16 TFLOPS**.

---

## 3. Root cause — `tl.dot` is NOT using tensor cores on SM75

I isolated the GEMM out of the model and A/B'd three implementations of the
*identical* [2048×2048]×[12288×2048] GEMM (the dominant GDN in_proj shape), with
consumed outputs and a correctness check (Triton vs cuBLAS max_rel = 0.006, OK):

| impl | ms | TFLOPS | vs cuBLAS |
|---|---|---|---|
| (A) Triton W8A16 emu — the profiled `_gemm` | 542 | 0.19 | 161× |
| (B) Triton plain FP16 GEMM (pre-decoded weights) | 578 | 0.18 | 172× |
| (C) cuBLAS FP16 | 3.4 | 32.6 | 1× |

Two conclusions fall out:

1. **The e4m3 in-kernel decode is NOT the problem.** (A) and (B) are within 7% of
   each other — decoding the FP8 weight to FP16 in-kernel costs essentially
   nothing relative to the GEMM itself.
2. **The Triton GEMM itself is the problem, regardless of dtype.** A plain FP16
   `tl.dot` GEMM (B) is just as slow as the W8A16 one (A), and both are ~160×
   slower than cuBLAS FP16 (C) on the *same* FLOPs.

Dumping the PTX of the Triton GEMM confirms the mechanism:

```
PTX: mma/hmma = 0      (tensor-core instructions)
     fma(fp16) = 2048  (scalar CUDA-core FP16 FMA)
```

**`tl.dot` with FP16 inputs / FP32 accumulate is lowering to scalar CUDA-core
FP16 `fma` instead of tensor-core `mma.sync` (HMMA) on this SM75 / Triton 3.6.0
build.** The RTX 2080's FP16 tensor-core peak is ~10 TFLOPS; cuBLAS reaches
~33 TFLOPS effective here (it uses HMMA.16816). The Triton path is stuck at
~0.2 TFLOPS on the CUDA cores — a ~160× gap. That single codegen regression is
why a 4.55 TFLOP prefill takes 28 s instead of ~0.14 s.

> Note: this is a *codegen* problem, not a "Triton is unaccelerated" problem.
> `torch.matmul` / `tl.dot` are both "accelerated" in the abstract; the question
> is which instruction they emit, and here Triton emits the slow scalar FMA path
> for these block sizes on SM75.

---

## 4. Amdahl analysis (the gate)

- `_gemm_kernel` is **76.5% of prefill-2048 elapsed time** (27,936 / 36,637 ms).
- The rest (MoE experts 17.4%, GDN recurrence 4.6%, H2D 4.5%, attention 1.1%,
  copies 4.7%, misc 3.4%) is already on reasonable paths (Triton NVFP4 experts,
  FLA GDN, overlapped H2D, Triton attention).
- If the dense GEMM runs at cuBLAS-FP16 speed (~140 ms for the full 4.55 TFLOP
  instead of 27,936 ms), prefill-2048 drops from **36,637 ms → ~8,840 ms**, a
  **~76% improvement** (≈4.1×).
- The Phase B target is **≥10% median uncached-prefill improvement** (38.9 s →
  ≤35.1 s) while preserving the prefix-cache hit. The measured headroom
  (~76%) is **~7.6× the target**. This is not a marginal win; it is *the* win.

Max useful gain is therefore bounded by the non-GEMM 23.5% → even a perfect GEMM
leaves ~8.8 s. To go below that requires attacking the NVFP4 expert GEMM (17.4%),
which is a separate, smaller Phase E item.

---

## 5. Branch decision

**Choose Phase E (dense-projection GEMM → tensor cores).** Not D (GDN) and not C
(transfers): GDN recurrence is only 4.6% and the H2D is already overlapped; the
dense GEMM is 76.5% and is the single dominant, root-caused cost.

Two candidate implementations, to be A/B'd in the next experiment:

- **E1 — Dequant FP8→FP16 at load, run cuBLAS FP16.** Simplest, provably correct
  (cuBLAS uses HMMA), ~33 TFLOPS. Cost: +1.09 GB VRAM (1.111 B dense params,
  FP8→FP16). Model is currently 6,390 MiB allocated; +1,111 MiB → **~7,501 MiB of
  7,784 MiB total — fits, but tight** (KV cache / activations headroom shrinks).
  This is the low-risk path to validate the win.
- **E2 — Fix the Triton GEMM to emit tensor-core MMA** (retune block sizes /
  num_warps / num_stages so `tl.dot` lowers to `mma` on SM75, or a dedicated SM75
  GEMM). Keeps FP8 storage (no +1 GB), is the "native" fix, and also fixes the
  decode-path GEMM. Higher risk / more effort; the PTX shows the current
  (64,128,64) config does not lower to mma, so a config search or a hand kernel is
  needed.

**Recommendation:** implement **E1 first** to lock in the ~76% win and de-risk,
then pursue **E2** to reclaim the +1 GB and make the fix dtype-independent. E1
also directly answers the plan's Amdahl gate: it proves the dense GEMM is the
bottleneck end-to-end (in-model), not just in isolation.

---

## 6. Next experiment (concrete, measurable)

1. Add a load-time option to materialize the dense FP8 projections as FP16 and
   route `Fp8PerTensorLinear.forward` through `torch.matmul` (cuBLAS) when the
   weight is FP16 (SM75 has no FP8 tensor cores, so cuBLAS FP16 is the right
   target; the existing `_scaled_mm` W8A8 path is SM89+ only).
2. Restart the server on loopback:1920 with the same baseline flags + the E1
   option, run the 7-workload `bench.py`, and compare prefill-2048 median to the
   38,979 ms baseline. **Accept if median ≤ 35,100 ms (≥10%) AND prefix-cached
   still hits (cached_tokens ≈ 1984).**
3. In parallel, run the PTX/config probe (E2) to find an SM75 block config where
   `tl.dot` emits `mma`; if found, it becomes the memory-cheaper default.

Success metric: **prefill-2048 median 38,979 ms → expected ~8.8–9.5 s (~75% cut),
prefix-cache hit preserved.**

---

## 7. Instrumentation notes / limitations

- **NVTX sub-stage attribution did not populate** (`by_nvtx_stage.num_nvtx_ranges=0`).
  `torch.profiler` in this build does not export `torch.cuda.nvtx` ranges as trace
  events (the pre-existing `@nvtx_annotate("Layer_{id}")` in `model.py` is also
  absent from the trace). This did **not** block the diagnosis: the per-op device
  table is authoritative, and the `_gemm_kernel` shape decomposition (grid
  analysis) already pins it to the exact dense projections. If per-layer / per-stage
  attribution is wanted later, `nsys` (available on PATH) captures NVTX ranges
  reliably and is the better tool for that view.
- Sub-stage NVTX ranges added in this phase (`GDN_*`, `MoE_*`, `MHA_*`,
  `MoE_topk/h2d_wait/expert_gemm`) are always-on and cheap (CPU push/pop) and will
  show up under `nsys` even though they don't in the torch chrome trace.
- The 2048-token workload is chunked (2,058 full tokens > 2,048 cap → 2048 + 10-tok
  tail forwards); the captured 36.6 s is the dominant 2048 chunk. The 10-token tail
  is negligible.

## 8. Reproduce

```bash
cd ~/src/freetoken-sm75-native-prefill
# profile (arms profiler, captures the warm 512 + 2048 prefills)
bash scripts/sm75/serve-profile.sh 3 2 &            # AFTER=3 COUNT=2
.venv/bin/python scripts/sm75/profile_driver.py --model ~/Models/Qwen3.6-35B-A3B-NVFP4
# isolation / root-cause microbenchmarks
.venv/bin/python scripts/sm75/gemm_ab.py            # W8A16 emu vs cuBLAS fp16
.venv/bin/python scripts/sm75/gemm_rigorous.py      # consumed + correctness + calib
.venv/bin/python scripts/sm75/gemm_ptx.py           # PTX mma vs fma count
```
