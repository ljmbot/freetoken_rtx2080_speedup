# Phase E2 — Fix the Triton GEMM to emit tensor-core `mma` (result: DEAD END)

**Status: E2 is a dead end — conclusively ruled out.** No Triton config on SM75
emits tensor-core instructions. E1 (cuBLAS fp16) is the correct and only
Triton-free fix. This document records the decisive negative result so the
dead-end isn't re-explored.

- Date: 2026-10-02
- Machine: RTX 2080, 8 GB, SM75 (Turing); Triton 3.6.0, torch 2.11.0+cu130
- Project: `~/src/freetoken-sm75-native-prefill`, branch `perf/sm75-prefill`
- Script: `scripts/sm75/e2_ptx_sweep.py`

---

## 1. Hypothesis

Phase B (profile-summary.md) showed `tl.dot` (FP16 in / FP32 acc) lowers to scalar
CUDA-core FP16 `fma` (0 tensor-core `mma`) for the *one* config the model uses
(BM=64, BN=128, BK=64, warps=8, stages=3). E2 asked: is that a *tuning* problem —
i.e. is there some other block-size / warp / stage combination (or a way to avoid
the in-kernel `tl.trans`) that makes `tl.dot` emit `mma.sync` (HMMA) on SM75?

## 2. Method

`e2_ptx_sweep.py` compiles a **bounded** grid of configs and, for each, dumps the
PTX and counts `mma`/`hmma` vs `ffma`/`fma`. Two kernel variants isolate the
transpose:

- **Variant A (trans):** weight loaded as `[BN, BK]` then `tl.trans` → `[BK, BN]`
  (the current `_gemm` style).
- **Variant B (no-tr):** weight loaded directly as `[BK, BN]` (no in-kernel
  transpose) — rules out the transpose as the cause of the FMA fallback.

Config grid: BM ∈ {64,128} × BN ∈ {128,256} × BK ∈ {64,128} × warps ∈ {4,8} ×
stages ∈ {2,3}, plus the exact current config and a few small ones. 75 configs
compiled (the large-tile ones OOM on SM75's 64 KB shared memory — expected).

## 3. Result

**Every compiled config emits `mma=0` / `hmma=0`.** The `fma` (scalar FP16) count
is 2048–16384 depending on tile size; `ffma` (fp32) is 0 in all. No config — in
either variant, at any block size, warp count, or stage count — produces a single
tensor-core instruction.

Representative rows (full table in the run log):

```
variant    BM   BN   BK  w st |   mma  ffma   fma | target
A(trans)   64  128   64  8  3 |     0     0  2048 | sm_75   <- current model config
A(trans)  128  256   64  8  3 |     0     0  16384 | sm_75
A(trans)  128  128  128  8  3 |     0     0   8192 | sm_75
B(no-tr)   64  128   64  8  3 |     0     0  2048 | sm_75
B(no-tr)  128  256   64  8  3 |     0     0  16384 | sm_75
... (all 75 rows: mma=0)
```

## 4. Conclusion

- **The `tl.dot` → CUDA-core FMA fallback is a hard codegen limitation of
  Triton 3.6.0 on SM75, not a tuning issue.** No block-size / warp / stage
  combination, and not removing the in-kernel transpose, produces tensor-core
  `mma`.
- **E2 (retune the Triton GEMM) is a dead end.** The only way to get tensor-core
  GEMM on SM75 is to leave Triton for a kernel that emits HMMA — i.e. **cuBLAS
  fp16 (E1, already merged)** or a hand-written CUDA/HMMA kernel.
- **E1 is validated as the correct fix.** It is the minimal, memory-safe
  (no VRAM growth), correctness-preserving path to tensor-core GEMM on SM75.
  The per-forward dequant is the price of staying fp8-resident; a hand kernel
  (E2-extended) could dequant in-kernel and use HMMA, but that is strictly more
  work for no measurable gain over E1 (E1 already hits ~33 TFLOPS via cuBLAS).

## 5. What this rules in / out

| path | verdict |
|---|---|
| E1 — dequant fp8→fp16 + cuBLAS fp16 | **ADOPTED** (merged, 4.0× prefill, gates PASS) |
| E2 — retune Triton GEMM for `mma` | **DEAD END** (no config emits mma) |
| E2-extended — hand HMMA kernel with in-kernel fp8 decode | feasible but no gain over E1; not worth the effort |

## 6. Reproduce

```bash
cd ~/src/freetoken-sm75-native-prefill
.venv/bin/python scripts/sm75/e2_ptx_sweep.py   # ~15 min (75 Triton compiles)
```
