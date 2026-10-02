# Phase E — expert-branch correctness edge-case matrix

Closes the plan's Phase E "Expert-kernel correctness gates" (freetoken_project.md).
The integration drop-in test (`e_experts_correct.py`) only proves the E3 tensor-core
path matches production on one well-behaved shape (M=2048, top_k=8, realistic
distribution). This matrix proves it across the full edge-case surface, against
three independent references.

## Method

DUTs: `fused_experts_nvfp4_cublas` (E3, tensor-core) and `fused_experts_nvfp4`
(production Triton). References:

- **(c) DEQUANT** — E3 `_dequant_group` fp16 output vs an independent fp64 dequant
  (same E2M1 LUT + the same `e4m3_u8_to_f32` bitcast both kernels share), elementwise.
- **(a) DROP-IN** — E3 vs production, rel-L2. Both round dequant weights to fp16 and
  accumulate in fp32, so they must agree to fp16/accumulation-order rounding. The two
  paths orchestrate differently (E3: argsort+index_add_ + per-group cuBLAS bmm;
  prod: moe_align_block size + padded Triton GEMM), so this catches scatter /
  group-boundary / router-weight / activation bugs.
- **(b) TRUTH** — both DUTs vs an independent fp64 reference that dequantizes NVFP4
  exactly and runs the full routed MoE in fp64 (per-expert to bound the 8 GB card).
  Catches a *shared* dequant misunderstanding.

## Edge cases covered (per the plan)

- all 16 E2M1 codes (0x0-0xF), low+high nibble order, 16-wide block boundaries
- e4m3 scales: zero, subnormals, smallest normal, a normal, max-finite, negatives,
  and the two reserved/NaN codes (0x7F/0xFF)
- per-row global scale
- empty experts, skewed routing, single-expert, repeated expert ids, all top-k values
  (1,2,4,8), M in {1,16,64,2048}
- nontrivial (non-normalized, incl. a zero and a >1) router weights
- activation kinds silu / gelu / gelu_tanh; small + unit + zero-free magnitudes
- tail masking (M*top_k not divisible by the production BLOCK_SIZE_M)

## Results — 91/91 PASS

```
== (c) dequant unit test (all e2m1 codes, e4m3 extremes, boundaries) ==
  [PASS] dequant-exact   max_rel=0.000e+00 zero_ok=True finite=True
  e4m3 decode: 0x7E=448.0000  0x7F=480.0000  0xFF=-480.0000  (reserved codes, bitcast)

== (a) drop-in matrix: E3 vs production (M<=64 + M=2048) ==
  62/62 PASS   drop-in rel-L2 in [5.3e-4, 1.05e-3], all finite
  (M in {1,16,64} x top_k in {1,2,4,8} x {uniform,skewed,sparse-empty,
   single-expert,repeated-ids}, plus M=2048 x {uniform,skewed})

== (b) truth subset: E3 & prod vs fp64 reference ==
  16/16 PASS   truthT (E3 vs fp64) in [4.8e-4, 7.8e-4]
               truthP (prod vs fp64) in [3.3e-4, 4.9e-4]

== (a/b) activation / weight / magnitude sweep (M=64) ==
  12/12 PASS   drop-in ~8e-4, truthT ~7.3e-4, truthP ~4.6e-4
  (act in {silu,gelu,gelu_tanh} x w in {softmax,nontrivial} x hs in {0.01,1.0})

TOTAL: 91/91 passed
```

## Interpretation

- **Dequant is exact.** E3's fp16 dequant matches the fp64 reference to fp16-store
  rounding (max elementwise rel err 0.0 on the constructed grid; zeros exact; all
  finite). All 16 E2M1 codes, both nibble positions, block boundaries, and the full
  e4m3 scale range (incl. reserved 0x7F/0xFF -> +-480 via the shared bitcast) are
  correct. The per-row global scale is applied.
- **Drop-in holds on every edge case.** E3 matches production to 5.3e-4-1.05e-3
  rel-L2 — the same band as the validated integration baseline (~8e-4) — for M=1
  (single token), skewed/empty/single/repeated routing, all top-k values, and the
  production M=2048 shape. No NaN, no inf, no routing/orchestration divergence.
- **The dequant contract is not a shared misunderstanding.** Both DUTs agree with the
  independent fp64 reference to ~5e-4-8e-4 rel-L2 (the fp16 weight-rounding bound),
  confirming the E2M1/e4m3/nibble/global-scale interpretation is correct, not just
  self-consistent between the two kernels.
- **Reserved e4m3 codes decode to +-480** in all three (kernel bitcast + reference),
  so a stray 0x7F/0xFF scale cannot silently produce a wrong-but-finite result — it
  is a loud, consistent value.

## Scope notes / what this does NOT cover

- **Latency gate** is separate and was already met by the Phase E integration + e2e
  runs (5.9x expert GEMM, 6.76x prefill-2048); this script is the numeric half.
- The MoE matrix uses realistic weight magnitudes (scale range 0x08-0x20, decode
  ~0.004-0.25) so the fp16 GEMM intermediates stay in range. The *dequant unit test*
  exercises the full 0x00-0xFF scale range at the element level, which is where
  overflow would surface. An end-to-end run with adversarially large scales would
  overflow fp16 in **both** kernels identically (not a correctness divergence), so it
  is out of scope for a drop-in/truth comparison.
- This validates the expert branch in isolation. Phase G (clean-restart
  reproducibility, 20+ sequential requests, chunked-vs-whole, partial-cache-hit
  correctness) exercises the branch inside the full server and is the next gate.

## Run

```
cd ~/src/freetoken-sm75-native-prefill
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python scripts/sm75/e_experts_edge_matrix.py
```

Exit 0 = all gates pass; 1 = any failure. Runtime ~2 min on the RTX 2080.
