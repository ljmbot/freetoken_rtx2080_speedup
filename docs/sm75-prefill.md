# FreeToken SM75 native prefill

Optimising uncached prefill latency for a 35B-A3B NVFP4 MoE model on a Turing
RTX 2080 (SM75, 8 GB). This is an **independent test installation** of FreeToken
(`~/src/freetoken-sm75-native-prefill`), kept separate from the original
(`~/src/freetoken-sm75`, unchanged). The new paths are **env-gated and
SM75-only**, so the original service and any sm_89+ deployment are unaffected.

## Problem

Uncached prefill scaled ~linearly with prompt length and was very slow:
39.0 s for a 2048-token prompt on the baseline. Decode was ~32 tok/s (offload
backend) and already acceptable. The goal: cut prefill/TTFT without hurting
decode, prefix-cache correctness, or VRAM headroom.

## Root cause (Phase B, profiled)

On SM75, Triton `tl.dot` (FP16 in / FP32 acc) lowers to **scalar CUDA-core FP16
`fma` and 0 tensor-core `mma`** — a hard codegen limitation of Triton 3.6.0 on
Turing, not a tuning issue (a 75-config sweep, Phase E2, confirmed `mma=0` for
every block/warp/stage/transpose combination). Two GEMM families hit this:

1. **Dense projection GEMM** (attention/GDN in/out projections, W8A16) —
   `kernel/triton/fp8_pertensor_linear.py`. **76.5% of prefill-2048** at
   **0.16 TFLOPS** (cuBLAS FP16 on the same FLOPs runs ~33 TFLOPS, ~160× faster).
2. **NVFP4 routed-expert GEMM** — `moe/fused_nvfp4.py` `_prefill_nvfp4_moe_kernel`,
   ~0.76 TFLOPS, 136 ms/layer at M=2048.

## What was changed (Phase E, env-gated, SM75-only)

| Path | Gate | Change |
|---|---|---|
| E1 dense GEMM | `FREETOKEN_E1_CUBLAS=1` + `_sm75_or_lower()` | Prefill (M>1) dense GEMM → per-forward fp8→fp16 dequant + **cuBLAS FP16** (tensor cores). Decode (M==1) untouched (W8A16 split-K GEMV is bandwidth-optimal). Dequant is a per-forward temp (~50 MB), **no VRAM growth**. |
| E3 NVFP4 expert GEMM | `FREETOKEN_E3_CUBLAS=1` + `_sm75_or_lower()` + plain `*_and_mul` activation | Prefill NVFP4 expert GEMM → **fused single-launch dequant** of a bounded expert group + **cuBLAS FP16 `bmm`** + vectorized sync-free scatter. `FREETOKEN_E3_GROUP=16` bounds the FP16 transient (~53–64 MiB) to fit the 8 GB card. Decode + sm_89+ untouched. |

Both are faithful drop-ins: they round the dequantized weights to fp16 and
accumulate in fp32 exactly like the Triton kernels, so numerics match to fp16
rounding (rel-L2 ~8e-4, cosine 1.0). The new code is **never compiled or
dispatched** unless the gate is on **and** the GPU is SM75-or-lower.

## Results (candidate = E1+E3 vs baseline, median of 5, loopback:1920, identical flags)

| workload | baseline | candidate (E1+E3) | speedup | decode (base→cand) | cached-tok |
|---|---|---|---|---|---|
| prefill-128 | 4033 ms | 1706 ms | 2.36× | – | – |
| prefill-512 | 10906 ms | 1736 ms | 6.28× | – | – |
| **prefill-2048** | **38979 ms** | **5739 ms** | **6.79×** | – | – |
| mixed-512 | 10941 ms | 1738 ms | 6.29× | 32.4→33.3 | – |
| mixed-2048 | 38941 ms | 5740 ms | 6.78× | 32.3→33.2 | – |
| prefix-cached | 1743 ms | 1718 ms | 1.01× | 31.2→32.3 | 1984→1984 |
| agent-envelope | 1766 ms | 1728 ms | 1.02× | 33.4→34.1 | – |

(Reproducibility re-run after a clean server restart; see `results-g.jsonl`.)

## Acceptance (Phase G)

- **Prefill ≥10% median improvement on 2048+**: 6.79× (85% cut) — **PASS**.
- **Decode ≤5% regression**: flat to slightly up (M==1 GEMV untouched) — **PASS**.
- **No OOMs, maintain headroom**: 1.09–1.11 GiB free after init (same as baseline) — **PASS**.
- **Short-prompt protection**: 128/512 improved (2.36×/6.28×), no regression — **PASS**.
- **Cache protection**: prefix-cache hit preserved (cached_tokens=1984 all reps) — **PASS**.
- **Reproducibility**: clean-restart re-run matches within noise — **PASS**.
- **Correctness**: 91/91 edge-case matrix (all 16 E2M1 codes, nibble order, e4m3
  extremes, empty/skewed/single/repeated routing, all top-k, M∈{1,16,64,2048},
  3 activations, softmax+nontrivial router weights, tail masking) — **PASS**.
- **No NaN/Inf/CPU-fallback/crash** across the full workload sweep — **PASS**.

## Recommendation

The E1+E3 SM75 prefill path is **recommended for SM75 (Turing) deployments** and
is safe to adopt: it is env-gated, SM75-only, a faithful numeric drop-in, and
passes every Phase G gate. It is **not** a change to the sm_89+ or decode paths.

## Layout

The experiment harness lives in the git-ignored `rnd/` tree (kept for
reproducibility, **not part of the release** — see `rnd/instrumentation/README.md`):

- `rnd/sm75-scripts/` — serve wrappers (`serve-baseline.sh`, `serve-e1.sh`,
  `serve-e3.sh`), `bench.py` harness, `e_experts_edge_matrix.py` (Phase G
  correctness), Phase B/E probes.
- `rnd/artifacts/sm75-prefill/` — `STATUS.md`, `final-report.md`,
  `profile-summary.md`, `results.jsonl` (baseline), `results-e1.jsonl`,
  `results-e3.jsonl`, `results-g.jsonl` (Phase G re-run), `summary.csv`,
  `launch-candidate.sh`, `rollback.md`, `build.log`,
  `environment-candidate.txt`, `e3-edge-matrix-results.md`, `test-results/`,
  `traces/`.
- `docs/sm75-prefill.md` — this file.

For the current shipping launch (Ornith on 8 GB, radix prefix cache, 970
experts), see [../launch_ornith.md](../launch_ornith.md) — the E1/E3/GDN
paths described here are the kernels that launch uses.

## Reproduce

```bash
# install (re-runnable)
bash ~/src/freetoken-sm75-native-prefill/rnd/sm75-scripts/setup-venv.sh

# serve the candidate (E1+E3, loopback:1920)
bash ~/src/freetoken-sm75-native-prefill/rnd/artifacts/sm75-prefill/launch-candidate.sh

# benchmark (7 workloads, 5 reps + 1 warmup)
~/src/freetoken-sm75-native-prefill/.venv/bin/python \
  ~/src/freetoken-sm75-native-prefill/rnd/sm75-scripts/bench.py \
  --host 127.0.0.1 --port 1920 \
  --model ~/Models/Qwen3.6-35B-A3B-NVFP4 \
  --variant candidate --reps 5 --warmup 1 \
  --out ~/src/freetoken-sm75-native-prefill/rnd/artifacts/sm75-prefill/results-g.jsonl

# correctness edge-case matrix (no server needed)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ~/src/freetoken-sm75-native-prefill/.venv/bin/python \
  ~/src/freetoken-sm75-native-prefill/rnd/sm75-scripts/e_experts_edge_matrix.py
```

## Rollback

Stop the test server and keep using the original endpoint — no code rollback,
dependency reinstall, or system CUDA change. See `rollback.md`.
