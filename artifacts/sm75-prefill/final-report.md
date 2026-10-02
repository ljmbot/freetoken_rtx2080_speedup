# FreeToken SM75 native prefill — Final Report

**Date:** 2026-10-02  ·  **Phase:** G (final acceptance gate)  ·  **Status:** ACCEPTED

## TL;DR

The Phase E candidate (E1 dense-GEMM→cuBLAS + E3 NVFP4-expert-GEMM→tensor-core
cuBLAS, both env-gated and SM75-only) **passes every Phase G acceptance gate** and
is recommended for SM75 (Turing) deployments. The headline result is reproduced
after a clean server restart: **prefill-2048 38,979 ms → 5,739 ms (6.79×, an 85%
latency cut)**, with decode flat, prefix-cache preserved, and no correctness
regressions.

## Acceptance gate results

| Gate (freetoken_project.md Phase G) | Requirement | Result | Verdict |
|---|---|---|---|
| Prefill improvement (2048+) | ≥10% median | 6.79× (85% cut) | **PASS** |
| Decode regression | ≤5% | 32.3→33.0 tok/s (flat/up) | **PASS** |
| VRAM headroom | no OOM, maintain headroom | 1.09–1.11 GiB free (same as baseline) | **PASS** |
| Short-prompt protection | no regression on 128/512 | 2.36× / 6.28× improved | **PASS** |
| Cache protection | prefix-cache hit preserved | cached_tokens=1984 all reps | **PASS** |
| Clean-restart reproducibility | re-run matches A/B | 5768→5739 ms prefill-2048 (0.5%) | **PASS** |
| Sequential-request sweep | 20+ state-slot reuse, no leak | 24/24 coding/JSON/tool prompts, 200, healthy | **PASS** |
| Partial-cache-hit correctness | shared prefix reused | 1792/1826 tokens cached | **PASS** |
| Whole vs chunked prefill | both correct | 723-tok whole + 1322-tok chunked, correct | **PASS** |
| Cancel + recovery | stream abort, server recovers | aborted mid-stream, next request OK | **PASS** |
| Error recovery | malformed request rejected, server alive | 400 returned, next request OK | **PASS** |
| Non-SM75 dispatch | sm_89+ never takes new path | 31/31 dispatch decisions (gate×cap matrix) | **PASS** |
| NaN/Inf/CPU-fallback/crash | none across sweep | 0 across all 42 bench rows + 31 checks | **PASS** |
| Correctness edge matrix (Phase E) | all codes/edges | 91/91 pass (commit 87ab3fe) | **PASS** |

## Reproducibility re-run (clean restart)

The candidate was relaunched from scratch (`launch-candidate.sh` config, E1+E3,
loopback:1920) and the full 7-workload bench re-run (5 reps + 1 warmup, identical
flags to the Phase E A/B). Medians:

| workload | Phase E A/B (ms) | Phase G re-run (ms) | Δ |
|---|---|---|---|
| prefill-128 | 1713 | 1705 | -0.5% |
| prefill-512 | 1744 | 1736 | -0.5% |
| prefill-2048 | 5768 | 5739 | -0.5% |
| mixed-512 TTFT | 1749 | 1739 | -0.6% |
| mixed-2048 TTFT | 5765 | 5738 | -0.5% |
| prefix-cached TTFT | 1718 | 1713 | -0.3% |
| agent-envelope TTFT | 1728 | 1719 | -0.5% |

All within ~0.5% — the winning configuration is reproducible across a clean restart.

## Headline speedups (candidate E1+E3 vs baseline, median of 5)

| workload | baseline (ms) | candidate (ms) | speedup |
|---|---|---|---|
| prefill-128 | 4033 | 1705 | 2.36× |
| prefill-512 | 10906 | 1736 | 6.28× |
| **prefill-2048** | **38979** | **5739** | **6.79×** |
| mixed-512 | 10941 | 1739 | 6.29× |
| mixed-2048 | 38941 | 5738 | 6.78× |
| prefix-cached | 1743 | 1713 | 1.02× |
| agent-envelope | 1766 | 1719 | 1.03× |

Decode (offload backend) is unaffected because the new paths only touch prefill
(M>1); M==1 decode stays on the bandwidth-optimal W8A16 split-K GEMV.

## What was changed

Two env-gated, SM75-only code paths (never compiled/dispatched unless the gate is
on **and** `cap < (8,0)`):

- **E1** — `kernel/triton/fp8_pertensor_linear.py`: prefill dense GEMM →
  per-forward fp8→fp16 dequant + cuBLAS FP16 (tensor cores). Fixes the ~0.16
  TFLOPS scalar-FMA Triton W8A16 GEMM (76.5% of prefill-2048).
- **E3** — `moe/fused_nvfp4.py` + `moe/fused_nvfp4_cublas.py`: prefill NVFP4
  expert GEMM → fused single-launch dequant of a bounded expert group (
  `FREETOKEN_E3_GROUP=16`) + cuBLAS FP16 `bmm`. 136 ms/layer → 24.2 ms/layer.

Both are faithful numeric drop-ins (rel-L2 ~8e-4 vs fp32, cosine 1.0).

## Amdahl note

The remaining ~1.5–2 s wall on prefill-2048 is the non-GEMM floor (GDN, H2D
offload, attention, routing) — out of scope for Phase E; that is Phase D/F
territory. The GEMM families are now at tensor-core speed.

## Deliverables

- `docs/sm75-prefill.md` — project documentation.
- `final-report.md` — this file.
- `profile-summary.md` — Phase B profiling summary.
- `results.jsonl` / `results-e1.jsonl` / `results-e3.jsonl` / `results-g.jsonl`
  — raw per-run rows (baseline, E1-only, E1+E3, Phase G re-run).
- `summary.csv` — aggregated medians across all runs.
- `g-sweep.jsonl` — 31-check Phase G correctness sweep (all pass).
- `e3-edge-matrix-results.md` — 91/91 Phase E correctness edge matrix.
- `launch-candidate.sh` — the acceptance launch config (reproducible).
- `rollback.md` — how to stop the test server and return to the original.
- `build.log` / `environment-candidate.txt` / `source-provenance.json` /
  `model-metadata.json` — build + environment + provenance identity.
- `scripts/sm75/g_sweep.py` / `g_dispatch_gate.py` — Phase G sweep + dispatch-gate
  harnesses.

## Rollback

Stop the test server (`pkill -f 'ft serve'`) and keep using the original
endpoint (0.0.0.0:8888). No code rollback, dependency reinstall, or system CUDA
change is required — the candidate is an independent install and the new paths are
env-gated. See `rollback.md`.
