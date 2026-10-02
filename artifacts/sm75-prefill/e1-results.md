# Phase E1 — Dense GEMM → cuBLAS FP16 (results)

**Status: ACCEPTANCE GATES PASSED.** Implemented the Phase B chosen branch (E1)
and validated end-to-end on the 7-workload benchmark.

- Date: 2026-10-02
- Machine: RTX 2080, 8 GB, SM75 (Turing)
- Project: `~/src/freetoken-sm75-native-prefill`, branch `perf/sm75-prefill`
- Baseline: commit `70a7888` (tag `baseline`); E1 runs on `perf/sm75-prefill`

---

## 1. What changed

On SM75 the dense FP8 projections (`Fp8PerTensorLinear`) route through the Triton
W8A16 `_gemm`, whose `tl.dot` lowers to scalar CUDA-core fp16 FMA (0 tensor-core
mma) → 0.16 TFLOPS. Phase B (profile-summary.md) showed this is 76.5% of
prefill-2048.

**E1** routes the *prefill* (M>1) dense GEMM through a **per-forward fp8→fp16
dequant + cuBLAS fp16** (tensor cores, ~33 TFLOPS). Decode (M==1) is **untouched**
— it stays on the W8A16 split-K GEMV, which reads the fp8 weight once (the
bandwidth-optimal decode path). The dequant is **not cached** (per-forward temp,
~50 MB/GEMM, freed on return), so the resident model stays fp8 and there is **no
VRAM growth**.

Gated behind `FREETOKEN_E1_CUBLAS=1` and `_sm75_or_lower()` (cap < (8,0)) so
sm_89+ (native fp8 W8A8) is never touched.

### Files
- `python/freetoken/kernel/triton/fp8_pertensor_linear.py` — added
  `_e1_dequant_cublas()`, `_sm75_or_lower()`, the `FREETOKEN_E1_CUBLAS` gate, and
  the M>1 dispatch branch.
- `scripts/sm75/serve-e1.sh` — launch wrapper: **identical flags to baseline** +
  `FREETOKEN_E1_CUBLAS=1` + separate cache dir (A/B comparability).
- `scripts/sm75/e1_probe.py`, `scripts/sm75/e1_correctness.py` — validation.

---

## 2. Validation

### Correctness (e1_correctness.py, GDN in_proj [2048×2048]×[12288×2048])
- E1 (fp16 dequant + cuBLAS) vs fp32 reference: **rel-L2 0.00029, cosine 1.000000**
- Current Triton W8A16 vs fp32 reference: rel-L2 0.00021, cosine 1.000000
- E1 vs current: rel-L2 0.00023 — **same numeric ballpark as what's deployed**; no
  quality regression.

### Isolation perf (e1_probe.py, full dense prefill, 80 GEMMs, M=2048)
- dequant + cuBLAS fp16: **178 ms** (vs 27,900 ms profiled in-model → ~156×)
- per-forward dequant transient: 50 MB/GEMM (not cached)

### End-to-end 7-workload A/B (bench.py, 5 reps + 1 warmup, loopback:1920)

| workload | baseline ms | E1 ms | speedup | base dec | E1 dec | base cached | E1 cached |
|---|---|---|---|---|---|---|---|
| prefill-128 | 4033 | 1734 | 2.33× | — | — | — | — |
| prefill-512 | 10906 | 3078 | 3.54× | — | — | — | — |
| **prefill-2048** | **38979** | **9734** | **4.00×** | — | — | — | — |
| mixed-512 | 10941 | 2835 | 3.86× | 32.4 | 32.7 | — | — |
| mixed-2048 | 38941 | 9807 | 3.97× | 32.3 | 32.6 | — | — |
| **prefix-cached** | 1743 | 1738 | 1.00× | 31.2 | 32.0 | 1984 | **1984** |
| agent-envelope | 1766 | 1750 | 1.01× | 33.4 | 33.4 | — | 128 |

### Acceptance gates (plan §Phase B)
- **prefill-2048 median 9,734 ms ≤ 35,100 ms (≥10% over 38,979) → PASS** (4.00×)
- **prefix-cache hit preserved (cached_tokens=1984, all 5 reps) → PASS**
- Decode rates flat (no regression) — the M==1 GEMV was correctly left on the
  fp8-weight path.

The 4.0× matches the Phase B Amdahl prediction (~76% cut → 36.6 s → ~8.8 s for the
2048 chunk; the full 2058-token workload including the 10-token tail forward +
per-request overhead lands at 9.7 s).

---

## 3. Why not cache the FP16 dequant (E1 vs E1-cached)

Caching all dense FP8→FP16 would add ~1.1 GB resident. The baseline leaves only
**0.84 GiB free** after load (KV cache + CUDA graphs), so a 1.1 GB cache would OOM.
The per-forward dequant costs ~10 ms/layer-set (178 ms total for the full prefill
vs 9,734 ms wall) — negligible, and it keeps the resident model fp8. This is why E1
is memory-safe at `--memory-ratio 0.95`.

## 4. Remaining / next

- **E2 (optional, memory-neutral alternative):** fix the Triton GEMM to emit
  tensor-core `mma` on SM75 (block-size/warp/stage search or a hand kernel). Would
  let the decode path also use tensor cores and avoid any dequant, but is higher
  effort and the decode GEMV is already bandwidth-optimal, so the payoff is smaller
  than E1's. E1 is the pragmatic win; E2 is a future refinement.
- The non-GEMM floor (~2,000 ms: NVFP4 experts 17%, GDN 4.6%, H2D 4.5%, attention
  1.1%) is the next target if a further cut is wanted — that is the NVFP4 expert
  GEMM (Phase E, expert branch).
- E1 is env-gated (`FREETOKEN_E1_CUBLAS=1`). Recommend making it the **default on
  SM75** once this is accepted (flip the gate to on-by-default for cap < (8,0)).

## 5. Reproduce

```bash
cd ~/src/freetoken-sm75-native-prefill
bash scripts/sm75/serve-e1.sh &                                   # FREETOKEN_E1_CUBLAS=1
.venv/bin/python scripts/sm75/e1_correctness.py                   # numerics
.venv/bin/python scripts/sm75/bench.py \
  --model ~/Models/Qwen3.6-35B-A3B-NVFP4 \
  --variant e1 --reps 5 --warmup 1 \
  --out artifacts/sm75-prefill/results-e1.jsonl
```
