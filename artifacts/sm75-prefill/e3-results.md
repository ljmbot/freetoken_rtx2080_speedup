# Phase E (expert branch) — NVFP4 MoE prefill → tensor-core cuBLAS (results)

**Status: ACCEPTANCE GATES PASSED.** Implemented the Phase E NVFP4 expert branch
and validated end-to-end on the 7-workload benchmark. This is the second half of
Phase E: E1 fixed the dense GEMM (76.5% of prefill); this fixes the NVFP4 routed
experts, which became the dominant cost after E1 (~65% of the E1 prefill).

- Date: 2026-10-02
- Machine: RTX 2080, 8 GB, SM75 (Turing)
- Project: `~/src/freetoken-sm75-native-prefill`, branch `perf/sm75-prefill`
- Baseline: commit `70a7888` (tag `baseline`); E1 on `5549a62`; E3 runs on
  `perf/sm75-prefill` with E1+E3 both enabled.

---

## 1. What changed

The NVFP4 routed-expert GEMM (`_prefill_nvfp4_moe_kernel`) dequantizes NVFP4 inline
in the GEMM K-loop, but on SM75 Triton emits **zero** tensor-core `mma` (the Phase E2
sweep proved this is a hard codegen limitation, not a tuning gap), so it runs at
~0.76 TFLOPS on CUDA-core FMA — **136 ms/layer** at M=2048 (5.4 s across 30 MoE
layers). After E1 cut the dense GEMM, this became the dominant prefill cost.

**E3** routes the *prefill* NVFP4 expert GEMM through a **tensor-core path**:

1. A **fused single-launch** Triton dequant (all experts in a group, one launch) →
   FP16 weights, with the per-row **global scale folded in** so the GEMM reads plain
   FP16 and never materializes an fp32 temporary. (A Python loop over 256 experts
   took 150 ms; one launch takes ~15 ms — the difference that makes the path viable.)
2. **cuBLAS FP16** `torch.bmm` (tensor cores) for gemm1 and gemm2.
3. A **vectorized, sync-free** scatter (`argsort` + `index_add_`) for routing.

The work is **grouped** to bound VRAM (the full-layer FP16 expansion, 1.6 GiB, does
not fit on the 8 GB card; only ~0.84 GiB is free after load). `group` experts' FP16
weights are live at once, `gate_up` is freed before `down` is allocated, and the
activation buffers are allocated **per group** (not once for all 256 experts). At
`group=8` the peak E3 transient is ~53 MiB, which fits alongside E1's dense temps.

Decode is **untouched** (stays on the NVFP4 GEMV, bandwidth-optimal). sm_89+ is
**untouched**.

Gated behind `FREETOKEN_E3_CUBLAS=1` + `_sm75_or_lower()` (cap < (8,0)) + a plain
`*_and_mul` activation + router weight on the gemm2 output (the model default).

### Files
- `python/freetoken/moe/fused_nvfp4_cublas.py` — **new**: self-contained
  tensor-core NVFP4 MoE prefill (fused dequant + cuBLAS bmm + vectorized scatter).
- `python/freetoken/moe/fused_nvfp4.py` — added the `FREETOKEN_E3_CUBLAS` gate,
  `_sm75_or_lower()`, and the dispatch branch at the top of `fused_experts_nvfp4`.
- `scripts/sm75/serve-e3.sh` — launch wrapper: **identical flags to baseline** +
  `FREETOKEN_E1_CUBLAS=1` + `FREETOKEN_E3_CUBLAS=1` + `FREETOKEN_E3_GROUP=8` +
  separate cache dir (A/B comparability).
- `scripts/sm75/e_experts_*.py` — isolation, decomposition, dequant, e2e,
  correctness, and integration validation.

---

## 2. Validation

### Drop-in correctness (e_experts_correct.py — full tensor-core path vs production
`fused_experts_nvfp4`, identical packed inputs, memory-efficient)
- **rel-L2 0.000813, cosine 1.000000, max abs diff 0.046875** — pure fp16 rounding;
  a faithful drop-in, not a re-quantization.

### Integration through the real dispatch (e_experts_integration.py)
- gate OFF (production Triton): 161.0 ms/layer
- gate ON (E3, group=8): **24.2 ms/layer (5.9× faster)**
- gate OFF vs ON: rel-L2 0.000813, cosine 1.000000

### Isolation benchmarks (M=2048, H=2048, I=512, E=256, top-8)
- Current Triton `_prefill_nvfp4_moe_kernel`: **136 ms/layer** (0 tensor cores, ~0.76 TFLOPS)
- Pure tensor-core GEMM ceiling (no dequant overhead): 3.53 ms (38× faster)
- Fused single-launch dequant (all 256 experts, one launch): 15 ms (vs 150 ms Python loop)
- **E3 tensor-core e2e (group=8): ~24 ms/layer (5.7× faster than current)**

### End-to-end 7-workload A/B (bench.py, 5 reps + 1 warmup, loopback:1920)

| workload | baseline ms | E1 ms | **E3 ms** | vs base | E3 vs E1 | base dec | E3 dec | cached |
|---|---|---|---|---|---|---|---|---|
| prefill-128 | 4033 | 1734 | **1713** | 2.36× | 1.01× | — | — | — |
| prefill-512 | 10906 | 3078 | **1744** | 6.25× | 1.76× | — | — | — |
| **prefill-2048** | **38979** | **9734** | **5768** | **6.76×** | **1.69×** | — | — | — |
| mixed-512 | 10941 | 2835 | **1749** | 6.26× | 1.62× | 32.4 | 33.8 | — |
| mixed-2048 | 38941 | 9807 | **5765** | 6.75× | 1.70× | 32.3 | 33.0 | — |
| **prefix-cached** | 1743 | 1738 | **1718** | 1.01× | 1.01× | 31.2 | 32.3 | 1984 |
| agent-envelope | 1766 | 1750 | **1728** | 1.02× | 1.01× | 33.4 | 34.1 | 128 |

### Acceptance gates (plan §Phase B / §Phase E)
- **prefill-2048 median 5,768 ms ≤ 35,100 ms (≥10% over 38,979) → PASS** (6.76×)
- **prefix-cache hit preserved (cached_tokens=1984, all 5 non-warmup reps) → PASS**
- **Decode flat/slightly up** (32.4→33.8 mixed-512, 32.3→33.0 mixed-2048) — the
  M==1 NVFP4 GEMV was correctly left untouched.
- **No NaN/Inf/crash across all 42 rows**; correct token counts per workload.

### Amdahl check
E1 alone: 38,979 → 9,734 ms. E3 adds the expert cut: 9,734 → 5,768 ms (1.69× on the
remaining expert-dominated prefill). Combined E1+E3: **6.76× over baseline** on
prefill-2048 — matching the Phase B prediction that fixing both the dense GEMM and
the NVFP4 experts would take the 36.6 s prefill to ~1.8–2 s of GEMM time (the 5.8 s
wall includes the non-GEMM floor: GDN, H2D offload, attention, routing, and the
per-request tail forward).

---

## 3. Why group=8 (memory)

The full-layer FP16 expansion (1.6 GiB) does not fit on the 8 GB card (0.84 GiB free
after load), and E1's dense temps + base activations consume most of that headroom.
Three OOMs during integration each removed one transient:

1. `gate_up` (268 MiB) + `down` (134 MiB) resident at once → free `gate_up` before
   allocating `down` (peak = `gate_up`, never `gate_up + down`).
2. Global activation scatter buffer `[256, Mp, H]` (~90 MiB) resident through the
   whole layer → allocate **per group** (~6 MiB).
3. The per-row global-scale multiply created a 256 MiB **fp32** temporary → **fold
   the global scale into the dequant kernel** (weight × global before the FP16
   store), so the GEMM is pure fp16 and no fp32 temp is ever materialized.

At `group=8` (`gate_up` = 32 MiB) the peak E3 transient is ~53 MiB at M=2048, which
fits alongside E1's dense temps. Speed is ~independent of group (same total dequant/
bmm work; only the bmm batch shape changes).

## 4. Remaining / next

- **Phase G** — end-to-end acceptance + final deliverables (12 required; see
  STATUS.md). Includes the clean-restart reproducibility re-run and the 20+ sequential
  request / chunked-vs-whole / partial-cache-hit correctness sweep.
- **Make E1+E3 the default on SM75** once accepted (flip `FREETOKEN_E1_CUBLAS` and
  `FREETOKEN_E3_CUBLAS` to on-by-default for cap < (8,0)).
- The non-GEMM floor (~1.5–2 s: GDN, H2D offload, attention, routing) is the only
  remaining prefill cost; further cuts would need Phase D (native GDN) or Phase F.

## 5. Reproduce

```bash
cd ~/src/freetoken-sm75-native-prefill
bash scripts/sm75/serve-e3.sh &                # E1 + E3, identical flags, loopback:1920
.venv/bin/python scripts/sm75/e_experts_correct.py      # drop-in numerics
.venv/bin/python scripts/sm75/e_experts_integration.py compare   # real dispatch
.venv/bin/python scripts/sm75/bench.py \
  --model ~/Models/Qwen3.6-35B-A3B-NVFP4 \
  --variant e3 --reps 5 --warmup 1 \
  --out artifacts/sm75-prefill/results-e3.jsonl
```
