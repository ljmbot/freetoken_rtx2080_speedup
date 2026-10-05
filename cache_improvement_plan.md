# cache_improvement_plan.md — Expert cache & PCIe expert-streaming (Qwen3.6-35B-A3B-NVFP4, RTX 2080 sm75 8 GB)

**Goal.** Cut decode time by attacking the measured bottleneck — **PCIe expert streaming,
25.7 ms/tok = 65% of decode** — and close the gap to the 60–70% expert-cache hit rates
the Strata engine reports. Target was **~1.3–1.5× decode (25 → ~33–39 tok/s)**, measured
by the production A/B harness, without touching the 64K-context / 640-cache / prefill
invariants.

**Status (2026-10-05): investigation complete.** Phase 1 (640-slot cache) shipped (+1–2
tok/s). Phase 2 (the read-ahead — both the YALIS/Strata next-layer *prefetch* and the
*prediction-guided eviction* pivot) **closed negative** with measurements: the real 3-layer
prototype shows prefetch is a net loss (k=0 baseline beats every k>0, 0.96×→0.84×), and the
router prediction is an intra-token signal that can't drive eviction (+1.5 pts honest, below
gate). LRU is near-optimal at 640 slots. Remaining wins: Phase 3 (static warm start,
cheap/optional) and Phase 5 (batching, the structural win that scales — separate project, S6).

**One-sentence diagnosis.** Decode is PCIe-bandwidth-bound on streaming expert weights
CPU→GPU (one CUDA C++ kernel, `fast_index_copy_multi`). The LRU expert cache *already*
implements read-ahead (45.7% hit at 640 slots) and is near-optimal among implementable
policies. The remaining structural win is **batching** (amortize one fetch across N
requests); the per-request read-ahead levers are exhausted on this hardware.

---

## 0. The evidence (what we actually measured, 2026-10-05)

All numbers from the graphed production decode path, cache 640, Q8-KV, chunked 4096 prefill.

### 0.1 The bottleneck is ONE kernel (not a "herd")

Full kernel dump (`s2_full_dump.py`, `/tmp/s2fd.log`), 39.3 ms/iter:

| kernel | ms/iter | % of 39.3 ms |
|---|---|---|
| **`fast_index_copy_multi`** (CUDA C++, CPU→GPU expert copy) | **25.7** | **65%** |
| `_gemv_splitk` (MoE GEMV) | 4.1 | 10% |
| `_decode_nvfp4_marlin` | 3.8 | 10% |
| GDN decode (fused_sigmoid_gating_delta_rule_update + causal_conv1d + layer_norm) | ~0.6 | 1.5% |
| router (gatherTopK + bitonicSort + softmax_warp) | ~0.6 | 1.4% |
| MoE-compute-adjacent (nvfp4_gemv*, gemvx, splitk_reduce, act_and_mul, moe_sum, grouped_stage1) | ~2.3 | ~6% |
| all elementwise/norm/copy combined | ~3.4 | ~9% |

**The "thundering herd" is a single fused multi-bank copy kernel** that streams on-demand
expert weights CPU→GPU over PCIe. It is already hand-written CUDA C++
(`python/freetoken/kernel/csrc/jit/fast_index_copy.cuh`, JIT-compiled) — *not* a Python problem.

### 0.2 PCIe ablation — the copy is the whole story

`s2_pc_test.py`, `FREETOKEN_SKIP_FAST_INDEX_COPY=1` = "experts resident" upper bound:

| config | ms/tok | tok/s |
|---|---|---|
| normal (copy on critical path) | **39.7** | 25.2 |
| copy free (experts resident) | **14.0** | 71.3 |
| **Δ = PCIe expert streaming** | **25.7** | — |

So the copy is 65% of decode and the compute (attention + GEMV + router + norms) is 14.0 ms.

### 0.3 We are bandwidth-saturated on the copy

At the measured 46.7% hit rate, ~53% of the ~0.57 GB/token of expert weights (NVFP4 4-bit
+ fp8 scales: 8 experts × 40 layers × ~1.77 MB/expert) still cross PCIe ≈ 0.30 GB.
0.30 GB / 25.7 ms ≈ **11.7 GB/s**, at the effective PCIe 3.0 x16 limit (~12–13 GB/s).
**The copy engine is pegged.** This decides the lever (below): a
*hit-rate* improvement removes bytes, but a *prefetch* (overlap) hides bytes behind compute —
and the latter works even when bandwidth-saturated, because copy (DMA) and compute (SMs)
are independent hardware that can run concurrently.

### 0.4 The LRU cache already IS the read-ahead

`offload_cache.py` keeps a shared 640-slot pool; a fetched expert stays resident and a
next-token reuse is a free hit. That *is* "guess the next expert, keep it, throw it away if
wrong." The measured success rate of that guess:

- **Live hit rate (cache 640): 45.7%** (`--moe-collect-stats` → `decode_miss_stats()`)
- **Working set: 7,054 distinct (layer,expert) pairs** vs 640 slots → **11× mismatch**
- Each layer touches **~176 of its 256 experts** over 300 tokens → the router picks are too
  diverse for a small cache to nail.

### 0.5 Hit-rate ceiling (offline sim over the real recorded routing, `s2_cache_sim.py`)

Recorded the actual per-token, per-layer routing (40 layers × 8 experts, 300 tokens,
`s2_route_record.py`, graphs off / deterministic-greedy so locality matches the graphed
path), then simulated caches offline (capacity = moe-cache-size):

| cache size | LRU hit% (what we run) | Belady-OPT hit% (perfect prediction = ceiling) | gap |
|---|---|---|---|
| 512 | 40.3% | 64.3% | 24.0 |
| **640** | **46.7%** (≈ live 45.7%, sim validated) | **68.2%** | **21.5** |
| 992 | 56.6% | 75.7% | 19.1 |
| 2048 | 74.6% | 87.0% | 12.4 |
| 8192 | 92.7% | 92.7% | 0 |

The LRU column reproduces the live 45.7%, so the sim is trustworthy. The ceiling at 640 is
68.2% — but see §1: a *practical* predictor can't get there.

### 0.6 Practical predictors can't beat LRU at 640 (`s2_cache_sim2.py`)

| policy @ 640 slots | hit% |
|---|---|
| **LRU (baseline)** | **46.7%** |
| LFU (global hot set) | 43.1% |
| per-layer hot set | 41.9% |
| Belady-OPT (ceiling) | 68.2% |

**The hit-rate lever is mostly exhausted at 640 — it's a *capacity* problem, not a
*policy* problem.** The working set (7,054 pairs) is 11× the cache (640 slots); no cheap
rule (recency, frequency, per-layer hot) recovers the gap. Only *perfect* prediction does,
and that requires knowing the future.

### 0.7 VRAM ceiling on cache size (`cs_probe.py`, chunked 4096 prefill)

| cache_size | VRAM after build | 4096-chunk prefill peak | decode |
|---|---|---|---|
| 512 (was current) | 6510 MiB | 7164 MiB | 24.93 tok/s |
| **640 (shipped, `d153424`)** | 6724 MiB | 7396–7399 MiB (fits, ~200 MiB spare) | 25.81–26.97 tok/s |
| 992 (marlin max) | 7330 MiB | **OOM** (90 MiB free) | — |

Real headroom to the 7600 MiB limit is **~436 MiB** (the build already holds the full 64K
KV+GDN pool + CUDA graphs). **640 is the max that fits; 992 OOMs.** Bigger cache is VRAM-capped.

---

## 1. Two independent levers (and which one pays)

The decode time is `copy + compute = 25.7 + 14.0 = 39.7 ms`. There are two ways to cut it:

- **Lever A — raise the hit rate (remove bytes).** At 640 slots, LRU 46.7% → OPT 68.2% is
  the *theoretical* max, but every *practical* predictor (LFU 43.1%, per-layer hot 41.9%) is
  **worse than LRU**. So this lever is ~exhausted at our cache size (capacity-bound).
  Raising it needs a bigger cache, which is VRAM-capped at 640 (§0.7).
- **Lever B — overlap copy with compute (hide the bytes).** Prefetch the next layer's
  experts while the current layer computes. Ceiling = `max(copy, compute) = max(25.7, 14.0)
  = 25.7 ms` → **1.545×**. This is *independent of hit rate* (prefetch moves bytes earlier,
  it doesn't remove them) and works even though we're bandwidth-saturated, because copy and
  compute run on independent hardware.

**Lever B is the win.** It is exactly the user's read-ahead idea, in the form that actually
works (next-*layer*, not next-*token* — see §2 and §3). Lever A is a minor supplement
(§4 Phase 3).

**Prefetch ceiling arithmetic:**

```
copy (PCIe expert streaming) = 25.7 ms/tok
compute (copy-free)          = 14.0 ms/tok
current serial total         = 39.7 ms/tok  (25.2 tok/s)
perfect-overlap total        = 25.7 ms/tok  (38.9 tok/s)   <- max(copy, compute)
ceiling speedup              = 1.545x
```

The win = hiding up to 14.0 ms of copy behind the 14.0 ms of compute. Realistic outcome is
Realistic outcome is between 1.0× (if overlap doesn't hold on sm75) and 1.545× (perfect overlap).
YALIS's real-world numbers: **12–14% TPOT on the A6000 (copy = 84–88% of TPOT — *more*
copy-bound than us) and 5–8% on A100/GH200 (more compute-bound)**. Our copy=65% sits between
those regimes, so expect the **mid-range (~8–12%)**, not the top of it; 1.545× is the
theoretical ceiling under *perfect* overlap only. The overlap fraction is the key thing to
measure (Phase 2 gate).

> **⚠️ Post-hoc correction (2026-10-05, §4 Phase 2 closed negative):** the arithmetic above
> holds `copy = 25.7 ms` constant while overlapping — but the overlap *requires* the
> prefetch, and the prefetch **adds bytes** to the link. *(The original draft here said
> "2.3× miss traffic" — that double-counted: the prefetch's useful fetches are the demand
> misses moved earlier, so they cancel. The only genuinely new bytes are the spurious
> predictions. Corrected: total ≈ 6.0 rows/layer vs 4.3 baseline, i.e. +1.75 spurious/layer.)*
> On a copy-dominated link the copy time grows with those extra bytes, eating the 14.0 ms
> of hidden compute. The 1.545× ceiling is only reachable when the link has spare
> bandwidth (YALIS's A6000 regime); on our copy-dominated sm75 link it is not. **The real
> 3-layer prototype (G2.7, §6) confirmed this directly: k=0 baseline beats every k>0.**
> See §4 Phase 2 for the full closure.

---

## 2. Why "next-layer" and not "next-token"

The user's original read-ahead idea was *next-token*: predict token *t+1*'s experts during
token *t*'s compute. That requires knowing token *t+1*'s hidden state, which needs a full
forward — i.e. the **MTP head**, which we already proved is too slow (MTP accept ≈ 0.46×,
reject ≈ 0.23× vs the graphed baseline; decisively negative, closed).

The **YALIS "Speculating Experts"** paper (arXiv 2603.19289) and the **Strata engine** both
use the cheaper **next-layer** variant: predict layer *l+1*'s experts from layer *l*'s
hidden state, *during* layer *l*'s compute. The prediction needs only the router (~0.6 ms),
not a full transformer layer. It's a different, cheaper, and more accurate problem:

- The hidden state entering layer *l* is a good proxy for the one entering layer *l+1*
  (MoE+attention are smooth transforms). YALIS measures **small inter-layer drift beyond
  the first 2 layers** for **Qwen3-30B-A3B — our exact model family** → high recall@k on the
  top-ranked (high-weight) experts.
- The prediction is used **only to prefetch** (move bytes earlier), **not to route**. The
  true routing is unchanged, so **there is no accuracy impact** (the safe version — see §5).

This is the version of the idea that works, and it's what Strata ships.

---

## 3. What Strata actually does (source + docs pulled)

**Strata = `Niko1221/Strata`** (GitHub, MIT). Pulled `README.md`, `docs/HOW_IT_WORKS.md`,
`docs/DETAILS.md`, `docs/BATCHING.md` (saved under `scratch/strata/`). The engine runs
Qwen3.8-Flash-Next (a *different*, bigger MoE: 512 experts/layer, 10 active per token,
24,576 total) on gaming PCs: **all 24,576 experts live pinned in system RAM**, the GPU cache
keeps the hottest, and the SSD holds the 28.8 GB n-gram table (PLE), not experts.

**The 60–70% hit rate is NOT primarily a predictor.** It comes from four things:

1. **A much bigger cache.** The GPU expert cache fills "the rest of VRAM" (VRAM − ~5 GB;
   "every extra GB holds ~700 more experts"). On a 12 GB card that's ~4,900 experts
   (7 GB × 700/GB) — ~7.7× our 640 slots. Bigger cache → higher hit rate, mechanically.
2. **A static expert profile preloaded at start.** `data/expert-profile.bin` ranks all
   24,576 (layer,expert) pairs from calibration (`tools/make_profile.py` + `--dump-routing`).
   The cache starts *warm* with the hottest experts before the first token — so the
   "conversation" hit rate is high from token 1.
3. **An adaptive tier** (`--adapt-every`) that moves experts in/out of VRAM from live usage
   counts, persisted across restarts (`expert_profile_save`).
4. **CPU co-computation of misses.** This is the big one: "the CPU computes the experts that
   are not on the GPU **in place**, at the same time as the GPU works on the cached ones
   (AVX-512/AVX2, ggml kernels)." Strata only streams a *fraction* of misses over PCIe
   (`--pcie-frac`); the rest are **computed on the CPU in parallel**. So a "miss" in Strata
   is not a PCIe stall — it's overlapped CPU work. That's why Strata can be fast *despite* a
   lower effective VRAM hit rate than it would otherwise need.

**The actual predictor = `STRATA_LOOKAHEAD` routing prefetch** (the one piece that matches
the user's hypothesis). From `docs/DETAILS.md`:

> "while the CPU works on a layer, a thread applies the **next layer's router to this
> layer's input** and asks the OS for the pages of the **predicted experts** that neither the
> GPU nor the RAM budget holds (only pages — **the experts computed are the same**;
> `STRATA_LOOKAHEAD=0` turns it off)."

That is the YALIS technique: predict layer *l+1*'s experts from layer *l*'s input, warm the
pages, and **the true routing still decides what's computed** (pure prefetch, no accuracy
trade). Strata's `--stats` reports it separately: `"routing prefetch: how many of the file
reads had been warmed."`

**Why Strata's 60–70% ≠ our 45.7% (not directly comparable):**
- Strata's cache is ~6× bigger (VRAM − 5 GB vs our 640-slot hard cap).
- Strata preloads a static profile (warm start); we start cold.
- Strata co-computes misses on the CPU; we stream all misses over PCIe, so we're more
  sensitive to hit rate.
- Different model (512 experts/layer top-10 vs our 256 top-8) → different working set.

**Net:** the user's suspicion is *partly* right — Strata does do speculative next-layer
fetch. But the 60–70% number is mostly bigger-cache + warm-profile + CPU-co-compute, with
the prefetch as a *supplement* that makes misses cheap (overlapped) rather than as the main
hit-rate driver. For us, the transferable, high-value piece is the **next-layer routing
prefetch** (Phase 2). The CPU co-compute (Phase 4) is the "Strata way" but is a large
architectural change (C++ CPU ggml kernels) for a PyTorch engine — noted, not recommended.

---

## 4. Phased plan (each phase: gated, default-off, rollback-safe, accuracy-reviewed)

### Phase 1 — 640 MoE cache. ✅ SHIPPED (`d153424`)
`qwen_q8.sh` `--moe-cache-size 512 → 640`. +1–2 tok/s (4–8%). Max that fits the 64K
chunked-prefill peak (~200 MiB spare). No further cache-size headroom (992 OOMs). Done.

### Phase 2 — Read-ahead (prefetch / prediction-eviction). ❌ CLOSED NEGATIVE (2026-10-05)

Both forms of the YALIS/Strata read-ahead were investigated and **closed negative on this
hardware**. This is a measured result, not a deferral.

**(a) Next-layer prefetch (original design) — net loss, MEASURED.** The PCIe link is
copy-dominated (copy 25–34 ms/tok ≫ compute 10 ms/tok). G2.1 confirmed the *overlap
mechanism* works (92–95% on sm75) and G2.2 confirmed the *prediction* is strong
(recall@8 78.1%), and the **real 3-layer prototype (G2.7, §6) measured the net effect
directly: prefetch loses at every k** (k=0 baseline 1.67 ms/layer; k=1 0.96×, k=4 0.93×,
k=8 0.84×). The reason is the byte accounting, corrected from the original "2.3×" claim
which double-counted: the prefetch's *useful* fetches are the demand misses moved earlier
(they cancel out of the total), so the only genuinely new bytes are the **spurious
predictions** (predicted but not actually routed). Total ≈ 6.0 rows/layer vs 4.3 baseline
(+1.75 spurious/layer). On a copy-dominated link those extra bytes grow the copy time and
eat the ~14 ms of compute the overlap would hide — there is no free compute window to hide
them under, because the prefetch and the demand copy **share the same PCIe link**. The
overlap mechanism (G2.1) is real, but it hides *demand* copy under compute only if the
prefetch didn't have to put *extra* bytes on the wire; on a pegged link it can't. Net loss.

**(b) Prediction-guided eviction (the pivot) — +1.5 pts, below gate.** The pivot's premise
was that the 78% router prediction could drive near-Belady eviction. **It cannot, because
the signal is the wrong kind.** The router prediction is an **intra-token (next-layer)**
signal: `topk(gate[l+1] @ H[t,l])` predicts layer *l+1*'s routing **at the same token t**.
Eviction needs a **next-token** signal: layer *l*'s routing at token **t+1**, which requires
`H[t+1, l-1]` — unavailable until token t+1. There is no honest 1-token-ahead prediction.

Corrected sim (`s2_evict_sim.py`, 1600 tokens × 40 layers × 16 slots/layer, **256 experts**):

| policy | hit rate | honest? |
|---|---|---|
| LRU (current) | 46.7% | — |
| `pred` (1-step protect) | 48.2% (+1.5) | **yes** — ~1.03×, below 1.10× gate |
| `predhist` | 45.3% | yes — worse than LRU |
| `realpred_peek` / `predbelady` | 58.5–58.9% | **no** — reads predictions made at future steps |
| Belady oracle (true future) | 58.6% | no — unobtainable |

The 58.9% "matches Belady" figure reported during the pivot was a **future-peeking
artifact** (`realpred_peek` consults the prediction stream at steps > t). The honest
1-step-ahead policy is +1.5 pts over LRU — real but below the ship gate. The 12-pt
LRU→Belady gap is real but requires true future knowledge no available signal provides.

**Conclusion:** the hit-rate lever is **near-exhausted at 640 slots** for this workload.
LRU is near-optimal among implementable policies. The read-ahead idea (the user's original
hypothesis) does not pay off on sm75 + saturated-PCIe + this model — now **measured
directly** (G2.7 prototype: k=0 baseline beats every k>0), not just argued. **Phase 2
closed negative.** The structural win that scales is **Phase 5 (batching)**; the cheap
optional win is **Phase 3 (static warm start)**.

<details><summary>Superseded design (prefetch) — kept for the record</summary>

The YALIS/Strata `wait-and-prefetch` read-ahead: move the expert copy onto a dedicated copy
stream, double-buffer GPU slots, run layer *l+1*'s router during layer *l*'s GEMV to prefetch
*l+1*'s predicted experts. Ceiling 1.545× assumed a non-saturated link. G2.1 (overlap
mechanism, 92–95%) and G2.2 (recall@8 78.1%) both PASSED, but the **real 3-layer prototype
(G2.7) measured the net effect and it is a loss at every k** (0.96×→0.84×): the prefetch
adds spurious bytes to a PCIe link already pegged by the demand copy, so there is no free
compute window to hide them under. Superseded.
</details>

### Phase 3 — Static expert-profile warm start (Strata-style). OPTIONAL, cheap
Preload the 640-slot cache with the **hottest experts for the user's actual workload**
(C/Java coding, bug hunts, docs-from-C) *before* the first token, instead of starting cold.
- Build the profile by running the engine once over representative prompts with
  `--dump-routing`-style capture (we already have the routing recorder, `s2_route_record.py`),
  rank the (layer,expert) pairs by frequency, and seed the cache at startup.
- **No accuracy impact** (static, no routing change). Raises the *early-token* hit rate
  (cold start is the worst case for LRU). Complements Phase 2 (warm start + prefetch).
- **Gate:** measure hit rate over the first N tokens with vs without the profile; accept if
  it raises the first-100-token hit rate by ≥ 5 pts without hurting steady state.
- **Rollback:** behind `--moe-profile <path>` (default off / no file = current behavior).

### Phase 4 — CPU co-computation of misses (the "Strata way"). ⚠️ OUT OF SCOPE (noted)
Strata's biggest lever — compute missed experts on the CPU (AVX-512/AVX2 ggml) in parallel
with the GPU — is a large architectural change: it needs a C++ CPU expert-compute path that
matches our NVFP4 marlin GEMV bit-for-bit, plus a `--pcie-frac` split of misses between
PCIe-stream and CPU-compute. For a PyTorch engine on sm75 this is a multi-week effort with
real correctness risk (CPU/GPU rounding divergence → output divergence, the exact class of
bug §5 is designed to avoid). **Not recommended now.** Noted for completeness; revisit only
if Phases 2+3 under-deliver and the copy remains the bottleneck.

### Phase 5 — Batching (S6). STRUCTURAL, DEFERRED
The only win that *scales*: amortize one expert fetch across N concurrent requests (the
fetched expert serves all N). This is the real path past the single-request PCIe ceiling,
but it's a separate, larger project (scheduler + KV + GDN for multi-request) already tracked
as S6. Not part of this plan.

---

## 5. Accuracy & correctness review (the non-negotiable)

- **Phase 2 (read-ahead): closed negative — no accuracy review needed** (nothing shipped;
  the prefetch variant would have had no accuracy impact by construction, the eviction
  variant never changed the computed expert set either).
- **Phase 3 (profile): no accuracy impact.** Static seed, no routing change.
- **Gates are the arbiter:** any shipped lever must be token-identical to the baseline over
  the A/B harness (greedy, temperature 0). Any divergence = stop, do not ship.
- **Invariants preserved through all phases:** 64K context, `--moe-cache-size 640`, Q8-KV,
  chunked 4096 prefill, prefill opts on `perf/sm75-prefill` protected.

---

## 6. Open questions / next experiments (short probes, per preference)

**Gate results (2026-10-05, measured):**

- **G2.2 router-prediction recall — PASS.** `s2_recall_record.py` on Qwen3.6-35B-A3B-NVFP4,
  300 decode tokens × 39 layer pairs: **recall@8 = 78.1%**, weight-covered = 85.0% at top-8;
  recall@12 = 89.1% (92.3% weight) at top-12; recall@16 = 93.2% (95.1% weight). Per-layer
  recall@8 spans 52% (L0) → 87% (L24), mean 78.1%. The next-layer router prediction from
  layer *l*'s MoE-input hidden is strong on our model — matches YALIS's Qwen3-30B-A3B
  expectation. Wasted prefetch at top-8 = 21.9% (prefetching 16 to cover 8 costs 53% waste;
  top-8 is the sweet spot).
- **G2.1 overlap mechanism — PASS (microbenchmark).** `s2_g21_overlap.py`: the SM-driven
  `fast_index_copy_multi` copy on a side stream overlaps the 24 decode GEMVs on the main
  stream at **92–95%** on sm75 (copy 1.6 ms/layer at ~12 GB/s PCIe-saturated vs compute
  0.26 ms/layer; concurrent = max(copy, compute), compute fully hidden). `blocks_per_bank`
  1→8 barely changes overlap (95% at bpb=8) — the copy is PCIe-bound, so it steals few SMs.
  End-to-end ms/tok gate (≥1.10×) needs the 2–4 layer prototype below.
- **G2.6 eviction sim (the pivot) — CLOSED NEGATIVE.** `s2_evict_sim.py` (1600 tokens ×
  40 layers × 16 slots/layer, 256 experts): the honest 1-step-ahead prediction-protect
  policy (`pred`) reaches **48.2%** vs LRU **46.7%** (+1.5 pts, ~1.03× — below the 1.10×
  gate); `predhist` (45.3%) is worse than LRU. The 58.9% "matches Belady" figure from the
  pivot was a **future-peeking artifact** (the policy read predictions made at future
  steps). Root cause: the router prediction is an **intra-token (next-layer)** signal —
  strong for prefetch (G2.2's 78.1% recall) but the wrong kind for eviction, which needs a
  **next-token** signal that does not exist. LRU is near-optimal among implementable
  policies at 640 slots. *(Note: an earlier draft of the sim used E=128; the model has
  256 experts — the corrected numbers above are from E=256.)*
- **G2.7 real 3-layer prefetch prototype — CLOSED NEGATIVE (the arbiter).**
  `s2_prefetch_proto.py`: real `fast_index_copy_multi` kernel (SM-driven, PCIe), real
  routing + prediction from the capture, 16 LRU slots/layer, 256-expert host pool, 3 layers,
  128 tokens. The prefetch for layer *l+1* (top-k of the next-layer router prediction,
  available at the start of layer *l*'s MoE) is enqueued on a side copy stream and overlaps
  layer *l*'s full window (faithful YALIS anchoring); the main stream waits for it, copies
  the demand-residual, computes. Calibration: per-expert copy 0.298 ms (bpb=2, small batch)
  vs per-layer compute 0.264 ms. Result — **k=0 (baseline, demand-only) wins at every k:**

  | k | ms/layer | vs baseline |
  |---|---|---|
  | 0 (baseline) | 1.67 | 1.000× |
  | 1 | 1.73 | 0.96× |
  | 2 | 1.73 | 0.96× |
  | 4 | 1.79 | 0.93× |
  | 8 | 1.99 | 0.84× |

  The loss grows with k because the prefetch adds **spurious** bytes (predicted but not
  routed) to a PCIe link already pegged by the demand copy — there is no free compute
  window to hide them under (prefetch and demand share the link). This confirms the
  corrected byte accounting (§3 note, §4 Phase 2a) and closes the residual uncertainty
  from G2.1. No k clears the 1.10× gate; nothing to scale to 40 layers.
  *(Caveat: G2.7's "compute" was a proxy — `torch.bmm` GEMVs at 0.26 ms/layer, matching
  production MoE GEMV timing — not the full engine; the copy side was the real
  `fast_index_copy_multi` kernel over real PCIe. The full-engine A/B (token-identical +
  ms/tok) is deferred; the 3-layer result + corrected byte accounting is sufficient to
  close Phase 2 negative. Revisit if a full-engine confirmation is wanted.)*

1. **Confirm bandwidth saturation directly** (not inferred): a microbenchmark that streams a
   known byte count CPU→GPU at cache 640 and reports effective GB/s. Expect ~13–16 GB/s
   (PCIe 3.0 x16 pegged). Confirms Lever B (overlap) is the right call over Lever A (hit rate).
   *(DONE: G2.1 microbenchmark copy = 18.9 MB/layer in 1.6 ms = ~11.8 GB/s, PCIe-saturated.)*
2. **Measure next-layer router-prediction recall@8** on the recorded stream (`s2_route_record.py`
   output). This is G2.2 and the single number that tells us how much of the 1.545× ceiling is
   actually reachable on *our* model (vs YALIS's Qwen3-30B-A3B).
   *(DONE: 78.1% recall@8, 85% weight-covered — see gate results above.)*
3. **Prototype the overlap on 2–4 layers first** (not all 40) to measure the per-layer
   copy/compute overlap fraction and the stream-sync overhead on sm75 before committing to the
   full rework. If 2–4 layers show ≥ 1.1× on their share, scale to 40.
   *(DONE — G2.7, see gate results below: the real 3-layer prototype shows k=0 baseline beats
   every k>0, so there is nothing to scale.)*

---

## 7. Rollback & safety

- Every phase is **behind a flag, default-off, byte-for-byte no-op when off**
  (`--moe-profile`).
- **Rollback path:** `git checkout backup/mtp-decode-pre-surgery` (to `cb1dda2`) or
  `git checkout perf/sm75-prefill && git branch -D perf/mtp-decode`.
- **Backup refs:** `backup/pre-open-heart-surgery` tag and `backup/mtp-decode-pre-surgery`
  branch, both at `cb1dda2`.
- **Production script invariants:** `qwen_q8.sh` keeps `--moe-cache-size 640`, `--kv-q8`,
  `--max-prefill-length 4096`; MTP stays off (no runtime cost).
- **A/B harness** is the arbiter for every "ship" decision (token-identical + tok/s).

---

## 8. Bottom line

- **The bottleneck is PCIe expert streaming (65% of decode), one CUDA C++ kernel, already
  C++, bandwidth-saturated.** Not a Python problem; a wire-limit problem.
- **The LRU cache already does read-ahead (45.7–46.7% hit at 640), and it is near-optimal
  among implementable policies.** The hit-rate lever is ~exhausted: the router prediction
  is an intra-token (next-layer) signal — strong for prefetch, the wrong kind for
  eviction (which needs a next-token signal that doesn't exist). Best honest eviction
  policy = LRU + 1.5 pts (~1.03×, below gate).
- **The read-ahead idea (YALIS/Strata, the user's original hypothesis) is CLOSED NEGATIVE
  on this hardware — measured, not argued.** The real 3-layer prototype (G2.7) shows
  prefetch loses at every k (0.96×→0.84×): it adds spurious bytes to a PCIe link already
  pegged by the demand copy, so there is no free compute window to hide them under.
  Prediction-eviction can't get the signal it needs (intra-token, not next-token).
- **Strata's 60–70% is mostly bigger-cache + warm-profile + CPU co-compute** — none of
  which fit here (no VRAM headroom for a bigger cache; CPU co-compute is out of scope).
  The transferable prefetch does not help on a copy-dominated link.
- **What's left that actually pays:** Phase 3 (static warm start — cheap, optional,
  raises early-token hit rate) and Phase 5 (batching — the structural win that scales
  past the single-request PCIe ceiling; separate project, tracked as S6).
- **Phased, gated, rollback-safe, accuracy-reviewed.** Phase 1 shipped (640 cache);
  Phase 2 closed negative; Phase 3 is the only remaining quick win.
