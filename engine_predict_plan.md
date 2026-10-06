# Engine plan — predictive expert prefetch / eviction (Phase 2, re-examined)

**Status: DRAFT — measurement-first.** The G2.7 prototype closed Phase 2 "prefetch"
negative, but it calibrated its compute/copy ratio from a **synthetic GEMM**, not the
real model. This plan (a) re-derives *why* the prototype lost, (b) answers the three
open questions against the real decode path, and (c) sequences a **real-engine**
measurement + integration so we decide on production numbers, not a stand-in GEMM.

Nothing here changes the engine until Step 1 (measure) returns a verdict.

---

## 0. The three questions (from the owner)

1. **Why is the PCIe streaming pipeline saturated?**
2. **Should the pipeline be fetch-compute-fetch-compute, with the compute window as the
   spot to send the speculative expert?**
3. **Is the demand path already doing unnecessary fetches?**

Plus the standing constraint: **integrate into the real engine, not an abstract sim.**

---

## 1. The real decode pipeline (grounded in code)

Source: `python/freetoken/layers/moe.py` (`OffloadMoELayer._decode_routed`),
`python/freetoken/moe/offload_cache.py` (`ensure_experts`, `copy_missing`),
`python/freetoken/moe/offload_kernels.py` (`ensure_experts` → flashlib `lru_ensure`),
`python/freetoken/kernel/fast_index_copy.py` (`fast_index_copy_multi_jit`).

Per token, per layer `l`, **all on the single main stream** (decode is a captured CUDA
graph; prefill is eager):

```
lru_ensure(layer l)          # device Triton kernel: LRU hit/miss, rewrites
                             #   topk_ids -> slot ids, stages evict_slots/src_indices/
                             #   num_indices. Tiny (~us).
copy_missing(layer l)        # ONE fast_index_copy_multi launch: H2D PCIe copy of the
                             #   num_indices MISSING experts, all banks. ~ms.
_expert_gemm(layer l)        # grouped GEMV over the 8 routed slots (HBM-bound read of
                             #   the slot cache). ~0.26 ms in the G2.7 calibration.
```

**Consequences (the whole analysis hangs on these):**

- **No overlap, ever.** Copy and compute are sequential on one stream. Layer `l+1`'s copy
  starts only after layer `l`'s GEMM finishes. The PCIe link is **idle during every GEMM**.
- **The copy is deliberately link-saturating.** `fast_index_copy_multi` launches
  `blocks_per_bank × num_threads` ≈ 8192 threads/bank. Its docstring: *"what saturates the
  link is the TOTAL in-flight request count … PCIe is the bottleneck, not the GPU, so few
  blocks already feed it."* So each copy **pegs the link by construction** — that is how a
  single H2D burst reaches peak PCIe BW.
- **The link is stop-go, not continuously saturated.** Saturated during each copy, idle
  during each GEMM (and during attention, GDN, and norms). The **measured** production
  split is 25.7 ms/tok copy (65 % of a ~39.5 ms/tok decode) and ~13.8 ms/tok non-copy —
  so the link is idle ~35 % of the token, and that idle is split across the GEMM,
  attention, GDN, and norm kernels, not just the GEMM. The prototype's per-expert copy
  (0.315 ms) and GEMM (0.262 ms) are **synthetic calibrations**, not the production
  values; Step 1 measures the real per-layer GEMM window and per-expert copy time.

---

## 2. Q1 — Why is the pipeline saturated?

It is **not** saturated across the token. Each *individual copy* saturates the link, by
design (the kernel over-provisions threads to keep enough H2D transactions in flight to
reach peak PCIe BW). Between copies, the link is idle.

So "saturated" = "each demand copy drives the link to its ceiling." The 25.7 ms/tok
(65 % of decode) is the **sum of the copy bursts**; the remaining ~13.8 ms/tok (≈35 %) is
the GEMMs, attention, GDN, and norms, during which the link sits idle. **That idle time is
the only place a speculative fetch could hide** — and within it, the *GEMM window* is the
only sub-interval that is both (a) free of link traffic and (b) causally after the
next-layer router output is available. That is exactly Q2.

---

## 3. Q2 — fetch-compute-fetch-compute, with compute as the speculative window?

The current pipeline **already is** fetch-compute-fetch-compute (per layer: copy then
GEMM). The question is really: *can we overlap a speculative fetch for layer `l+1` into
layer `l`'s GEMM window?* The G2.7 prototype tested precisely this and reported a loss
(k=0 beats k>0: 0.96× at k=1 → 0.84× at k=8). Here is the mechanism, re-derived:

**The prototype's choreography** (`s2_prefetch_proto.py`, `token_pass`):
1. At the start of layer `l`, on a **separate copy stream**, enqueue a prefetch of
   `pred_at(l+1, t)[:k]` (the router's top-k for the next layer, minus already-resident).
2. Layer `l`'s demand copy runs on the **main** stream; layer `l`'s GEMM runs after it.
3. Before layer `l+1`'s demand copy, `main_stream.wait_event(pf_done[l+1])` — the demand
   copy **blocks until the prefetch lands.**

**Why it loses (three compounding effects):**

1. **The prefetch is starved during the demand copy.** At the start of layer `l`, the main
   stream immediately does layer `l`'s demand copy, which pegs the link. The copy-stream
   prefetch competes for the same link and gets ~no progress. It only runs during layer
   `l`'s GEMM window — **0.262 ms**.
2. **The GEMM window is smaller than one expert copy.** 0.262 ms (GEMM) < 0.315 ms
   (per-expert copy). So even a k=1 prefetch **cannot finish inside the window**; it spills
   ~0.05 ms into layer `l+1`'s copy window. The `wait_event` then **stalls the main stream**
   for that spill. (The wait is *required* — the prefetch and the demand copy both write the
   slot cache, so racing them is a correctness bug. The stall is the price of the overlap.)
3. **The router is only ~78 % right (recall@8).** When the prediction is wrong (≈22 % of
   layers), the prefetch fetched an expert that is **not routed** — its bytes are pure
   waste, it occupies a slot (possibly evicting a useful one), and the demand copy still has
   to fetch the real misses. So 22 % of the time the prefetch is strictly negative.

**Net:** the prefetch's cost (0.315 ms + spill-stall) is paid on *every* layer, but the
benefit (one fewer demand copy) is realized only when the prediction is right *and* the
spill is small — which, given effect 2, never fully happens. Hence k=0 wins. This is a
**structural** result, not a bug in the prototype: the window is too small to hide a full
expert, and the wait that makes the overlap safe converts the spill into a stall.

**The one thing that could overturn it — and why the prototype is not final:**
the 0.262 ms GEMM and 0.315 ms copy come from the prototype's **synthetic calibration**
(`do_compute` = `torch.bmm` over random `NE=8, H=2048, I=768` weights). The *real* model's
per-layer GEMM time and per-expert copy time are **not** those numbers. The whole verdict
hinges on the ratio **GEMM_window / expert_copy**:

- If **real GEMM < real copy** (as the prototype assumes): the window can't hide an expert →
  prefetch is structurally dead. Phase 2 stays closed.
- If **real GEMM ≥ real copy** (plausible — the real GEMM reads a different shape, and the
  real copy may be faster than the synthetic 3-bank burst): the window *can* hide an expert,
  the spill vanishes, and a **corrected** prefetch (below) becomes viable.

**This ratio is not measured in the production engine. That is Step 1.**

### A corrected prefetch design (only if Step 1 says real GEMM ≥ real copy)

The prototype's `wait_event`-before-demand design is the *safe but worst* way to overlap.
If the window is big enough, a better design avoids the stall:

- **Decouple prefetch and demand on the link, coordinate on the slot cache.** The demand
  copy does *not* wait for the prefetch. Instead, the demand path and the prefetch write to
  **disjoint slot sets** (prefetch owns a reserved sub-range, e.g. the top `P` slots of each
  layer's band), and the demand copy **skips** any expert the prefetch already owns (a
  device-side `prefetch_owner[e]` flag). No wait, no stall.
  - **Dependency note:** the demand LRU logic lives in **flashlib**
    (`from flashlib.kernels.slot_cache import lru_ensure`), a separate dependency, not
    freetoken. Making it "skip prefetch-owned experts" means either (a) patching flashlib,
    (b) forking that kernel into `freetoken/moe/`, or (c) a thin device-side wrapper that
    post-filters `evict_slots`/`num_indices` after `lru_ensure`. Decide in Step 2; (c) is
    the least invasive and keeps flashlib untouched.
- **The GEMM waits only for the experts it actually uses** (event per prefetched expert),
  not for the whole prefetch — so a wrong prediction (unrouted expert) costs link bytes but
  **no main-stream stall**.
- **CUDA-graph safe:** the *which-experts-to-prefetch* decision is a **new device-side
  kernel** (the new piece), structured exactly like `lru_ensure` — reads the router output
  and `slot_for_id` on-device, fixed shapes, no host sync — so it is captured into the
  decode graph and re-evaluated at replay (the router output changes per token, but the
  graph is static). The copy stream is created once at capture; `copy_missing` already
  proves a cross-bank copy captures cleanly.
- **Gate:** accept only if it beats k=0 by ≥ 1.10× *in the real engine* (same gate as the
  rest of Phase 2). If it can't clear the gate, close it and record the real ratio.

---

## 4. Q3 — Is the demand path already doing unnecessary fetches?

**No — the demand path is clean.** Verified against the code:

- **No redundant copies.** `lru_ensure` stages only experts with `slot == -1` (misses). An
  expert already resident is never re-copied. `num_indices` = the miss count, and
  `copy_missing` copies exactly those.
- **No over-fetch.** The copy fetches all banks for each missed expert — required by the
  GEMM. Nothing extra.
- **Eviction is near-optimal.** The one way a "necessary" fetch could become unnecessary is
  the LRU evicting an expert that is needed again soon (forcing a re-fetch). The cache sim
  (`s2_cache_sim.py`) measured LRU vs Belady-optimal (the theoretical ceiling) at 640 slots
  and found the gap small — LRU is near-optimal at this size. The prediction-eviction pivot
  (Phase 2's second arm) extracted only **+1.5 pts** over LRU (~1.03×, below the 1.10× gate),
  confirming there is little eviction waste left to harvest.

So the demand path fetches exactly what it needs, with near-optimal eviction. **The
"unnecessary fetches" the owner suspects are not in the demand path** — the only waste in the
system is the *prefetch's* 22 % wrong-prediction bytes (Section 3, effect 3), which is a
property of the speculative arm, not the baseline.

---

## 5. The fundamental limit (pending Step 1 confirmation)

The per-request prefetch is bounded by **GEMM_window vs expert_copy**:

- The prefetch for layer `l+1` can start at the start of layer `l`'s MoE (the router for
  `l+1` needs `H[t, l]`, the attention output, available then). But layer `l`'s **demand
  copy runs in that same window and re-pegs the link**, starving the prefetch. The only
  time the link is free *and* causally after the `l+1` router output is layer `l`'s
  **GEMM window**. So the prefetch's effective window is **one GEMM window** — it cannot
  accumulate across layers, because each subsequent layer's demand copy re-pegs the link
  and starves it again.
- If that window < one expert copy, the prefetch spills, the safe wait stalls the main
  stream, and the 78 % recall means 22 % of prefetches are pure waste. **Net loss.**

This is a **hard per-request limit**. It does not change with a better predictor (recall
caps at ~78 % on the intra-token signal) or a better eviction (LRU is near-optimal). The
only escapes are structural, and they are the *other* phases:

- **Phase 5 (batching):** amortize one expert fetch across N concurrent requests — the fetch
  is no longer per-request, so the window constraint vanishes. The real scaling win.
- **Phase 4 (CPU co-compute):** compute the overflow misses on the CPU (high RAM BW),
  overlapped with the GPU copy — removes bytes from the link entirely. Multi-week C++ path.

---

## 6. Plan — measurement-first, then integrate

### Step 1 — Measure the real ratio (decides everything). ~0.5 day, no engine change.
Instrument the **production** engine (Qwen3.6-35B-A3B-NVFP4, `qwen_q8.sh` config, 640-slot
cache, CUDA graphs on) and record, per token. **All three measurements go through the
existing `collect_stats` / `decode_miss_stats` host-side accumulation path** (wired up in
`engine.py` for decode), read once per request — *not* NVTX. The `MoE_expert_gemm` NVTX
range exists only in `_prefill_routed`, **not** the decode path, and NVTX is host-side and
unreliable inside a captured CUDA graph, so it is the wrong tool here.
- **Per-layer GEMM window** = time of the `_expert_gemm` call in `_decode_routed`,
  accumulated host-side in the `collect_stats` path (extend it alongside the copy stats
  below). Report the median per layer.
- **Per-expert copy time** = total `copy_missing` time / total `num_indices` (miss count)
  over the token. `num_indices` is already device-side; expose it via the existing
  `collect_stats`/`decode_miss_stats` path (host-side accumulate, read once per request).
- **Link duty cycle** = (sum of copy times) / (total decode time). This replaces the
  prototype-derived ~65 % copy / ~26 % idle estimates with production numbers (the idle
  fraction is actually *larger* than the GEMM windows, since the link is also idle during
  attention, GDN, and norms — expect ~35 % idle, i.e. 100 % − copy-fraction).
- **Link utilization (peak, not just time)** — the "100 % saturated" claim currently rests
  on the kernel's over-provisioning design + a prior Phase 2 measurement (~11.7 GB/s on
  PCIe 3.0 x16). Measure it directly in the same run: sample PCIe counters during decode
  (NVML `nvmlDeviceGetPcieThroughput` / `nvmlDeviceGetUtilizationRates`, or NCU
  `pcie__bytes.sum` on the copy kernel, or a concurrent pinned-memory H2D probe on a
  separate stream) and report GB/s during copy windows vs the link's practical ceiling
  (~12.5 GB/s for PCIe 3.0 x16). This answers whether the copy time is a *floor* (link
  pegged → only byte-avoidance helps: prefetch/batching) or a *tunable* (link not pegged →
  kernel tuning / more in-flight requests is a separate lever).

Deliverable: **real GEMM_window / real expert_copy**, the link duty cycle, and the **peak
link utilization (GB/s)**, all on the actual model. **Verdict:**
- ratio < 1.0 → prefetch structurally dead. Close Phase 2 with the real numbers (replaces
  the synthetic-calibration closure). Move to Phase 5.
- ratio ≥ 1.0 → proceed to Step 2.
- **Independent of the ratio:** if peak utilization is *below* the link ceiling (not pegged),
  the copy time is tunable — open a separate lever for kernel tuning / more in-flight
  requests (does not require the prefetch). If pegged, copy time is a floor and only
  byte-avoidance (prefetch/batching) helps.

### Step 1 — RESULTS (measured 2026-10-05)

Production run: Qwen3.6-35B-A3B-NVFP4, `qwen_q8.sh` config, 640-slot cache, CUDA graphs on.
Workload: 611-token prefill + 1023 decode tokens at **23.0 tok/s** (matches the ~25 baseline).
Capture: `nsys --cuda-graph-trace=node` (kernels inside the captured graph are visible) +
`--moe-collect-stats` for miss counts. Full trace + analysis in
`~/.hermes/cache/scratch/step1/` (`STEP1_FINDINGS.md`, `prof.nsys-rep`, `kernel_data.pkl`).

**Q1 — who uses 100% of PCIe?** The MoE expert demand copy (`fast_index_copy_multi`) is the
consumer; it is the only significant H2D traffic in decode (KV / attention / GDN / norms all
sit in GPU HBM). **Peak link utilization during the copy: 12.54 GB/s = 100% of the PCIe 3.0
x16 ceiling (~12.5 GB/s). The link is genuinely pegged during each copy burst** — copy time
is a floor, so only byte-avoidance (prefetch / batching) helps; kernel tuning is not a lever.

**Link duty cycle:** total copy time 25,050 ms of 40,830 ms decode = **61.4% of decode time**
in expert copy; the other 38.6% is idle (GEMM, attention, GDN, norms). Confirms the ~65% copy
/ ~35% idle production split.

**Per-layer timing (median, decode):**
| quantity | value |
|---|---|
| copy (`fast_index_copy_multi`, one launch/layer of all missed experts) | **590 us** |
| MoE GEMM window (marlin + gemv + reduce + act + moe_sum, after the copy) | **225 us** |
| full link-free gap between consecutive copies | **322 us** |
| per-expert copy (590 us / 4.179 experts) | **141 us** |
| experts per layer (misses) | **4.179** (miss rate 51.8%) |
| expert size (NVFP4, H=2048, I=512) | **1.77 MB** |

**Decisive ratio (window / copy):**
- per-expert: GEMM window / per-expert copy = 225 / 141 = **1.55** (≥ 1.0)
- per-layer: full gap / full-layer copy = 322 / 590 = **0.55** (< 1.0)
- per-layer: GEMM window / full-layer copy = 225 / 590 = **0.38** (< 1.0)

**Interpretation — the prototype's verdict does not directly transfer, and a window subtlety
needs Step 2 to resolve:**
1. The prototype (G2.7) used a **synthetic expert of 768 KB / 315 us** (2.44 GB/s, *not*
   link-saturated). The real NVFP4 expert is **1.77 MB / 141 us** (12.5 GB/s, *fully*
   link-saturated) — a **2.2× larger expert** but **2.2× faster per byte**. The prototype's
   negative result (k=0 beats k>0) was driven by its small synthetic expert, not a structural
   limit. The real engine is more favorable to the prefetch.
2. The **window definition** is the crux. The plan (Section 5) said the effective window is
   "one GEMM window" (225 us), giving 0.38 < 1.0 (dead). But the G2.7 prototype enqueues the
   prefetch for layer `l+1` at the **start of layer `l`'s MoE** (the router for `l+1` needs
   `H[t, l]`, available then) and overlaps it with layer `l`'s **full window** (demand copy +
   GEMM ≈ 815 us), not just the GEMM. If the real effective window is the full MoE window,
   the per-layer ratio is **815 / 590 = 1.38 (≥ 1.0, viable)**; if it is only the GEMM window
   (because layer `l`'s demand copy re-pegs the link and starves the prefetch), it is 0.38.
   **Step 2 must measure the real effective prefetch window (when the prefetch can actually
   start, and how much link it gets while the demand copy is running) before committing.**

**Verdict:** The per-expert ratio is **1.55 (≥ 1.0)** and the link is a floor (100% pegged).
The real engine is more favorable to the prefetch than the prototype suggested. **Proceed to
Step 2** to implement the corrected prefetch behind `--moe-prefetch-k` (gated 1.10×), with the
first task being to **measure the real effective prefetch window** (full MoE window vs GEMM
window, and link sharing between the demand copy and the prefetch). If the effective window is
the full MoE window (ratio 1.38), the prefetch is viable; if it is only the GEMM window
(ratio 0.38), close Phase 2 with the real numbers.

### Step 2 — RESULTS (effective window measured in the real engine, 2026-10-05)

The window ambiguity is **resolved**, and both Step 1 hypotheses were wrong. Measured by
running a real concurrent prefetch probe in the real decode path (a 1-expert H2D copy,
same `fast_index_copy_multi` kernel + 1.77 MB geometry + pinned source, on a second
stream, enqueued at the start of each layer's MoE so it runs **concurrently with the
demand copy** — the G2.7 choreography; dst in a disjoint scratch buffer; graph-captured).
Full record + probe code in `~/.hermes/cache/scratch/step2/` (`STEP2_FINDINGS.md`).

| quantity | alone (Step 1) | concurrent (Step 2) |
|---|---|---|
| demand copy (4.18 experts) | 590 µs | **740 µs** (+150 µs) |
| probe (1 expert, 1.77 MB) | 141 µs | **401 µs** (2.8× slower) |

- **97.8%** of probe copies overlap the concurrent demand copy — the prefetch runs
  **during the demand copy window, not the GEMM window**. 90% finish inside it; 10% spill.
- **The link is pegged: the probe extends the demand copy by exactly its own byte count**
  (590 + 1.77 MB/12.5 GB/s = 732 µs predicted; 740 µs observed). The demand copy's
  over-provisioned threads win link arbitration, so the probe runs at ~3.7 GB/s, not 12.5.
- **Net cost of a 1-expert prefetch (right prediction): +9 µs (slower).** At 78% recall:
  **+40 µs/layer (slower)**. e2e: 19.4 tok/s (probe) vs 23.0 (baseline).

**Verdict — Phase 2 prefetch is structurally dead, for a stronger reason than G2.7.**
Neither the "full MoE window (1.38, viable)" nor the "GEMM-only window (0.38, dead)"
hypothesis holds: the prefetch does **not** get a free window. On a **pegged link**,
prefetching an expert into an already-saturated window is a **no-op for total bytes and
pure latency** — it fetches bytes the demand copy would have fetched anyway (Q3: no
redundancy to eliminate), so it **cannot reduce the 61.4% copy duty cycle**; it only adds
latency, and the 22% wrong-prediction bytes are strictly negative. A better predictor
(recall caps at 78%) or better eviction (LRU near-optimal) cannot fix this. This is a
**harder** dead-end than G2.7's "GEMM window < expert copy": even an arbitrarily large
GEMM window would not help a concurrent prefetch on a pegged link with no redundant
bytes.

**The only structural escape is Phase 5 (batching)** — amortize one expert fetch across
N concurrent requests, which **does** reduce total H2D bytes per token. That is the real
win. **Close Phase 2 (prefetch) with these real numbers; point the plan at Phase 5.**

### Step 2 — Corrected prefetch in the real engine (only if ratio ≥ 1.0). ~2–3 days.
Integrate the Section 3 design behind `--moe-prefetch-k <k>` (default 0 = today):
- Create one **copy stream** at graph capture; enqueue the next-layer prefetch at the start
  of layer `l` (device-side decision from the router output + `slot_for_id`).
- Reserve a **disjoint prefetch slot sub-range** per layer; add a device-side
  `prefetch_owner` flag; make the demand path skip prefetch-owned experts (see the
  flashlib dependency note in Section 3 — decide patch/fork/wrapper here) and the GEMM
  wait per-used-expert (no whole-prefetch stall).
- **CUDA-graph capture** must pass (fixed shapes, no host sync) — verify with the existing
  graph-capture test path.
- **Gate:** ≥ 1.10× over k=0 in the real engine, bit-identical outputs (the prefetch only
  changes *when* bytes arrive, not *which* — verify with the existing output-diff harness).

### Step 3 — Decide and record.
- If Step 2 clears the gate: ship behind the flag, document the real ratio that made it work.
- If not: close Phase 2 prefetch with the **real** ratio + the corrected-design result, and
  point the plan at **Phase 5 (batching)** as the only per-request→structural win.

### Step 4 (parallel, independent) — Phase 5 scoping.
Regardless of Step 2, batching is the structural win. Scope the scheduler + KV + GDN
multi-request work as a separate project (S6). This does not depend on the prefetch verdict.

### Step 5 — RESULTS (Phase 5 batching measured in the real engine, 2026-10-05)

**The engine already has full batching** — no build needed: CUDA graphs at
`[1,2,4,8,...]`, batched split-k decode attention, batched GDN, and a **single shared**
MoE cache whose `lru_ensure` copies the **per-union** unique-missing experts per step.
The production config just pins `--max-running-requests 1`. So the only open question was
empirical: do concurrent requests route to enough of the same experts that the shared
cache dedupes the H2D? Measured with the real engine's `--moe-collect-stats` counters at
N ∈ {1,2,4} concurrent requests (distinct prompts, 512 tokens each).

| bs | miss/step/layer | per-token copy | amort ratio | expert overlap | miss rate | agg tok/s |
|---|---|---|---|---|---|---|
| 1 | 4.240 | 4.240 | **1.000** | — | 53.0% | 22.2 |
| 2 | 8.976 | 4.488 | **1.058** | 7.1% | 60.4% | 12.4 |
| 4 | 26.715 | 6.679 | **1.575** | 16.0% | 99.4% | 13.6 |

**Verdict — Phase 5 batching does NOT help; it makes things worse.** Per-token copy cost
*increases* with N (4.24 → 4.49 → 6.68 experts/token); the amortization ratio is ≥ 1.0 at
every N. Two compounding reasons: (1) **low expert overlap** — different prompts route to
mostly-disjoint experts (7.1% at bs=2, 16.0% at bs=4), so the shared cache dedupes almost
nothing; (2) **LRU thrashing** — the per-step working set (N × 8 × 40 layers) exceeds the
640-slot cache (bs=4: ~1072 vs 640 → 99.4% miss). Per-step copy time balloons 24 ms →
151 ms, so the pegged link is the binding constraint: aggregate throughput *drops*
(22.2 → 12.4 → 13.6 tok/s) and per-request latency worsens 7×.

**Why the "structural win" assumption was wrong:** batching amortizes the fetch only *if
the requests share experts*. Qwen3.6's router does not (7–16% overlap on diverse prompts)
and the cache can't hold the union working set. The verdict is **load-dependent**: it
could flip only under high expert overlap (same-task/same-topic request mixes, or a
skewed routing distribution) and/or a cache large enough to hold the per-step union.

Full record + scripts: `~/.hermes/cache/scratch/step5/` (`STEP5_FINDINGS.md`,
`run_serve.sh`, `workload.py`, `workload_N{1,2,4}.json`).

### Step 6 — RESULTS (cache-size knob measured in the real engine, 2026-10-06)

The 640-slot `--moe-cache-size` in `qwen_q8.sh` was checked: it is **VRAM-bound at the
64K KV budget, not arbitrary** (git `d153424`: "512->640 (fits 64K chunked prefill; 992
OOMs)"). Bigger cache = fewer H2D misses, but the VRAM comes from the KV token budget.
Measured frontier (bs=1, real engine, `--moe-collect-stats`):

| config | fits | miss rate | missing/step/layer | e2e tok/s |
|---|---|---|---|---|
| 640 slots @ 64K (production) | yes | 53.0% | 4.24 | 22.2 |
| **1280 slots @ 64K** | **yes** | **36.0%** | **2.88** | **25.4** |
| 2048 slots @ 64K | **OOM** | — | — | — |
| **2048 slots @ 16K** | **yes** | **24.2%** | **1.93** | 24.3 (511 tok) / **35.9** (2047 tok) |

**Verdict — the cache-size knob is the biggest real win found on this project.** Up to a
55% cut in the per-step copy floor, decode 22.2 → 25.4 tok/s at the full 64K budget
(1280 slots), or 33–37 tok/s if the workload tolerates 16K context (2048 slots). No code
change. The router is sticky (implied expert working set grows with cache: 339 experts at
640 slots, 496 at 2048), so the LRU catches recurring experts as the cache holds more
history. It is a **KV-vs-cache trade at 8 GB** (per 1024 extra slots = +1.81 GB).
Ship 1280 @ 64K (safe, full context) or 2048 @ 16K (max decode, if context allows).
Phase 4 (CPU co-compute) remains open, but its motivation (copy floor) just dropped up to
55% — re-evaluate before starting the multi-week C++ path.

Full record + scripts: `~/.hermes/cache/scratch/step5/CACHE_SIZE_FINDINGS.md`,
`run_serve_cache.sh CACHE TOKS`.

### Step 6b — RESULTS (CUTLASS extend-path VRAM guard at 1280, 2026-10-06)

Question: does the bigger 1280-slot cache break the CUTLASS long-prefill path, which
bails to Triton when the GQA expansion can't fit in 60% of *free* VRAM
(`cutlass.py`, chunk-2+ extend path only — fresh prefill is unaffected)?

Instrumented the guard (`FREETOKEN_CUTLASS_GUARD_LOG=1`, temporary, reverted) and drove
real chunked prefills at 1280 @ 64K:

- **5782-token prompt: CUTLASS OK.** Chunk 1 (fresh, 4096) + chunk 2 (extend,
  device_len=5782): `peak=0.118GB < 60%free=0.218GB` → guard passes.
- **60020-token prompt: CUTLASS bails to Triton from chunk 2 on.** Free VRAM during
  prefill at 1280 is only ~0.30 GB (resting 0.91 GB − prefill activations). Peak grows
  with device_len: 0.20 GB @ 8K → 1.21 GB @ 57K, all > 0.18 GB = 60%×0.30 GB → every
  extend chunk bails. (60K extend then crawls at ~70–120 tok/s on Triton.)

**Verdict — 1280 did not newly break anything; the guard threshold is prefill-activation
limited, not cache limited.** The bail boundary is where `peak(device_len) ≤ 0.6 ×
free_during_prefill`. At 1280, free during prefill ≈ 0.30 GB → CUTLASS extend is only
viable up to device_len ≈ 6–8K (chunk 1 + maybe chunk 2); beyond that it was *always*
Triton, at 640 or 1280 (640's extra ~1.1 GB of free VRAM only moves the boundary to
~11K). The decode win (the whole point of the 1280 change) is unaffected: decode
attention is a different, much smaller path. Trade-off accepted: long-context (>~8K)
prefill stays on Triton. If long-prefill speed ever matters, the fix is cutting prefill
activation memory (e.g. smaller `--max-prefill-length` chunks don't help — peak scales
with device_len, not chunk size), not the MoE cache.

---

## 7. Risks / open questions

- **Synthetic calibration is the whole risk.** The "prefetch is dead" conclusion rests on a
  `torch.bmm` stand-in GEMM. Step 1 exists to replace that with production numbers before we
  spend Step 2.
- **HBM contention:** the prefetch writes the slot cache (HBM) while the GEMM reads it.
  The GEMM is compute-bound (reads only the 8 routed rows of a 640-slot cache), so the
  contention should be small — but Step 2 must confirm the GEMM doesn't slow when the
  prefetch is active.
- **Prediction ceiling:** the intra-token router signal caps at ~78 % recall@8. A higher
  recall would help, but there is no next-token signal available, so 78 % is the ceiling.
- **CUDA-graph capture** of a cross-stream prefetch is the main engineering risk in Step 2;
  the existing `copy_missing` already captures a cross-bank copy, so the mechanism is
  proven, but the *event choreography* (per-expert waits) is new.

---

## 8. Bottom line

- **Q1:** each copy saturates the link by design; the link is idle ~35 % of the token
  (the GEMM, attention, GDN, and norm kernels), not just the GEMM windows. Step 1 also
  measures peak link utilization (GB/s) directly, to confirm "saturated" is a floor and not
  a tunable.
- **Q2:** compute *is* the only free window, and the G2.7 prototype did target it — but
  Step 2 (real engine) shows the prefetch **doesn't get a free window at all**: on the
  pegged link it runs concurrently with the demand copy and extends it by its own byte
  count (net +9 µs right / +40 µs/layer at 78% recall — slower). **Phase 2 prefetch is
  closed dead with real numbers**; the structural win is Phase 5 (batching).
- **Q3:** the demand path is not doing unnecessary fetches — it copies only misses, with
  near-optimal LRU eviction. The only waste is the prefetch's 22 % wrong-prediction bytes.
- **Next action (updated 2026-10-06):** Phase 2 (prefetch) **closed** (Step 2) and
  Phase 5 (batching) **closed** (Step 5) — both measured dead in the real engine. All
  three decode-side structural levers (prefetch / prefill warm start / batching) are now
  closed with real numbers. **The win is the `--moe-cache-size` knob** (Step 6): the
  640-slot cap was VRAM-bound at the 64K KV budget, not arbitrary. Measured frontier —
  640 slots @ 64K (22.2 tok/s, 53% miss) → **1280 slots @ 64K (25.4 tok/s, 36% miss)** →
  **2048 slots @ 16K (33–37 tok/s, 24% miss)**; 2048 @ 64K OOMs. It's a KV-vs-cache
  trade at 8 GB. Ship 1280 @ 64K (safe, full context) or 2048 @ 16K (max decode, if
  context allows). Remaining: **Phase 4 (CPU co-compute)** — still open, but the copy
  floor just dropped up to 55%, so re-evaluate whether the multi-week C++ path is worth
  it before starting.
