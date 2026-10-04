# open_heart_surgery.md — In-engine MTP decode loop for FreeToken (Qwen3.6-35B-A3B-NVFP4, RTX 2080 sm75)

**Goal.** Wire the *proven* MTP draft step (Phase 4 A2: token-identical, acceptance
0.853; production fused head committed `8db04f8`/`cb1dda2`) into the engine's live
decode path so a request commits **1–2 tokens per decode step** instead of 1.
Target: ~1.6–1.8× decode (24 → ~38–43 tok/s), measured by the production A/B harness.

**This is the only remaining Phase 4/5 work.** Everything else is committed and gated
behind `--mtp` (default-off, byte-for-byte no-op when off).

---

## 0. Why this is "open heart" (the risks)

The scheduler and engine are built on an invariant: **exactly one token is committed
per request per decode step.** MTP breaks that invariant. Concretely, the surgery
touches four load-bearing subsystems, each a place a bug becomes silent corruption:

| # | Subsystem | Why it's load-bearing |
|---|-----------|-----------------------|
| R1 | **Commit path** (`scheduler._process_last_data`, `Req.complete_one/append_host`) | Hardcoded one-token-per-req. MTP must commit 1 *or* 2. A miscount corrupts KV offsets, the radix cache, and the client stream. |
| R2 | **GDN (GatedDeltaNet) recurrent state** (30 of 40 layers) | The 2-token verify advances conv+SSM state by 2. On reject we must roll back to the pre-verify state and replay the 1 accepted token. Wrong = every later GDN layer reads poisoned state. (Phase 2 A1 proved the rollback/replay primitive bit-exact; we reuse it.) |
| R3 | **KV page allocation** | A 2-token verify needs 2 free slots at the request's `out_loc`; the scheduler's decode allocation assumes 1. Under-allocation = OOB page write. |
| R4 | **CUDA graph decode path** | `GraphRunner.can_use_cuda_graph` is `is_decode and bs<=max`. The 2-token verify is an *extend* (not a 1-token decode), so it runs eagerly (prefill path), not through the captured decode graph. That's correct (A2 used exactly this) but means the verify is ~1.5–2× a decode step in wall time — the speedup comes from the *amortized* MoE weight read over 2 tokens, not from a faster verify. |
| R5 | **fp16 tie divergence** | The decode path (1-token graph) and the verify path (2-token extend) use different kernels → fp16 rounding differs → a near-tie argmax can flip. Observed: 1-in-102 divergence at token 42 (A2 ran only 24). This is *not* a head bug (head argmax-validated); it's inherent. The correctness gate must tolerate rare ties, not require bit-identity. |

**Non-negotiable constraints (from prior phases, must stay true after surgery):**
- 64000-token context. ~2 GiB VRAM headroom for 61K CUTLASS prefill.
- `--moe-cache-size` never silently shrunk below 512 / overlap-off.
- Prefill optimizations on `perf/sm75-prefill` (`4d5d0fb`) stay intact and protected.
- `--mtp` default-off ⇒ byte-for-byte identical to current behavior.

---

## 1. Patient vitals (current state, verified)

- Branch `perf/mtp-decode`, HEAD `cb1dda2`.
- Committed & gated: production `Qwen3_5MTP` head (fused nvfp4, GPU-resident 256
  experts, 0.53 GB) + `load_mtp_weights`; `--mtp/--mtp-draft/--mtp-head` flags;
  `Engine._init_mtp_head` (loads head before KV sizing so its VRAM is subtracted).
- Proven standalone: `phase4_prod_speed.py` runs the §4 k=1 loop end-to-end in the real
  engine — acceptance 87/102 = **0.853**, 2/3 prompts token-identical.
- **Known-good reference (the "healthy heart"):** `phase4_prod_speed.py`'s `mtp_loop`
  is the oracle. Any in-engine implementation must match it token-for-token on the
  same prompts before we trust it.

---

## 2. The backup (in case the patient becomes unresponsive)

Already done and verified:

```
git tag    backup/pre-open-heart-surgery   cb1dda2   # immutable checkpoint
git branch backup/mtp-decode-pre-surgery   cb1dda2   # movable backup branch
```

**Recovery runbook (if anything goes wrong):**

1. **Soft stop (no code lost):** the feature is behind `--mtp`. If the in-engine loop
   misbehaves, simply *don't pass `--mtp`* — the engine runs the exact current decode
   path. This is the first line of defense and requires no git at all.
2. **Revert the surgery (keep the head):** if the decode-loop commits are bad, revert
   only the engine/scheduler commits, keep the head module:
   ```
   git revert <surgery-commits...>      # or
   git reset --hard backup/pre-open-heart-surgery   # nukes surgery, keeps cb1dda2
   ```
3. **Full flatline (total loss of the branch):**
   ```
   git reset --hard backup/pre-open-heart-surgery   # back to the last known-good MTP state
   git checkout perf/sm75-prefill && git branch -D perf/mtp-decode   # back to protected prefill
   ```
4. **GPU hang (CUDA graph corruption / OOM wedge):** `nvidia-smi` then
   `sudo fuser -k /dev/nvidia*` to drop all GPU processes, confirm 58 MiB idle,
   relaunch. Never `pkill -f <pattern>` from a shell whose own cmdline contains the
   pattern (killed our shell before, exit -9).
5. **After any recovery:** re-run the Phase 0 baseline (`c0dc5a2` harness) to confirm
   24 tok/s is back, and re-run `phase4_module_test.py` to confirm the head still loads.

**Backup hygiene during surgery:** commit *every* passing gate as its own commit on
`perf/mtp-decode` (never a pile of uncommitted work). The backup tag is the floor;
each gate commit is a higher floor.

---

## 3. What strata contributes (and what it does not)

Strata lives at `/home/lmeek/.hermes/cache/scratch/strata` (C++/CUDA, iq3_xxs dynamic
quants, CPU expert pool, multi-GPU layer split). **Its code is not portable** to
freetoken (different language, quant, and engine). But it is the **authoritative design
reference** for exactly the hard part — the verify window + GDN commit. Read these:

| Strata file | What to steal (the *idea*, not the code) |
|-------------|-------------------------------------------|
| `include/strata/core/verify.hpp` | **The verify-window contract** (lines 1–19): "T tokens through all layers in ONE graph; token t's argmax is what plain greedy would produce **bit for bit**; every kernel is either the single-token kernel or a multi-token kernel whose per-token arithmetic is the single-token kernel's." This is the correctness invariant our 2-token verify must satisfy. |
| `include/strata/core/verify.hpp` + `src/core/verify.cpp` (`commit`, `gdn_conv_commit`, `capture_commit`, ~lines 956–1010, 1326–1341) | **GDN commit = replay-from-snapshot, not in-place advance.** "The window appends K/V … and leaves the GDN state untouched. `commit(n_keep)` advances the GDN conv history and recurrent state **by replaying those tokens from inputs the window stored**." This is *exactly* our Phase 2 A1 rollback/replay primitive — strata independently arrived at the same design, which validates our approach. |
| `include/strata/core/mtp.hpp` (lines 1–49) | **Head sizing before the VRAM/expert tier is sized** ("Call before the VRAM expert tier is sized: this takes ~0.9 GB"). Confirms our decision to load the head *before* KV sizing in `_init_mtp_head`. Also: "all 512 routed experts live in VRAM (708 MB) and run through the grouped hit kernels" — validates our GPU-resident fused-head choice. |
| `include/strata/core/expert_cache.hpp` (lines 1–30) | **Expert-cache sizing is measurement-driven, not assumption-driven** (they measured 6414 MiB free at steady state vs 5413 MiB for the slots; they even note "with the cache on and nothing consuming it, the engine is slower by the fill cost and faster by nothing — that's why it defaults to 0"). Lesson: *measure* the head's real VRAM + cost on this live engine before claiming a speedup; don't trust the 0.8548 alpha projection alone. |
| `src/spec/controller.cpp` (lines 16–69) | **Adaptive window policy**: they start at k drafts and grow/shrink the window from *observed* per-position acceptance (prior 0.86 for MTP). For us: start `--mtp-draft 1` (k=1, the proven config); only consider k=2 after k=1 is measured, using real acceptance. |
| `src/core/verify.cpp` `set_sampling` / `head_sampling` | **Sampling runs outside the captured graph** (parameters would bake in). Relevant if we ever do non-greedy MTP; out of scope now (greedy only). |

**Bottom line on strata:** it does *not* give us code, but it confirms our design is
the right one (verify-window + replay-from-snapshot GDN commit + GPU-resident experts +
measure-don't-assume). The one thing it does that we deliberately *skip* is the
multi-GPU layer split and the CPU expert pool — irrelevant to a single sm75 GPU with
GPU-resident nvfp4 experts.

---

## 4. The approach (minimal-surgery, staged)

**Core decision: do NOT rewrite the scheduler's overlap loop.** Instead:

- Add a **new engine method** `Engine.mtp_decode_step(req, ...)` that runs the *entire*
  §4 k=1 MTP loop for one request (draft → 2-token verify → commit 1 or 2 tokens),
  returning the committed token ids + acceptance stats. It reuses the proven
  `phase4_prod_speed.py::mtp_loop` logic, refactored to operate on a live `Req` and the
  engine's real `page_table` / `linear_state_pool` / `embed_tokens` / `lm_head`.
- The **scheduler** calls it *instead of* `engine.forward_batch` for MTP requests, and
  its commit path learns to commit 1 or 2 tokens.
- **Gate to single-request decode batches first** (`batch.size == 1`, decode phase,
  `--mtp`). Multi-request MTP batches are a later, separate stage (S6) — the overlap
  scheduler's multi-req bookkeeping is where most of the R1 risk lives, and isolating
  it to size-1 first shrinks the blast radius.

This keeps the existing decode graph, sampler, and multi-request path **untouched** for
the non-MTP case (byte-for-byte no-op when `--mtp` off).

### The MTP step (what `mtp_decode_step` does, per request, k=1)

State at entry: request has `cached_len` committed tokens; we already have the
pre-final-norm hidden `h` and the greedy next token `nxt` from the last decode step
(these come from the normal decode step — see S2/S3 for the hidden tap).

```
loop until the request's remaining budget is met:
    draft  = argmax( head.forward(h, nxt, pos=..., embed_tokens, lm_head) )   # 1 draft
    snap   = GDN snapshot of the request's linear_slot (copy_from live -> scratch)
    logits, pre2 = base_forward([nxt, draft], cached_len)   # 2-token EXTEND (eager, prefill path)
    t_c2 = argmax( lm_head( pre2[0] ) )   # model's pick after nxt
    t_c3 = argmax( lm_head( pre2[1] ) )   # model's pick after draft
    if t_c2 == draft:                      # DRAFT ACCEPTED -> commit 2 tokens
        commit(nxt, draft);  cached_len += 2
        h = pre2[1];  nxt = t_c3;  acc += 1
    else:                                  # REJECTED -> commit 1 token
        commit(nxt);  cached_len += 1
        GDN restore: copy_from(scratch -> live)   # roll back the 2-token verify state
        _, pre1 = base_forward([nxt], cached_len - 1)   # replay the 1 accepted token
        h = pre1[-1];  nxt = t_c2;  rej += 1
```

- `commit(a, b)` = append to `req.input_ids`, advance `cached_len/device_len` by the
  count, write the KV `out_loc` for the new positions, and (for the 2-token case) the
  GDN state is *already* advanced by the verify's extend (no replay needed on accept).
- The **accept path advances GDN by 2 for free** (the verify extend did it); the
  **reject path** is the only one that needs rollback+replay (Phase 2 A1 primitive).
- `base_forward` here is the eager extend (R4): it runs the full 40-layer model for 2
  tokens. That's the cost; the win is the MoE dense weights are read once for 2 tokens.

### The hidden-state tap (R-prerequisite)

`mtp_decode_step` needs the **pre-final-norm hidden** `h` at the last committed token.
The engine's `forward_batch` returns only sampled tokens. We add a **clean, gated tap**:
a small hook on the base model's final norm (`forward_add_residual`) that, *only when
`--mtp` is on and the request is an MTP request*, stashes the summed residual
(`residual + x` = pre-final-norm hidden) into a per-request buffer. When `--mtp` is off
the hook is a no-op (no allocation, no copy). This replaces the A2 harness's
monkeypatch with a real, engine-owned tap.

---

## 5. Staged plan with gates and test steps

Each stage: **implement → gate (must PASS) → commit → (only then) next stage.**
A failed gate = stop, diagnose, do NOT proceed. Every commit advances the floor.

### S0 — Baseline re-confirmation (vitals before incision)
- **Do:** run the Phase 0 baseline harness on `cb1dda2` (no `--mtp`).
- **Gate:** mixed-512 ≈ 24.49 tok/s (±3%), 64K context loads, ~2 GiB headroom.
- **Test:** existing Phase 0 harness; record numbers in this file.
- **Commit:** none (just a recorded checkpoint).
- **RESULT: PASS (2026-10-04).** Server: 64K KV 0.62 GiB, 2.72 GiB free after
  graphs, `mtp=False`. Medians vs Phase 0: mixed-512 23.87 (−2.5%), mixed-2048
  23.06 (−3.8%), agent-envelope 22.51 (−3.6%), prefix-cached 24.62 (−2.8%). All
  within normal run-to-run variance (Phase 0 mixed-512 ranged 24.01–24.73). Decode
  path unchanged and healthy. Results: `artifacts/sm75-prefill/results-s0.jsonl`.

### S1 — Hidden-state tap (no behavior change)
- **Do:** add the gated pre-final-norm hidden tap to the base model's final norm;
  store into a per-request buffer only when `--mtp` and the req is an MTP req.
- **Gate:**
  - `--mtp` off: `git diff` shows the tap is a no-op; run baseline → **identical** 24 tok/s,
    and a token-identity check (greedy 64 tokens, two runs) is bit-identical.
  - `--mtp` on: the tapped `h` matches the A2 harness's monkeypatched `pre[-1]`
    (max |Δ| = 0, since it's the same tensor).
- **Test:** small script comparing tapped `h` vs A2's `pre[-1]` on 3 prompts.
- **Commit:** `S1: gated pre-final-norm hidden tap (no-op when --mtp off)`.
- **RESULT: PASS (2026-10-04).** Tap added to `Qwen3_5Model.forward` (the `_`
  that discarded the residual-sum is now a gated `mtp_hidden_sink`); armed in
  `Engine._init_mtp_head` (only runs when `--mtp`). `fused_add_rmsnorm` is
  in-place, so `residual` after the call *is* the pre-final-norm hidden (what A2
  captured as `summ`).
  - `--mtp` ON (`s1_tap_on.py`): sink armed + wired; 1-token decode sink and
    2-token extend sink both **bit-exact (max|d|=0.000e+00)** vs the independent
    A2 monkeypatch capture.
  - `--mtp` OFF (`s1_tap_off.py`): sink stays `None`, forward is a no-op, greedy
    decode still works.
  - **Graph caveat (resolved):** the MTP decode loop is entirely eager (verify =
    2-token extend, reject = 1-token replay; both bypass the graph runner, as the
    A2 oracle does). The "normal" 1-token decode that uses the captured graph is
    exactly what the MTP step replaces, so the MTP path never replays a graph and
    the eager tap is sufficient. The sink is armed before `GraphRunner` is built
    (line 345 < 416), so capture with `--mtp` on is harmless.

### S2 — `Engine.mtp_decode_step` (the core, standalone)
- **Do:** implement `mtp_decode_step(req, ...)` per §4, reusing the proven
  `phase4_prod_speed.py::mtp_loop` logic, operating on a live `Req` + engine state.
- **Gate (correctness — the heart of the surgery):**
  - **Token-identity vs the oracle:** run `mtp_decode_step` and the standalone
    `mtp_loop` on the *same* 3 prompts × 64 tokens, same seed. They must be
    token-identical (both are the §4 loop). Any divergence = a bug in the refactor,
    stop.
  - **Acceptance in range:** 0.80 ≤ acceptance ≤ 0.90 (A2 was 0.853).
  - **GDN rollback bit-exactness (reject path):** force rejections (e.g. a draft that
    the model rejects) and confirm the post-replay GDN state equals the state from a
    plain 1-token decode of the same token (Phase 2 A1 primitive, re-asserted in-engine).
  - **No `--mtp`-off change:** with `--mtp` off, `mtp_decode_step` is never called;
    baseline unchanged.
- **Test:** new `phase5_engine_step_test.py`: builds engine + head, runs
  `mtp_decode_step` per prompt, compares to the oracle `mtp_loop`, prints acceptance +
  a GDN-state hash after each reject.
- **Commit:** `S2: Engine.mtp_decode_step (draft->verify->commit/rollback, k=1)`.

### S2 investigation findings (2026-10-04, UNCOMMITTED — read before continuing)

**TL;DR: MTP at k=1 does NOT speed up decode on this model. Clean measurement gives
0.78–0.91× (i.e. *slower* than greedy). The earlier "1.16–1.36×" I reported was a
measurement artifact — the S2 gate's reference loop included `gdn_hash` D2H copies
that slowed the *reference* more than MTP. Root cause: the 2-token verify does not
amortize — it costs ~2× a 1-token decode because the MoE experts are loaded
on-demand *per token*, so 2 tokens load ~2× the experts. This is a structural
limitation of MTP at k=1 on an on-demand-MoE model, not a bug. The original
~1.6–1.8× goal is not reachable with MTP at k=1 on this model.**

**Clean perf numbers (real varying tokens, prefill subtracted, no gdn_hash overhead):**

| Component | Cost |
|---|---|
| ref 1-token decode (D) | 93 ms/tok |
| head (H, draft) | 3 ms |
| d1 / d2 (1-token decode, two-decode variant) | 87 ms each |
| 2-token verify (V, two-token-extend variant) | 190 ms |
| replay (R, 1-token, reject-only, two-token-extend variant) | 87 ms |

Measured acceptance ~0.75–0.94 (varies by prompt).

**Speedup, by variant (decode-only):**
- Two-token-extend + replay: accept=(H+V)/2=96.5, reject=H+V+R=280 → ~0.75–0.91×.
- Two-decode (no replay): accept=(H+d1+d2)/2=88.5, reject=H+d1+d2=177 → ~0.91× at p=0.85.
- Both are **below 1.0×**. The two-decode variant is strictly better and is the one
  currently in the tree.

**Root cause — the verify does not amortize (the crux):**
- On a *dense* model a 2-token forward is memory-bandwidth-bound (weights loaded once,
  shared by both tokens) → V ≈ D → MTP would win.
- On *this* MoE model the experts are loaded on-demand **per token**. 2 tokens route to
  ~16 experts (vs ~8 for 1 token), each a separate GEMV → V ≈ 2D. Measured V=190ms ≈
  2×D(93ms). Confirmed.
- Consequence: MTP at k=1 does 2 base forwards per step for 1–2 tokens ≈ 2× the work
  per token at best → it **can never exceed 1.0×** on this model, regardless of
  acceptance or head cost. (The head is cheap, H/D≈0.03, so it's not the issue.)

**The two-decode variant (currently in the tree) has a snapshot BUG I introduced:**
- Idea: split the 2-token verify into two 1-token decodes (d1, d2). A reject then only
  needs a GDN snapshot restore (no replay), and the path uses the same 1-token decode
  kernel as greedy → should make the GDN state **bit-identical** to greedy (resolves R5).
- Bug: the GDN snapshot is taken **before d1**, but on reject the state must be
  **after d1** (S_{C+1}), not before d1 (S_C). As written, a reject rolls GDN back one
  step too far → every subsequent hidden state is computed with the wrong GDN state →
  cascading divergence → the draft is almost always wrong (prompt1 acceptance collapsed
  to 0.016 in the probe). **Fix: move `pool.copy_from(live, scratch)` from before d1 to
  before d2.** Then a reject restores S_{C+1}, correct for committed=C+1.
- Even fixed, the two-decode variant caps at ~0.91× (no amortization) — but it is the
  cleaner variant and resolves R5.

**Uncommitted working-tree state (3 files modified, nothing committed since b751abf):**
- `engine.py` — `mtp_decode_step` now has the **TWO-DECODE** variant (with the snapshot
  bug above) + per-component timing. The `batch.moe_decode=True` wiring is now unused
  (two-decode uses 1-token decodes, not a 2-token extend).
- `core.py` — `Batch.moe_decode` flag + `is_moe_prefill` property (used only by the
  two-token-extend variant).
- `layers/moe.py` — `OffloadMoELayer.forward` + `routed_forward` branch on
  `is_moe_prefill` (the moe_decode fix; still correct and harmless).
- Scratch probes: `s2_perf_probe.py` (clean perf), `s2_diag*.py` (cost breakdowns).

**Options for tomorrow (ranked):**
1. **Fix the two-decode snapshot bug** (move `copy_from` before d2) and re-run
   `s2_perf_probe.py` to confirm ~0.91× **and** bit-identical GDN state vs greedy
   (this would close R5 / `gdn_match=False`). Cheapest, highest-value next step.
2. **Confirm the "verify does not amortize" root cause** with one clean V-vs-D probe
   (expect V/D ≈ 2). If confirmed, the negative result is solid.
3. **Quick check: larger `--moe-cache-size`** (512 → 1024/2048, we have ~2 GiB headroom;
   this is an *increase*, not the forbidden decrease). Prediction: it does NOT change
   V/D (2-token compute is still 2× 1-token compute even with resident experts), so it
   won't cross 1.0× — but a 10-minute check rules it out.
4. **Document the negative result** and **do NOT wire S3** (production) on this model.
   MTP at k=1 is the wrong tool for an on-demand-MoE model. If a decode speedup is
   required later, the real options are (a) a fused 2-token MoE kernel that shares expert
   loading (significant kernel investment), or (b) a model/backend where the verify
   amortizes. The MTP code itself is correct and reusable for such models.

**Recommendation:** keep the (correct) MTP code, fix the snapshot bug, confirm the
negative result, and record it. Do not proceed to S3 on this model.

### S3 — Scheduler wiring (size-1 decode batches, `--mtp` gated)
- **Do:** in the scheduler decode path, when `--mtp` and the batch is a single decode
  request, call `engine.mtp_decode_step` instead of `forward_batch`; extend the commit
  path (`_process_last_data` + `Req`) to commit the 1-or-2 returned tokens (R1).
  Ensure KV allocates 2 slots for the verify (R3).
- **Gate:**
  - **Server end-to-end:** launch the real server with `--mtp --kv-q8`, send the 3
    prompts via the client, confirm it streams and completes (no hang, no 503, no OOM).
  - **Token-identity vs greedy:** the MTP server's output for each prompt must match a
    greedy (`--mtp` off) run. **Tolerance (R5):** allow ≤1 divergence per 1000 tokens
    attributable to fp16 near-ties; *zero* divergences expected on the 3 short prompts.
    If divergences exceed tolerance → real bug, stop.
  - **`--mtp` off:** byte-for-byte identical output + 24 tok/s (regression).
  - **64K context still loads** with the head resident (VRAM headroom ≥ ~2 GiB).
- **Test:** `phase5_server_e2e.py`: start server (mtp on/off), send prompts, diff
  outputs, report tok/s for both.
- **Commit:** `S3: scheduler MTP wiring (size-1 decode, --mtp gated)`.

### S4 — Production A/B speed benchmark (the payoff)
- **Do:** run the production speed harness (the real `qwen_q8.sh` workload: C-coding,
  Java-coding, bug hunts, doc-gen) with `--mtp` on and off, same prompts, same 64K
  context.
- **Gate:**
  - **Speedup ≥ 1.4×** (conservative floor; target ~1.6–1.8×). If < 1.4×, the fused
    head's real cost is higher than the 0.8548 alpha projection assumed — report the
    real number, do NOT claim the projection. (This is the strata lesson: measure, don't
    assume.)
  - **No regression** on the non-MTP path (24 tok/s held).
  - **Stability:** 3 runs each, variance < 5%.
- **Test:** existing production benchmark harness, parameterized by `--mtp`.
- **Commit:** `S4: production A/B — MTP decode measured at X tok/s (Y× vs baseline)`.

### S5 — Soak + edge cases
- **Do:** longer generations (512+ tokens), EOS/stop-string hits mid-MTP, aborts,
  prefix-cached requests, the agent-envelope workload.
- **Gate:** no corruption, no leaks (VRAM flat over 100 requests), correct
  finish-reasons, aborts clean up GDN slots.
- **Test:** soak script; `nvidia-smi` sampling for leak detection.
- **Commit:** `S5: soak + edge cases PASS`.

### S6 — (Optional, later) Multi-request MTP batches
- Only after S1–S5 are green. Generalize the scheduler wiring from size-1 to the
  overlap multi-req decode path. **Highest R1 risk; do last, with the strongest
  token-identity + throughput gates.** Out of scope for the initial "open heart" — the
  size-1 path already delivers the speedup for the user's single-stream coding workload.

---

## 6. Test matrix (consolidated)

| Test | Stage | Pass criterion |
|------|-------|----------------|
| Baseline 24 tok/s, 64K, ~2 GiB headroom | S0 | within ±3% of `c0dc5a2` |
| Tap no-op when `--mtp` off | S1 | baseline bit-identical + 24 tok/s |
| Tapped `h` == A2 `pre[-1]` | S1 | max\|Δ\| = 0 |
| `mtp_decode_step` == oracle `mtp_loop` | S2 | token-identical, 3 prompts × 64 |
| Acceptance 0.80–0.90 | S2 | in range |
| GDN reject rollback bit-exact | S2 | state hash == 1-token decode state |
| Server streams, no hang/503/OOM | S3 | completes 3 prompts |
| MTP server output == greedy (≤1/1000 ties) | S3 | within tolerance |
| `--mtp` off regression | S3 | byte-identical + 24 tok/s |
| Speedup ≥ 1.4× (target 1.6–1.8×) | S4 | measured, 3-run variance < 5% |
| Soak: no leak, correct finish/abort | S5 | VRAM flat, correct reasons |

---

## 7. Definition of done

- `--mtp` on: ~1.6–1.8× decode (measured, S4), token-correct (S3), stable (S5).
- `--mtp` off: byte-for-byte identical to `cb1dda2` (regression, S3).
- 64K context + ~2 GiB headroom + `--moe-cache-size` 512 all preserved.
- `perf/sm75-prefill` untouched; backup tag `backup/pre-open-heart-surgery` intact.
- Every stage committed; floor = last passing gate.

## 8. Explicit non-goals (keep the incision small)

- No non-greedy / sampled MTP (strata's coupled sampling) — greedy only, k=1.
- No k=2+ draft windows until k=1 is measured (strata controller lesson).
- No multi-GPU, no CPU expert pool, no iq3_xxs — single sm75 GPU, GPU-resident nvfp4.
- No prefill-path MTP (MTP is a decode optimization only).
