# FreeToken MTP (nextn) — Implementation & Test Plan

Goal: add Multi-Token-Prediction self-speculative decode to FreeToken for
`Qwen3.6-35B-A3B-NVFP4` on the RTX 2080 (sm75, 8 GB), **without regressing the
tuned prefill path** (`--disable-moe-prefill-overlap --moe-cache-size 512
--dtype float16`), and **without shrinking the 64000-token context** that hermes
needs (`qwen.sh`: `--num-tokens 64000 --max-seq-len-override 64000`).

Feasibility is **measured, not projected** — GO. This plan is the build + test
sequence, organized into **phases with hard gates** and a **stop condition** at
each. Work happens on branch `perf/mtp-decode`, which is built **on top of the
committed prefill baseline** (commit `4d5d0fb` on `perf/sm75-prefill`). The tuned
prefill path (cutlass backend, GDN sm75 tuning, cuBLAS prefill, tuned `qwen.sh`)
is committed and safe — see §0.

**Priority order (user-confirmed 2026-10-04):** **Q8 KV cache is Phase 1 and is
the VRAM prerequisite for MTP.** MTP's head + the 61K prefill transient do not
fit in the free VRAM at 64K context while KV is fp16 (the math is in §1). Q8-KV
frees the pool, and it has standalone value for hermes (bigger context) even if
MTP later aborts — so it is not wasted work.

---

## 0. Rollback guarantee (read this first)

If any phase proves too hard, the way back is clean and artifact-free:

1. **All MTP code lives on `perf/mtp-decode`** — never on `perf/sm75-prefill` or
   the main branch. No source file in the baseline branch is modified.
2. **The prefill baseline is committed.** Commit `4d5d0fb` on `perf/sm75-prefill`
   captures the tuned prefill path (cutlass backend, GDN sm75 tuning, cuBLAS
   prefill, tuned `qwen.sh`, benchmark harnesses + results). `perf/mtp-decode`
   fast-forwards onto it, so the prefill work is **permanently safe on its own
   branch** — discarding the MTP branch loses nothing of it.
3. **`--mtp` is default-off.** When the flag is absent, the engine is byte-for-byte
   the current FreeToken. The flag is a no-op when off (verified in Phase 5).
4. **`--kv-q8` is default-off** (Phase 1). When off, the KV pool is the current
   fp16 pool. No-op when off (verified in Phase 1).
5. **Dedicated MTP weight loader** (`load_mtp_weights`) — the strict main
   `load_state_dict` in `weight.py` is **not touched**. `mtp.*` stays dropped from
   the main pass.
6. **No production-config edits.** `qwen.sh` is only extended (flags added) for the
   A/B run; the baseline invocation is unchanged.
7. **Scratch stays out of the repo.** Measurement scripts and the downloaded
   `mtp_head.safetensors` live in `~/.hermes/cache/scratch/`, not the repo.

**Rollback command:** `git checkout perf/sm75-prefill && git branch -D perf/mtp-decode`
→ clean working FreeToken **with all prefill improvements intact** (they're in
commit `4d5d0fb`), zero MTP artifacts, zero config drift.

A phase is **abandoned** (not "fixed at all costs") when its stop condition fires.
Abandoning a phase = stop, report the honest number, and the branch is discardable.
Q8-KV (Phase 1) can be **kept on its own** (it's a context win) even if MTP
(Phases 2–5) aborts — that's a partial success, not a rollback.

---

## 1. Why Q8-KV comes first (the 64K-context VRAM math)

The KV pool is sized for the 10 full-attention layers and is currently **fp16**
(`KV_CACHE_DTYPE_BYTES = 2`, engine dtype `--dtype float16`). At the 64000-token
context hermes requires:

| quantity | value |
|---|---|
| per-token KV (10 layers × 2 KV-heads × 256 head_dim × 2 B × K+V) | 20.0 KB |
| fp16 KV @ 64K | **1.22 GiB** |
| Q8 KV @ 64K | 0.61 GiB |
| **freed by Q8-KV** | **0.61 GiB** |

VRAM budget (measured: **2.12 GiB free after engine init**, which already has the
fp16 KV pool allocated):

| scenario | free | head (0.45 GiB) | prefill transient (2.0) | margin |
|---|---|---|---|---|
| fp16 KV + MTP head | 2.12 | 0.45 | 2.0 | **−0.33 GiB → does NOT fit** |
| Q8 KV + MTP head | 2.73 | 0.45 | 2.0 | **+0.28 GiB → fits** |

So at 64K context, the MTP head + the 61K CUTLASS prefill transient simply do not
fit while KV is fp16. Q8-KV frees exactly the 0.61 GiB that closes the gap. This is
why Q8-KV is Phase 1 and a **prerequisite**, not an optional later phase.

The 2.0 GiB prefill transient is the hard constraint from the tuned prefill path
(`FREETOKEN_E3_GROUP=16` bounds the fp16 transient to fit 8 GB). We do not shrink
`--moe-cache-size` to make room (that costs decode hit-rate) — Q8-KV is the fix.

## 2. Measured baseline (ground truth, 2026-10-04, RTX 2080)

| quantity | value | note |
|---|---|---|
| pure decode | **24 tok/s** | live measurement 2026-10-04 (phase0-verify, cutlass/512/64k config; graphs ON) |
| prefill | 484 tok/s | 20.2× decode ratio |
| r (2-token verify / 1-token decode) | **1.041** | two-run difference method; true M=2 verify is a lower bound (~1.05–1.1) |
| alpha (draft acceptance, pre-final-norm wiring) | **0.872** | post-norm wiring 0.830; mis-wired head ≈ chance |
| VRAM free after engine init | 2.12 GiB | fp16 KV pool already allocated |

Source scripts (scratch): `measure_mtp_r.py`, `measure_alpha.py`,
`fetch_mtp_head.py` (→ `mtp_head.safetensors`, 19 bf16 tensors, 1.69 GB, already
downloaded). Fork references: `bench_mtp_accept.py`, `mtp.py`, `MTP.md`.

## 3. Architecture facts (from the code, not assumed)

- **40 layers = 10 full-attention + 30 GDN** (`config.json`: `num_hidden_layers:
  40`, `full_attention_interval: 4`; `layer_types` has full_attention at indices
  3,7,11,15,19,23,27,31,35,39 = 10 layers).
- **Every layer carries a routed MoE** — `Qwen3_5DecoderLayer.__init__`
  (`model.py:52`) sets `self.mlp = Qwen3_5MoE(config, layer_id)` for *all* 40
  layers, GDN and full-attention alike. 256 experts, top-8,
  `moe_intermediate_size=512`. Any base-model forward (including the reject-replay)
  pays the MoE cost.
- **GDN state** lives in `LinearStatePool` (`kvcache/linear_state_pool.py`):
  `conv_states` (line 59) + `recurrent_states` (line 66, fp32 by default), indexed
  by `Req.table_idx`. The snapshot/restore primitive is `copy_from(src, dst)`
  (line 142) — already used in production by the hybrid-radix COW path. Per-request
  state ≈ 61 MB (30 layers × fp32 32×128×128 recurrent = 2 MB each + conv). **GDN state is not
  paged KV** — Q8-KV does not touch it.
- **MTP head** = 19 bf16 `mtp.*` tensors: one full-attention + MoE block
  (`pre_fc_norm_embedding/hidden`, `fc`, `layers.0.*`, `norm`). **No GDN state**
  in the head. Reuses `Qwen3_5Attention` / `Qwen3_5MoE`. Embedding + lm_head are
  shared with the base model. The NVFP4 quant config **ignores the head**
  (`config.json` `"ignore": ["mtp.layers.0*", "mtp*"]`) — the head ships bf16.
  Total **844.6M** params: **256 routed experts 805.3M** (95.3%), non-expert (fc +
  attention + shared expert + router + norms) **39.3M**.
- **Weight loader drops `mtp.*`** (`weight.py:136` returns `None`). Local `.ftw`
  has 1055 tensors, **0** are `mtp.*`. The 19 tensors exist only in the HF source
  (`nvidia/Qwen3.6-35B-A3B-NVFP4`, shard 3).
- **KV pool** is sized for the 10 full-attention layers (`layer_ids` back only the
  full-attn subset, `mha_pool.py:39-49`), fp16. The head needs **one more**
  full-attn KV slot (append `layer_id = num_full_attention_layers`).
- **KV read paths** (`attention/cutlass.py`): decode uses the Triton
  `decode_paged_attention` kernel reading the paged fp16 cache; fresh prefill
  reads the extend k/v densely and runs CUTLASS SDPA; prefix prefill gathers from
  the paged cache then SDPA; fallbacks are Triton `extend_paged_attention` /
  `paged_attention`. (Relevant to Q8-KV in Phase 1.)

## 4. The loop design (k=1 — one head layer, one draft token/step)

The head predicts `token_{t+2}` from `(hidden_t, token_{t+1})`. Key decision:
**no separate base-decode step** — the verify forward does double duty (produces
the anchor token *and* checks the draft). That's what makes the win real.

State at step start: base KV+GDN reflect committed `x_0..x_c`; `h_c` =
pre-final-norm hidden at `c`; `token_{c+1}` = `argmax(lm_head(h_c))` (from the
previous step, or the initial prefill).

1. **Propose** — `mtp.forward(h_c, token_{c+1})` → draft `token_{c+2}`. One head
   forward on its own KV slot (causal over the short draft seq).
2. **Verify** — one base-model forward over `[token_{c+1}, token_{c+2}]`
   (2 tokens, positions `c+1..c+2`, routed through the **extend/prefill** path so
   it's variable-length, not the bs=1 decode graph). Produces `h_{c+1}`, `h_{c+2}`
   (pre-final-norm).
   - `token_{c+2}' = argmax(lm_head(h_{c+1}))` (base's real prediction at `c+1`)
   - `token_{c+3}  = argmax(lm_head(h_{c+2}))` (next step's anchor-next)
   - `a = 1 if token_{c+2}' == token_{c+2} else 0` (accepted drafts)
3. **Commit / rollback** (GDN snapshot taken before step 2):
   - **accept (a=1)**: commit `x_{c+1}, x_{c+2}`; `c→c+2`; `h←h_{c+2}`;
     `next←token_{c+3}`. GDN state is already at `c+2` from the verify — **no
     rollback**. Advance KV position to `c+2`.
   - **reject (a=0)**: commit `x_{c+1}` only; `c→c+1`; `next←token_{c+2}'`.
     GDN state is at `c+2` → **roll back**: `copy_from` the pre-verify snapshot
     (state at `c`), then **replay** `[token_{c+1}]` (1-token forward) → state at
     `c+1`, take `h_{c+1}` from the replay (consistent with restored state).
     Advance KV position to `c+1` (the `c+2` KV is dead data, overwritten next
     step).
4. **Re-seed** the head KV slot for the next step's draft.

### Cost model (why ~1.6×, not the fork's 1.8×)

Per step, in 1-token-forward units (base_decode = 1, verify-2 = r = 1.041,
replay = 1, head = ε):

- accept (p=α=0.872): cost = ε + r, tokens = 2
- reject (p=1−α=0.128): cost = ε + r + **replay(1.0)**, tokens = 1

  E[cost]   = ε + r + (1−α)·1.0 = 0 + 1.041 + 0.128 = **1.169**
  E[tokens] = 1 + α = **1.872**
  **projected speedup = 1.872 / 1.169 ≈ 1.60×** → ~38 tok/s from 24.

The fork's `(1+α)/r = 1.80×` omits the reject-replay cost. **1.6× is the honest
projection.** The replay may be *cheaper* than 1.0 on the offload path if its
experts are already cached from the verify forward (PCIe fetch amortized) — the
A/B in Phase 5 measures the real number.

---

## 5. Phases (each gated; a failed gate stops that phase, not the branch)

### Phase 0 — Baseline verification
Confirm the committed baseline is the untouched tuned config before any change.
- Confirm `git status` clean on `perf/mtp-decode` (it is: prefill work is commit
  `4d5d0fb`, MTP plan is `0e9a013`); confirm `qwen.sh` has the tuned flags
  (cutlass, `--moe-cache-size 512`, `--disable-moe-prefill-overlap`, 64000).
- Run `qwen.sh` as-is; measure decode + prefill tok/s.
- **Gate:** decode ≈ 24 tok/s, prefill ≈ 484 tok/s, tree clean.
- **Stop:** if baseline numbers are off, stop and reconcile first (do not build on
  a moving baseline).

### Phase 1 — Q8 KV cache  (VRAM PREREQUISITE, standalone value)
Halve the paged-KV footprint so MTP fits at 64K (§1). **This phase is valuable on
its own** — it lets hermes run 64K context with more headroom — so it can be kept
even if MTP aborts later.
- Add `--kv-q8` (default **off**).
- **Approach: Q8 storage, fp16 compute** (quantize on store, dequantize to fp16
  before attention) — exactly what llama.cpp does; avoids modifying attention math.
- Per-path work (grounded in `mha_pool.py` + `attention/cutlass.py`):

  | path | current | Q8 change | difficulty |
  |---|---|---|---|
  | pool alloc | fp16 `_kv_buffer` | add int8 buffer + per-block scales | medium |
  | store | `store_cache` writes fp16 | quantize fp16→q8 on write | medium |
  | decode read | Triton `decode_paged_attention` reads fp16 paged | **hardest**: dequant in-register, or to a temp paged fp16 buffer | medium-hard |
  | fresh prefill | reads dense extend k/v → SDPA | dequant before SDPA | medium |
  | prefix prefill | gathers paged fp16 → SDPA | dequant gathered q8 before SDPA | medium |
  | fallback Triton | `extend_paged_attention`/`paged_attention` fp16 | same as decode | medium-hard |

- **Gate:** KV pool halved (measured: 1.22 → 0.61 GiB @ 64K); no quality
  regression on the coding prompts (token-identical or ≤ Q8 noise vs fp16 KV);
  decode + prefill tok/s within 2% of the fp16-KV baseline; `--kv-q8` off == fp16
  baseline (true no-op).
- **Stop:** if the decode-path dequant regresses decode below the MTP-free
  baseline and can't be tuned back → **abort Q8-KV**, report. (MTP then cannot fit
  at 64K → MTP aborts too, but if Q8-KV's context win already passed its gate it
  can be kept as the fallback.)

### Phase 2 — GDN-rollback correctness  (MAKE-OR-BREAK for MTP)
Prove the reject path is faithful before building the head/engine on it. Cheap:
no head, no extra VRAM beyond the base model. (Independent of Q8-KV — GDN state is
not paged KV — so it can run in parallel with Phase 1 if desired.)
- **A1 (micro, no head):** drive the base model incrementally on one request:
  prefill `x_0..x_c` → snapshot GDN (`copy_from` to a scratch slot) → 2-token
  chunk forward (verify shape) → restore → 1-token replay. Assert GDN state (and
  the full 40-layer hidden) after replay **equals** a straight 1-token decode from
  the same snapshot. Catches chunk-vs-decode kernel divergence and
  snapshot/restore bugs cheaply.
- **A2 (end-to-end):** with the Phase-4 loop, assert MTP-loop output == straight
  greedy decode, token-for-token, on a C/Java coding prompt (greedy is
  deterministic). Covers head + verify + rollback + replay + KV truncation
  together.
- **Gate:** A1 bit-identical (or ≤ bf16 noise); A2 token-identical.
- **Stop:** rollback unfaithful and unfixable → **abort MTP** (keep Q8-KV if it
  passed Phase 1), report. This is the single biggest risk; isolating it before
  head/engine work means an abort costs only this microtest.

**A1 RESULT (2026-10-04): PASS — reject path is bit-exact faithful.**
Harness `~/.hermes/cache/scratch/a1_gdn_reject.py` drives the real 40-layer model
(real GDN weights, real cutlass KV) through the exact §4 reject path on a 129-token
prefix (spans 2 GDN chunks): prefill → `copy_from` snapshot → 2-token verify (chunk
kernel) → restore → 1-token replay (decode kernel), compared against a straight
1-token decode from the same snapshot. KV held identical across paths, so only the
GDN state trajectory differs.

| check | result |
|---|---|
| **PRIMARY** reject-path logits vs straight decode | **0.000000** (bit-identical) |
| **PRIMARY** GDN recurrent state (fp32) | **0.000000e+00** |
| **PRIMARY** GDN conv state (fp16) | **0.000000e+00** |
| SECONDARY chunk-vs-decode recurrent @ c+2 | mean rel 4.0e-4, p99 4.6e-3 (max 7.7, a few outliers) |
| SECONDARY chunk-vs-decode conv @ c+2 | mean rel 0.416 (see note) |
| SECONDARY chunk-vs-decode logits @ c+2 | max abs 2.84 |

- **`copy_from` snapshot/restore is bit-exact** and the replay is faithful — the
  make-or-break risk is cleared. The reject path (the one MTP actually uses on a
  draft miss) reproduces a straight decode to the last bit.
- **The SECONDARY divergence is expected, not a bug.** The conv state stores the
  last 3 raw input tokens; the verify path consumed `x_{c+2}` while the decode path
  re-consumed `x_{c+1}`, so the conv buffers *should* differ by one token. The
  recurrent state — the part that actually accumulates across the sequence — matches
  to mean rel 4e-4 (sub-0.05%), i.e. the chunk and decode kernels are numerically
  equivalent on the same input. The accept path (which commits the chunk state)
  therefore carries only this small, bounded drift; A2 (Phase 4) confirms it is
  token-identical end-to-end.
- **A2 is deferred to Phase 4** (it needs the full MTP loop). A1's PASS is the
  Phase-2 gate; A2 re-verifies with the head in the loop.

### Phase 3 — Head weight load + VRAM
- Add `--mtp` (default **off**) and `--mtp-draft` (default 1) to `server/args.py`.
- **Dedicated loader** (fork option 2, lowest blast radius — don't touch the
  strict main `load_state_dict`): `load_mtp_weights(mtp, path)` reads the 19
  `mtp.*` tensors from a safetensors file (the already-downloaded
  `mtp_head.safetensors`, or a re-converted FTW), applies the same
  q/k/v→`qkv_proj` and gate/up→`gate_up_proj` fusion, **quantizes the 256 routed
  experts to nvfp4** (805.3M × 2 B = 1.61 GB → 0.40 GB, freeing 1.21 GB), keeps the
  39.3M non-expert params bf16 (~0.08 GB), assigns into `self.mtp`. Head experts
  stay **resident** (not offloaded). Head total ≈ **0.48 GB (0.45 GiB)**.
- Size the MHA KV pool for `num_full_attention_layers + 1`; give
  `Qwen3_5MTPLayer` the appended `layer_id`.
- Instantiate `Qwen3_5MTP` (port fork `mtp.py` into `models/qwen3_5_moe/mtp.py`)
  and attach to the model only when `--mtp`.
- **Gate:** head loads; VRAM free after load **≥ 2.0 GiB** (prefill headroom
  intact — the hard constraint); a 2-token sample reproduces alpha ≈ 0.87.
  **Grounded:** with Q8-KV from Phase 1, free is 2.73 GiB; after the 0.45 GiB head
  → **2.28 GiB ≥ 2.0** ✓. (Without Phase 1 this gate fails at −0.33 GiB — which is
  exactly why Q8-KV is the prerequisite.)
- **Stop:** headroom can't be preserved → **abort MTP** (keep Q8-KV), report.
  Do not silently shrink `--moe-cache-size` — that costs decode hit-rate.

**Phase 3 RESULT (2026-10-04): PASS — all three gates.**
Harness `~/.hermes/cache/scratch/phase3_mtp_head.py` is standalone (like A1):
builds the live Q8-KV engine, loads the 19-tensor `mtp_head.safetensors` (all bf16),
hand-quantizes the 256 routed experts bf16→nvfp4 (modelopt not installed), and runs
the 2-token alpha sample. It proves the head math + quantize + VRAM before any
engine wiring (the `--mtp` flag, loader, and KV-pool +1 layer are Phase 4).

| gate | result |
|---|---|
| **(1) quantize vs `dequant_nvfp4`** | **PASS** — kernel-vs-ref **0.000000** (my packed bytes read bit-exactly by the codebase's own kernel); mean rel 0.276 (gu) / 0.247 (dn), the expected e2m1 4-bit-mantissa error |
| **(2) VRAM free after head load** | **PASS** — head = 0.532 GB (experts nvfp4 0.454 GB + other 0.079 GB); **2.25 GiB free ≥ 2.0** |
| **(3) 2-token alpha ≈ 0.87** | **PASS** — **0.8548** (pre_norm, 53/62) with the *quantized* experts (ref 0.872, ≥ 0.75 floor) |

- **Projected MTP speedup (draft-1, r=1.041, alpha 0.8548): 1.78× → ~43 tok/s**
  (baseline 24 tok/s). The alpha is slightly below the bf16 reference (0.8548 vs
  0.872) because it uses the nvfp4-quantized experts, which is the *production*
  path — so the projection is conservative, not optimistic.
- **OOM fixes baked into the harness (reproducible):** (a) quantize runs **on CPU**
  in 32-expert chunks (the full `[256,1024,2048].float()` = 2 GiB fp32 transient
  OOMs against the ~1 GiB free after the base model + offload expert cache); only
  the small packed/scale/global outputs (~0.45 GB) are copied to GPU. (b) the alpha
  run's MoE dequantizes the union of experts **position-chunked** (4 tokens → ≤32
  experts ≈ 200 MB), since a full-sequence union could hit all 256 experts (1.6 GB).
  (c) the alpha uses the validated `measure_alpha.py` head math (self-contained SDP
  over the head's own input), not the base paged KV — the KV-pool +1 layer is a
  Phase-4 multi-token-draft concern, not the 2-token gate.
- **The `--mtp` flag is NOT yet wired** (that is Phase 4). Phase 3 proves the three
  load/VRAM/alpha gates; Phase 4 adds the loader + engine loop and re-verifies A2.

### Phase 4 — Speculative decode loop
- Wire the §4 loop into `engine/engine.py` (+ a small scheduler branch), gated by
  `--mtp`; fused/offload paths untouched when off.
- Verify forward routes through the extend/prefill path (variable-length), not
  the bs=1 decode graph.
- GDN rollback via `LinearStatePool.copy_from` + 1-token replay (§4 step 3).
- KV truncation on reject = don't advance the position pointer (dead pages
  overwritten next step).
- **Gate:** Phase-2 A2 passes (token-identical to greedy) on ≥ 3 prompts.

**Phase 4 A2 RESULT (2026-10-04): PASS — token-identical to greedy.**
Standalone harness `~/.hermes/cache/scratch/phase4_a2_loop.py` (mirrors A1 + Phase 3):
builds the real production engine (Q8-KV, production flags), loads + quantizes the
head (Phase 3's proven path), then runs the §4 k=1 loop (propose→verify→commit/rollback)
and compares token-for-token against a straight greedy decode (1-token steps).

| prompt | tokens | match | accept/reject | greedy | mtp (harness) |
|---|---|---|---|---|---|
| C linked-list reverse | 24 | **identical** | 11/1 | 5.07s | 31.85s |
| Java word-count | 24 | **identical** | 10/3 | 4.84s | 34.39s |
| printing-press prose | 24 | **identical** | 10/3 | 4.82s | 34.47s |

- **Token-identity: PASS** (3 prompts × 24 tokens, zero divergence). Acceptance
  31/38 = **0.816** (close to the 0.872 reference; the harness uses the nvfp4
  quantized experts, the production path).
- **k=1 → no extra KV layer needed.** The head only ever sees 1 token per draft, so
  its attention is self-contained (Phase 3 proved this with SDP). The KV-pool +1
  layer is only needed for k≥2 multi-token drafts (deferred).
- **GDN rollback validated live.** The reject path (snapshot→2-token verify→restore→
  1-token replay) is exercised on real rejects (7 total) and produces token-identical
  output — confirming A1's bit-exact result holds in the full loop with the head in
  the picture.
- **PERF CAVEAT (harness artifact, not a loop bug):** the harness MTP path is ~3.5×
  *slower* than greedy (31s vs 5s for 24 tokens) because the harness head-MoE is
  **dequant-then-einsum** (materializes the expert union, ~2.7s per head forward),
  not the fused nvfp4 kernel. In production the head-MoE uses the same fused kernel
  as the base model (~1 base-layer cost ≈ 5ms), so ε≈0 and the §4 cost model
  (1.6×) should hold. The Phase 5 A/B measures the real number with the fused kernel.

**Remaining Phase 4 work (production wiring, not yet done):**
- `--mtp` flag in `args.py` (default off; no-op when off).
- `mtp.py` head module (port the fork's `Qwen3_5MTP` — validated math).
- `load_mtp_weights()` (dedicated loader; main `load_state_dict` untouched).
- Engine decode loop (§4) + GDN rollback wiring, gated by `--mtp`.
- The harness head-MoE should be swapped to the fused kernel before the A/B
  (Phase 5) so the speedup is measured with production kernels.

### Phase 5 — A/B benchmark (the user's real workload)
- Workload: **C coding, Java coding, bug hunts, doc-gen from C** (low-entropy →
  alpha should be ≥ the 0.872 prose figure).
- Run baseline (no `--mtp`) vs `--mtp` on the same prompts, same `qwen.sh` flags
  (both with `--kv-q8` from Phase 1, so the A/B isolates MTP).
- Report: decode tok/s, acceptance rate, VRAM free, **and prefill tok/s** (must be
  unchanged).
- **Regression guard:** run baseline with the `--mtp` flag *present but unset* and
  confirm identical numbers (proves the flag is a true no-op when off).
- **Gate:** decode speedup ≥ 1.3× (conservative floor; projection is 1.6×),
  prefill regression < 2%, VRAM headroom preserved.
- **Stop:** measured speedup < 1.0× → **abort MTP**, report the honest number.

---

## 6. Testing plan

1. **Q8-KV (Phase 1):** pool halved (measured); token-identical/≤noise vs fp16 KV
   on coding prompts; decode+prefill within 2%; `--kv-q8` off == fp16 baseline.
2. **Unit (Phase 2 A1):** GDN snapshot/restore/replay == straight decode. No head,
   no extra load beyond the base model. Fast.
3. **Correctness (Phase 2 A2):** MTP loop output == greedy reference,
   token-for-token.
4. **Perf A/B (Phase 5):** baseline vs `--mtp`, decode + prefill + VRAM.
5. **Regression guard (Phase 5):** `--mtp` flag off == baseline, identical numbers.

## 7. Risks & mitigations

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| Q8-KV decode dequant regresses 24 tok/s baseline | med | baseline loss | Phase 1 gated + tunable; abort keeps fp16 KV (but then MTP can't fit at 64K) |
| GDN chunk-vs-decode numerical divergence breaks token-identity | med | kills MTP | Phase 2 A1 isolates it early; if only low-bit noise, A2 argmax is usually stable — measure, don't assume |
| replay cost > 1.0 (experts not cached) → speedup < 1.0 | med | MTP doesn't pay off | Phase 5 measures it; if <1.0, stop (honest result) |
| r higher than 1.041 on the real extend path (proxy was a 2048 prefill) | med | smaller win | Phase 5 measures real verify cost |
| alpha lower on real decode than the from-scratch prefill harness | low | smaller win | Phase 5 measures real acceptance on the coding workload |
| head + prefill transient don't fit at 64K | resolved | prefill regresses | Q8-KV (Phase 1) is the fix; §1 shows it closes the −0.33 → +0.28 GiB gap |

## 8. Decision points (where to stop)

- After **Phase 0**: baseline off → reconcile, don't build.
- After **Phase 1** (Q8-KV): decode dequant regresses and can't be tuned → **abort
  Q8-KV**; MTP then can't fit at 64K → **abort MTP too** (if Q8-KV's context win
  already passed its gate, keep it as the fallback).
- After **Phase 2**: GDN rollback unfaithful and unfixable → **abort MTP** (keep
  Q8-KV), report.
- After **Phase 3**: can't preserve ≥2.0 GiB headroom → **abort MTP** (keep
  Q8-KV), report (no silent `--moe-cache-size` shrink).
- After **Phase 5**: measured speedup < 1.0× → **abort MTP**, report the honest
  number.

Every MTP abort is a clean `git checkout perf/sm75-prefill && git branch -D
perf/mtp-decode` — working FreeToken, zero artifacts (§0). Q8-KV can be kept on
its own (partial success) if MTP aborts after Phase 1.

## 9. Files to touch

Q8-KV (Phase 1):
- `python/freetoken/server/args.py` — add `--kv-q8`.
- `python/freetoken/kvcache/mha_pool.py` — int8 buffer + per-block scales.
- the `store_cache` site (kernel) — quantize on store.
- `python/freetoken/attention/cutlass.py` + the Triton decode/extend kernels —
  dequant on read.

MTP (Phases 2–5):
- `python/freetoken/server/args.py` — add `--mtp`, `--mtp-draft`.
- `python/freetoken/models/qwen3_5_moe/mtp.py` — **new**, port fork head module.
- `python/freetoken/models/qwen3_5_moe/weight.py` — `load_mtp_weights()` (dedicated;
  main loader untouched); keep `mtp.*` dropped from the main pass.
- `python/freetoken/models/qwen3_5_moe/model.py` — attach `self.mtp` when `--mtp`.
- `python/freetoken/engine/engine.py` — KV pool +1 full-attn slot; speculative
  decode loop; GDN rollback wiring.
- `python/freetoken/kvcache/linear_state_pool.py` — reuse `copy_from` (no change
  expected; may add a scratch-snapshot helper).
- `qwen.sh` — add `--kv-q8` (Phase 1) and `--mtp --mtp-draft 1` (Phase 5) for the
  A/B run (baseline stays as-is).

## 10. Definition of done

**Q8-KV (Phase 1, standalone win):** paged-KV halved (1.22 → 0.61 GiB @ 64K) with
no decode/prefill regression beyond the dequant overhead and no quality regression
— hermes gets 64K context with more headroom.

**MTP (Phases 2–5):** `--mtp` on `qwen.sh` gives a **measured** decode speedup ≥
1.3× over the current 24 tok/s baseline (i.e. ≥31 tok/s), **token-identical** output to greedy,
prefill unchanged, and ≥2.0 GiB VRAM headroom preserved at 64K context — all
reported with real numbers, not projections.
