# Post-E1+E3 re-profile + Strata-idea cost/benefit (2026-10-03)

Reference config: `qwen.sh` (E1+E3 cuBLAS, GROUP=16, moe-cache-size 512, radix, 0.95),
context 64000 (per user), loopback :1920. Model Qwen3.6-35B-A3B-NVFP4 (40 layers,
256 experts top-8, 32 GDN linear-attn + 8 full-attn layers, hidden 2048, moe_inter 512).
Hardware: RTX 2080 8 GB (sm75).

## 1. The post-E1+E3 floor is a FIXED per-forward offload cost

Per-op device time (torch profiler, steady state). Note `aten::copy_` and `Memcpy HtoD`
counts/times are ~constant across chunk sizes — they are paid once per forward, not per token:

| capture | n_tok | wall_ms | copy_ (ct / ms) | HtoD (ct / ms) |
|---|---|---|---|---|
| n2048 | 2048 | 5697 | 2385 / 1743 | 240 / 1657 |
| n524  | 524  | 2572 | 2385 / 1700 | 240 / 1661 |
| n12   | 12   | 2374 | 2362 / 1687 | 240 / 1660 |

Fixed per-forward cost ≈ **3.4 s** (2385 copies + 240 pinned HtoD) + ~1.3 s launch/sync/
dequant ≈ **~4.7 s wall per chunk**, independent of how many tokens are in the chunk.
This is the offload machinery (experts live in CPU RAM, streamed over PCIe; the copies are
the staging/scatter around it). It is the single dominant cost of prefill on this card.

n2048 composition (device ms, total 11085, wall 5697, 1.95x overlap):
- GEMMs (dense + bmm + mm)  2078  (18.7%)  <- E1+E3 already at tensor-core speed
- GDN recurrence           3266  (29.5%)
- aten::copy_              1743  (15.7%)  <- fixed
- HtoD                     1657  (14.9%)  <- fixed
- attention                1218  (11.0%)
- dequant nvfp4             342  ( 3.1%)

## 2. Idea 1 (bigger chunks) — MEASURED: 4096 is the sweet spot, 8192 regresses

Clean TTFT sweep, `--max-prefill-length 8192` (each request = one chunk of its size):

| chunk | TTFT_tok/s (clean) | vs 2048 |
|---|---|---|
| 1024 | 429 | -15% |
| 2048 | 502 | baseline |
| 4096 | **510** | +2% |
| 8192 | 408 | -19% (REGRESSION) |

Why 8192 regresses — n8192 op table (device ms, total 32849):
- ChunkGatedDeltaRuleFunction (GDN)  6466  (19.7%)
- _extend_attention_split_kernel     6373  (19.4%)
- chunk_fwd_kernel_o                 4431  (13.5%)
- GEMMs (bmm + turing_fp16 + mm)     6268  (19.1%)
- copies + HtoD (fixed)              3582  (10.9%)

GDN recurrence + full-attention grow super-linearly with chunk length (GDN is a sequential
recurrence over the chunk; attention is O(n^2) over it). At 8192 they total ~17.3 s device
(52.6%) and overwhelm the ~4.7 s fixed offload cost that bigger chunks were supposed to
amortize. Net: bigger chunks help until the recurrence/attention term catches up, which on
this card is between 4096 and 8192.

**Cost of idea 1:** `--max-prefill-length 4096` uses ~2.1x the activation VRAM of 2048.
Server still fit with 2.12 GiB free after CUDA graphs at 8192, so 4096 is safe on 8 GB.
**Verdict: adopt 4096. Easy (one flag), +2% at 4096 and a large win for short prompts
(a 2048 prompt currently takes 2 chunks = 2x the 4.7 s fixed cost; at 4096 it is 1 chunk).**
Do NOT go to 8192.

## 3. Idea 3 (the aten::copy_ floor) — REAL, ~1.7 s/forward, but NOT "easy" yet

2385 copies/forward ~= 59.6 copies/layer. ~1.7 s of the ~4.7 s fixed cost. These are the
offload staging/scatter (experts stream CPU->GPU; activations/expert outputs staged). To cut
them we need to know exactly which tensors: the likely wins are (a) in-place scatter instead
of copy+scatter, (b) fewer, larger HtoD transfers (batch the 240 copies), (c) keep hot experts
resident instead of re-streaming every forward. All three are real kernel/offload-engine work,
not a flag. **Verdict: highest remaining upside (~1.7 s/forward = ~30% of the floor) but the
only idea that requires new code. Next step if pursued: profile ONE layer's offload path to
identify the 59.6 copies/layer and confirm which are staging vs scatter.**

## 4. Idea 4 (deeper H2D overlap / in-kernel GDN) — partially already done

HtoD (1657 ms) already overlaps: wall 5697 vs device 11085 = 1.95x overlap, so the offload
stream is running under the compute. The non-overlapped remainder is the GDN recurrence
(sequential — cannot be overlapped across its own steps) + launch/sync. Strata's chunk-walking
GDN and multi-stage expert streaming would attack the GDN 3266 ms (n2048) / 6466 ms (n8192),
which is now the single largest per-token term. **Verdict: this is where the last real
per-token win is (GDN is 29% of n2048 device time), but it is the hardest — it means rewriting
the linear-attention prefill kernel. Not "easy".**

## Bottom line
- E1+E3 already captured the big win (GEMMs now 18.7% of device time, tensor-core speed).
- **Idea 1: adopt `--max-prefill-length 4096` now** — easy, safe on 8 GB, and it halves the
  fixed-cost penalty for prompts <= 4096. Cap at 4096, not 8192.
- **Idea 3: the copies (~1.7 s/forward) are the biggest single remaining target** but need a
  targeted offload-path profile + kernel work — not a quick win.
- **Idea 4: GDN recurrence is the biggest per-token term** (29% at 2048, 52% at 8192) — the
  hardest, and the reason 8192 regresses.
- Strata's actual fast MoE kernels are Ampere-only (sm_80+) and do not run on this sm75 card;
  the portable ideas are the orchestration ones above.

## Artifacts
- profile-post/  : n2048/n524/n12 op tables + driver-requests.json (first capture run)
- sweep/         : sweep-8192.json (clean chunk-size TTFT)
- profile-8192/  : n8192 op table + driver.json (8192 regression breakdown)
- scripts/sm75/serve-profile-post.sh (MAX_PREFILL/PORT/FREETOKEN_PROFILE* env-overridable)
- scripts/sm75/profile_driver_post.py (--sweep mode)
- scripts/sm75/profile_driver_8192.py
