# FreeToken SM75 native prefill — STATUS

Project: `~/src/freetoken-sm75-native-prefill` (independent clone of `~/src/freetoken-sm75`)
Original (read-only reference): `~/src/freetoken-sm75` @ `35668da` (PR #24 "fix(cuda): support Turing GPUs")
Branch: `baseline/local-sm75` (at `35668da`; `perf/sm75-prefill` to be created after baseline tag)
GPU: RTX 2080 8 GB, SM75, driver 595.91.07
Model: `~/Models/Qwen3.6-35B-A3B-NVFP4` (read-only; 3 FTW shards, 20.89 GB)

## Completed gates

- [x] **A-1 Recon** — env, venv origins, 90-pkg freeze, disk (116 GB free), symlinks (none in tree), nested 5.5 GB `.venv` artifact identified.
- [x] **A-2 Independent clone** — `git clone --no-local --no-hardlinks --no-checkout` at `35668da`; `origin` removed; object store verified independent (pack file, link count 1). Branch `baseline/local-sm75` created.
- [x] **A-3 Structure** — `artifacts/sm75-prefill/{test-results,traces}`, `reference/`, `scripts/sm75/`, `third_party/flashqla-sm75/`, `.cache/{baseline,candidate}` created. `.gitignore` extended. Repo-local git identity set; global config untouched.
- [x] **A-4 Local content reproduced** — 3 untracked files copied with SHA-256 provenance. `profile.py` **relocated to `reference/profile.py`** (byte-identical) because at the project root it shadowed the stdlib `profile` module (transformers→torch._dynamo→cProfile→`import profile` → NameError, breaking every `python -c`/`-m`/interactive run from the root). See `source-provenance.json`.
- [x] **A-5 Fresh venv** — `.venv` (Python 3.12.3); pip/setuptools/wheel upgraded.
- [x] **A-6 Dep install** — torch 2.11.0+cu130 + 88 pinned deps installed (background, exit 0).
- [x] **A-7 Editable build** — `freetoken-0.1.2` editable; `_cpu_moe.cpython-312` + `_pinned_tensor.cpython-312` compiled and import OK. `ft` CLI v0.1.2 present.
- [x] **A-8 Verify** — `freetoken.__file__` → new project; torch 2.11.0+cu130, CUDA available, RTX 2080 cap (7,5), `sm_75` in arch list; `pip check` clean.
- [x] **A-9 Baseline launch wrapper** — `serve-baseline.sh` (loopback:1920, project-isolated caches, baseline flags from `qwen.sh`). Server **up and generating**.
- [x] **A-11 Benchmark harness** — `bench.py` (SSE, TTFT-to-first-token, `enable_thinking=False`, `stream_options.include_usage`, incremental JSONL writes, corpus with token IDs + hashes + sampling + seed).
- [x] **A-10 Baseline timings** — full 7-workload run (5 reps + 1 warmup each) complete; medians below.
- [x] **A-12 Finalize** — medians computed; buggy prefix-cached rows preserved as provenance; corpus rebuilt; baseline committed + tagged `baseline`.

## Baseline results (median of 5 timed reps, warmup excluded)

Server: loopback:1920, flags `--max-prefill-length 2048 --moe-backend offload --memory-ratio 0.95 --moe-cache-size 1500 --cache-type radix`, `enable_thinking=False`, `temperature 0.0 top_p 1.0 seed 1234`.

| workload | median TTFT | decode tok/s | in-tok | cached-tok |
|---|---|---|---|---|
| prefill-128 | 4033 ms | – | 140 | – |
| prefill-512 | 10906 ms | – | 524 | – |
| prefill-2048 | 38979 ms | – | 2060 | – |
| mixed-512 | 10941 ms | 32.4 | 524 | – |
| mixed-2048 | 38941 ms | 32.3 | 2060 | – |
| prefix-cached | **1743 ms** | 31.2 | 2012 | **1984** |
| agent-envelope | 1766 ms | 33.4 | 157 | – |

Uncached prefill scales ~linearly with length (128→512→2048 = 4.0→10.9→39.0 s); decode is ~32 tok/s on the offload backend. The **prefix-cached** workload shows a **~22× cache-hit speedup** (38979 → 1743 ms, 1984/2012 tokens served from the radix cache) — this is the "before" number the candidate must beat while preserving the hit.

## Known issues found & fixed during Phase A

1. **Broken pip `ninja`** — the pip `ninja` package installed an **empty (0-byte)** binary on this platform (`pip check`: "ninja is not supported on this platform"). The editable build survived (torch BuildExtension falls back to make), but the server's runtime kernel-cache build execs `ninja` directly → `OSError: [Errno 8] Exec format error: 'ninja'` → crash during CUDA graph capture. **Fix:** `pip uninstall ninja`, then placed the known-good 370 KB ninja 1.13.2 (identical to the working original venv's) into `.venv/bin/ninja`. `pip check` now clean.
2. **`profile.py` shadowing stdlib** — see A-4. Relocated to `reference/`.
3. **Reasoning model** — Qwen3.6 emits `reasoning_content` before `content`. Benchmark runs with `enable_thinking=False` (verified the server honors `chat_template_kwargs`) and measures TTFT to the first token of any kind.
4. **Radix cache only hits for prompts ≤ `--max-prefill-length`** — `scheduler.py` only caches *non-chunk* prefills (`cache_req` is called for `batch.is_prefill and not ChunkedReq`); a prompt whose full token count exceeds `max_extend_tokens` (2048) is chunked and its prefix is not inserted. Verified empirically with unique-base prompts (no prefix sharing): full ≈1510/1910 → 2nd request ~1.8 s with `cached_tokens` ≈ 1472/1856 (HIT); full ≈2048/2058/2066–2070 → 2nd request ~38.9 s, `cached_tokens=0` (MISS). Consequence: the original `prefix-cached` workload (2048 content + 10 template = 2058 full) **could never hit** and its 39 s "median" was just uncached prefill. **Fix:** workload resized to 2000 content tokens (2010 full, 38-token margin under the cap); fixed rows show 1743 ms median with `cached_tokens=1984` (~22× hit). The buggy rows are preserved in `results-prefixcached-2058-bug.jsonl` and the corpus was rebuilt to the 2010-full prompt (hash `014350f2e4d2464b`). **A/B implication:** candidate must keep `--max-prefill-length 2048` (or raise it for both and log it) or the prefix-cached comparison is invalid; the cache-hit ceiling is ~2048 full tokens.
5. **`prefix_cached_tokens` harness field** — the OpenAI API reports radix hits at `usage.prompt_tokens_details.cached_tokens` (openai_api.py, gated by `--enable-cache-report`); the harness now reads that path (earlier rows show `None` for this field because they predate the fix — their TTFT timings are still valid).

## Phase B — Prefill diagnostic (COMPLETE, gate passed)

Goal: find where the 39 s of 2048-token uncached prefill goes, then pick the
optimization branch from evidence. Gate: `profile-summary.md`.

- **Instrumentation** (committed on `perf/sm75-prefill`): env-gated `freetoken/profiling.py`
  (`FREETOKEN_PROFILE=1`, `_AFTER`, `_COUNT`, `_OUT`); engine hook
  `maybe_profile_prefill` in the eager prefill path; `_emit_backend_report()` at
  `Engine.__init__`; always-on sub-stage NVTX (`GDN_in_proj/conv/recurrence/out_proj`,
  `MoE_router/shared_expert/routed`, `MHA_qkv_proj/attn_kernel/o_proj`,
  `MoE_topk/h2d_wait/expert_gemm`). New scripts: `serve-profile.sh`,
  `profile_driver.py`, `gemm_ceil.py`, `gemm_ab.py`, `gemm_isolate.py`,
  `gemm_rigorous.py`, `gemm_ptx.py`.
- **Captured** (uncached, radix-verified): prefill-512 (523 tok, 10.9 s) and
  prefill-2048 (2048-tok chunk, 36.6 s). Artifacts in `artifacts/sm75-prefill/profile/`.
- **Finding:** `_gemm_kernel` (W8A16 dense-projection GEMM in
  `kernel/triton/fp8_pertensor_linear.py`) is **76.5% of prefill-2048** (27.9 s of
  36.6 s) and 71.7% of prefill-512. It computes the attention/GDN dense projections
  (4.55 TFLOP) at only **0.16 TFLOPS**.
- **Root cause:** on SM75, Triton `tl.dot` (FP16 in / FP32 acc) lowers to **scalar
  CUDA-core FP16 `fma` (2048 of them) and 0 tensor-core `mma`** — PTX-verified.
  cuBLAS FP16 on the same FLOPs runs ~160× faster (~33 TFLOPS, HMMA.16816). The e4m3
  in-kernel decode is *not* the cause (pre-decoded FP16 Triton GEMM is equally slow).
- **Amdahl:** GEMM is 76.5% → fixing it to cuBLAS-FP16 speed takes prefill-2048
  36.6 s → ~8.8 s (**~76%**, ~7.6× the 10% target). Non-GEMM floor ~8.8 s.
- **Branch chosen: Phase E (dense GEMM → tensor cores).** E1 = dequant FP8→FP16 at
  load + cuBLAS (low-risk, +1.09 GB, fits at ~7.5 GiB/7.8 GiB); E2 = fix the Triton
  GEMM to emit `mma` on SM75 (keeps FP8, memory-cheaper, higher effort). E1 first.

Full detail + reproduce steps: `artifacts/sm75-prefill/profile-summary.md`.

## Phase E1 — Dense GEMM → cuBLAS FP16 (COMPLETE, gates PASSED)

Implemented the Phase B branch (E1): on SM75, the **prefill (M>1)** dense GEMM now
runs through a per-forward fp8→fp16 dequant + cuBLAS fp16 (tensor cores). Decode
(M==1) stays on the W8A16 split-K GEMV (fp8 weight, bandwidth-optimal). Dequant is
not cached (per-forward temp ~50 MB) → **no VRAM growth**. Gated behind
`FREETOKEN_E1_CUBLAS=1` + `_sm75_or_lower()` (cap < (8,0)).

- **Files:** `kernel/triton/fp8_pertensor_linear.py` (`_e1_dequant_cublas`,
  `_sm75_or_lower`, dispatch branch); `scripts/sm75/serve-e1.sh` (baseline flags +
  env flag); `scripts/sm75/e1_probe.py`, `e1_correctness.py`.
- **Correctness:** E1 vs fp32 ref rel-L2 0.00029 (current W8A16: 0.00021) — same
  numeric ballpark, no quality regression.
- **End-to-end A/B (5 reps, loopback:1920, identical flags):**

| workload | baseline | E1 | speedup | decode (base→E1) | cached (base→E1) |
|---|---|---|---|---|---|
| prefill-128 | 4033 ms | 1734 ms | 2.33× | – | – |
| prefill-512 | 10906 ms | 3078 ms | 3.54× | – | – |
| **prefill-2048** | **38979 ms** | **9734 ms** | **4.00×** | – | – |
| mixed-512 | 10941 ms | 2835 ms | 3.86× | 32.4→32.7 | – |
| mixed-2048 | 38941 ms | 9807 ms | 3.97× | 32.3→32.6 | – |
| prefix-cached | 1743 ms | 1738 ms | 1.00× | 31.2→32.0 | 1984→**1984** |
| agent-envelope | 1766 ms | 1750 ms | 1.01× | 33.4→33.4 | – |

- **Acceptance gates:** prefill-2048 9734 ms ≤ 35100 ms → **PASS** (4.0×, ~75% cut,
  matches the Amdahl prediction); prefix-cache hit preserved (cached_tokens=1984
  all reps) → **PASS**; decode flat (no regression).
- **Why not cache FP16:** +1.1 GB resident would OOM (only 0.84 GiB free after
  load); per-forward dequant costs ~178 ms for the full prefill (negligible vs
  9.7 s wall) and keeps the model fp8.

Full detail: `artifacts/sm75-prefill/e1-results.md`.

## Phase E2 — Retune Triton GEMM for tensor cores (COMPLETE, dead end ruled out)

Swept 75 Triton GEMM configs (BM∈{64,128} × BN∈{128,256} × BK∈{64,128} ×
warps∈{4,8} × stages∈{2,3}, both with and without the in-kernel `tl.trans`) and
counted tensor-core `mma`/`hmma` in each compiled PTX. **Every config emits
`mma=0`** (2048–16384 scalar FP16 `fma`, 0 `hmma`). The `tl.dot` → CUDA-core FMA
fallback is a **hard codegen limitation of Triton 3.6.0 on SM75, not a tuning
issue** — no block/warp/stage/transpose combination produces tensor-core MMA.
**E2 is a dead end; E1 (cuBLAS fp16) is the correct fix.** Script:
`scripts/sm75/e2_ptx_sweep.py`; detail: `artifacts/sm75-prefill/e2-results.md`.

## Phase E (expert branch) — NVFP4 MoE prefill → tensor-core cuBLAS (COMPLETE, gates PASSED)

After E1 cut the dense GEMM, the NVFP4 routed-expert GEMM became the dominant
prefill cost (~65% of the E1 prefill). Its `_prefill_nvfp4_moe_kernel` dequantizes
NVFP4 inline in the K-loop, but on SM75 Triton emits **0 tensor-core `mma`** (E2
proved this is a hard codegen limitation) → ~0.76 TFLOPS, **136 ms/layer** at
M=2048. **E3** routes the *prefill* NVFP4 expert GEMM through a tensor-core path:
a **fused single-launch** Triton dequant (all experts in a group, one launch, global
scale folded in so the GEMM is pure fp16) → **cuBLAS FP16 `bmm`** (gemm1+gemm2) →
vectorized sync-free scatter. Decode + sm_89+ untouched. Gated behind
`FREETOKEN_E3_CUBLAS=1` + `_sm75_or_lower()` + plain `*_and_mul` activation +
router weight on gemm2 output (model default). `FREETOKEN_E3_GROUP=8` bounds the
FP16 transient (~53 MiB at M=2048) to fit the 8 GB card alongside E1's dense temps.

- **Files:** `moe/fused_nvfp4_cublas.py` (new, self-contained); `moe/fused_nvfp4.py`
  (E3 gate + dispatch); `scripts/sm75/serve-e3.sh` (E1+E3, baseline flags,
  loopback:1920); `scripts/sm75/e_experts_{isolate,decompose,dequant,e2e,correct,integration}.py`.
- **Correctness (drop-in vs production `fused_experts_nvfp4`):** rel-L2 **0.000813**,
  cosine **1.000000**, max abs 0.046875 — pure fp16 rounding. Integration through the
  real dispatch: gate OFF 161.0 ms → gate ON **24.2 ms/layer (5.9×)**, rel-L2 0.000813.
- **End-to-end A/B (E1+E3 vs baseline, 5 reps, loopback:1920, identical flags):**

| workload | baseline | E1 | **E3 (E1+expert)** | vs base | E3 vs E1 | decode (base→E3) | cached |
|---|---|---|---|---|---|---|---|
| prefill-128 | 4033 ms | 1734 ms | **1713 ms** | 2.36× | 1.01× | – | – |
| prefill-512 | 10906 ms | 3078 ms | **1744 ms** | 6.25× | 1.76× | – | – |
| **prefill-2048** | **38979 ms** | **9734 ms** | **5768 ms** | **6.76×** | **1.69×** | – | – |
| mixed-512 | 10941 ms | 2835 ms | **1749 ms** | 6.26× | 1.62× | 32.4→33.8 | – |
| mixed-2048 | 38941 ms | 9807 ms | **5765 ms** | 6.75× | 1.70× | 32.3→33.0 | – |
| prefix-cached | 1743 ms | 1738 ms | **1718 ms** | 1.01× | 1.01× | 31.2→32.3 | 1984→**1984** |
| agent-envelope | 1766 ms | 1750 ms | **1728 ms** | 1.02× | 1.01× | 33.4→34.1 | 128 |

- **Acceptance gates:** prefill-2048 **5768 ms ≤ 35100 ms → PASS** (6.76×, ~85% cut);
  prefix-cache hit preserved (cached_tokens=1984 all reps) → PASS; decode
  flat/slightly up (M==1 NVFP4 GEMV untouched); no NaN/Inf/crash across 42 rows.
- **Memory:** three OOMs during integration each removed one transient (free
  `gate_up` before `down`; per-group activation buffer instead of global; fold the
  global scale into the dequant kernel to kill a 256 MiB fp32 temp). At group=8 the
  peak E3 transient is ~53 MiB at M=2048, which fits alongside E1's dense temps.

Full detail: `artifacts/sm75-prefill/e3-results.md`.

## Phase G — End-to-end acceptance + final deliverables (COMPLETE, ACCEPTED)

The final acceptance gate. Clean-restarted the candidate from scratch
(`launch-candidate.sh` config, E1+E3, loopback:1920) and re-ran the full
7-workload bench (5 reps + 1 warmup, identical flags to the Phase E A/B).
**Reproducibility: every workload reproduced within ~0.5%** (prefill-2048
5768→5739 ms; prefix-cached 1718→1713 ms with cached_tokens=1984 preserved;
decode 32.8–33.5). Headline: **prefill-2048 38,979→5,739 ms (6.79×, 85% cut)**.

- **Extended correctness sweep** (`scripts/sm75/g_sweep.py`, 31/31 pass,
  `test-results/g-sweep.jsonl`): 24 sequential state-slot-reuse requests
  (representative Hermes coding / structured-JSON / tool-call prompts — all 200,
  non-empty, no NaN/Inf, server healthy after each); full prefix-cache hit
  (1216 tok cached); **partial** prefix-cache hit (1792/1826 tok — the radix cache
  stores completed prefill chunks, so the shared prefix must span ≥1 chunk);
  continuation (shared prefix, different follow-ups); whole (723-tok) vs chunked
  (1322-tok) prefill both correct; cancel-mid-stream + recovery; malformed-request
  rejected + recovery.
- **Non-SM75 dispatch gate** (`scripts/sm75/g_dispatch_gate.py`, 31/31 pass,
  `test-results/g-dispatch-gate.log`): verified the dispatch decision logic across
  the full (env-gate × GPU-capability) matrix — on SM80/86/89/90 and with either
  gate off, the baseline Triton path always wins; the E1/E3 cuBLAS paths are taken
  only when gate ON **and** cap < (8,0) (and, for E3, plain `*_and_mul` activation
  with router weight on the gemm2 output). sm_89+ is never touched.
- **Correctness edge matrix** (Phase E, re-confirmed): 91/91
  (`test-results/e3-edge-matrix-results.md`).
- **No NaN/Inf/CPU-fallback/crash** across the 42-row bench + 31-check sweep.
- **Deliverables produced:** `docs/sm75-prefill.md`, `final-report.md`,
  `environment-candidate.txt`, `build.log`, `summary.csv`,
  `launch-candidate.sh`, `rollback.md`, `test-results/`, `traces/`.

**All Phase G acceptance gates PASS → candidate ACCEPTED for SM75 (Turing).**
Full detail: `artifacts/sm75-prefill/final-report.md`.

## Phase H — Post-acceptance wins (COMPLETE, 2026-10-03)

Post-acceptance optimization pass on `perf/sm75-prefill`. All work env-gated for
clean A/B; production `qwen.sh` updated only where a win was measured and safe.
Full detail: `/home/lmeek/freetoken_prefill_next_plan.md`.

### H-1 — `--max-prefill-length 2048 → 4096` (LANDED in `qwen.sh`)

At cap 2048 a 2060-token prompt is one 2048-chunk + a 12-token tail (two
forwards, two × the ~4.7 s fixed offload floor); at cap 4096 it is a single
forward. Measured (8-workload bench, `results-candidate-4096.jsonl`): prefill-2048
5768 → 4117 ms (29% faster), no OOM (1.09 GiB free), decode flat, 0 errors. Sweep
that motivated it: 4096 = 510 tok/s vs 502 at 2048; 8192 regresses to 408 (GDN +
attention grow super-linearly and overwhelm the fixed cost).

### H-2 — `--moe-prefill-hit-d2d` (MEASURED, NOT landed — user: "leave it as is")

Serves cache-resident experts via a device-side D2D gather (needs
`--moe-cache-size > 512`). Measured ~7% on cache-hit workloads (prefix-cached
1711→1587 ms, agent-envelope 1712→1586 ms, stable all 5 reps), flat on fresh
prefill. Not landed in `qwen.sh` (production stays at cache 512 where the flag is
a no-op); would need its own A/B at 130k context before landing.

### H-3 — Idea 4: sm75 GDN recurrence tile tuning (LANDED in `qwen.sh`)

The two dominant GDN kernels shipped Hopper/Ampere-tuned tile configs
(`chunk_fwd_kernel_o` hardcoded BK=128/num_warps=4; `chunk_h` num_warps=4). On
sm75 they are ~4.5× slower than the sm75-optimal config. Fix: an opt-in master
switch **`FREETOKEN_GDN_SM75_TUNE`** (default 0 = original config, so
Ampere/Hopper are byte-identical; `qwen.sh` sets it to 1). When on:
`chunk_fwd_kernel_o` BK 128→64 / num_warps 4→8, `chunk_h` num_warps 4→8.
Individual `SGLANG_GDN_CHUNK_O_*`/`SGLANG_GDN_CHUNK_H_*` knobs still override.

- **Files:** `kernel/fla/chunk_o.py` (+`import os`, `FREETOKEN_GDN_SM75_TUNE`
  gate, env-driven launch config); `kernel/fla/chunk_delta_h.py` (same gate on
  num_warps); `scripts/sm75/gdn_microbench.py` (config sweep),
  `scripts/sm75/gdn_correctness.py` (old-vs-new rel-L2),
  `scripts/sm75/profile_breakdown.py` (raw chrome-trace device breakdown).
- **Microbench gate (≥1.5×):** PASS, 4.64× (o) / 4.12× (h) / 4.49× combined at
  T=8192, validated against the measured per-call time.
- **Correctness gate:** PASS, bit-identical (rel-L2 = 0.000e+00, max abs diff 0.0)
  for both the kernel and the full `gdn_prefill_chunk_fla` output. The chunk GDN
  path is prefill-only (decode uses `gdn_decode_fla`), so decode is untouched.
- **End-to-end A/B** (`results-candidate-gdn4096.jsonl`, 8 workloads, 5 reps):
  prefill-2048 4119→**2869 ms (−30.3%)**, prefill-4096 9753→**7352 ms (−24.6%)**,
  mixed-2048 7874→**6644 ms (−15.6%)**, short-prefill flat, decode flat
  (33.5→33.4), no OOM, 0 errors.
- **Re-profile n8192** (`profile-8192-gdn/`): `chunk_fwd_kernel_o` 4431→967 ms
  (4.58×), `chunk_h` 1790→427 ms (4.19×), total device 19882→15043 ms (−24.3%).
  GDN is now 10.8% of device (was 32.5%); **full attention is the new
  co-dominant cost at 42.3%** — the next lever.

### H-4 — Idea 5: full-attention extend kernel (tile config dead-end; CUTLASS backend = 13× WIN found)

Full attention (`_extend_attention_split_kernel`, 10 layers) is 42.3% of device
post-GDN-tune. Three levers investigated:

- **Tile-config sweep — NO WIN at true geometry.** The model's full-attention
  layers are **16 q-heads / 2 kv-heads / head_dim=256** (GQA group 8), *not* the
  32/16/128 the first microbench assumed. At hd=256 the current (32,16) tile is
  already fastest (161.9 ms @ T=4096): (16,32)=164.3, (32,32)=171.2, and
  (64,16)/(64,32)/(16,64) all OOM (81–98 KiB > 64 KiB smem). An earlier "1.27×"
  figure was a **measurement artifact** (wrong geometry, 32/16/128); a config
  switch was landed then **reverted** after the server crashed on startup
  (`out of resource: shared memory, Required: 98304, limit 65536`) — the (32,64)
  tile fits only hd≤128. `attention.py` and `qwen.sh` are back to the pre-idea-5
  state (verified: empty `git diff` for `attention.py`).
- **FlashInfer / FA2 backend — NOT VIABLE on sm75.** Installed
  `flashinfer-python 0.6.18` + matching cubin/jit-cache. The FA2 prefill kernel
  fails with `no kernel image is available for execution on the device`: prebuilt
  cubins ship sm80/89/90a/100a/103a/120f only, no Turing build. FA2's pipelining
  needs sm80+ instructions. **Removed** from the venv (was ~10.5 GB).
- **CUTLASS mem-efficient (PyTorch SDPA) — 13.4× WIN on fresh prefill (new).**
  PyTorch's built-in **memory-efficient (CUTLASS) SDPA backend runs on sm75** and,
  at the true production geometry (16/2/256, T=4096, fresh), is **12.1 ms vs the
  Triton kernel's 162.0 ms = 13.4×** (measured in the same process, same inputs).
  The only wrinkle: the CUTLASS backend requires equal q/kv head counts, so GQA
  (16 qh / 2 kvh) must be expanded to MHA via `repeat_interleave` (0.4 ms + 34 MB
  at T=4096 — negligible). Correctness vs the Triton kernel: relL2 1.73e-04 (OK).
  PyTorch's Flash and cuDNN SDPA backends are **sm80–sm121 only** (rejected on
  sm75); only the CUTLASS mem-efficient + math backends run here, and mem-efficient
  is the fast one. PyTorch SDPA cannot take GQA natively (16 qh / 2 kvh → "No
  available kernel"), so the MHA expansion is mandatory.
- **CUTLASS backend — BUILT, INTEGRATED, LANDED in `qwen.sh` (H-5).** New
  `python/freetoken/attention/cutlass.py` (registered as `--attention-backend
  cutlass`; default-off, `qwen.sh` now opts in). Two prefill paths use CUTLASS
  mem-efficient SDPA; decode is always Triton:
  - *fresh* (prefix==0): dense, `is_causal=True`, GQA→MHA expand.
  - *prefix* (chunk 2+ of a chunked prefill): gather the request's full KV from the
    paged cache, GQA→MHA expand, offset-causal mask. A memory guard bails to the
    Triton extend kernel if the expansion would exceed 60% of free VRAM (it scales
    with context: ~1.1 GB at 64K).
  - **Correctness:** byte-for-byte identical to Triton (greedy, seed=1234, 29 tok).
  - **Kernel:** fresh 4096 = 13.4×, prefix 4096 (prefix=4096) = 10.1× vs the Triton
    extend kernel (`scripts/sm75/cutlass_prefix_bench.py`, relL2 2.2e-04).
  - **End-to-end (64K config, 5 reps):** flat at ≤4096 (attention is ~2% of a
    ~5 s prefill), **~2.9× at 61K tokens cold** (Triton 941 s vs CUTLASS 325 s) —
    the win scales with length because attention cost is quadratic and only 10 of 40
    layers are full-attention. `scripts/sm75/attn_microbench.py` +
    `cutlass_prefix_bench.py` kept (`--qh/--kvh/--hd`).

### H-5 — 64K context + reduced expert cache to make room for CUTLASS (LANDED in `qwen.sh`)

The CUTLASS prefix path needs headroom for the GQA→MHA expansion at long context
(~1.1 GB at 64K). The old 130K config left only 0.85 GiB free. User-authorized
levers, applied to `qwen.sh`:

- **`--num-tokens` / `--max-seq-len-override` 130000 → 64000** — KV cache 2.48 →
  1.22 GiB.
- **`--moe-cache-size` 512 → 256 + `--disable-moe-prefill-overlap`** — the overlap
  feature is what forces the 2×-experts (512) floor; disabling it lets the cache
  drop to 1× (256), freeing ~0.85 GiB (the hit-D2D feature H-2 needs >512, so it
  is off under this config).
- **`--cuda-graph-max-bs` 1 → 0** — no CUDA-graph capture (saves the graph
  scratch; decode is eager). Fine for a headless prefill-focused GPU.
- **`--attention-backend` triton → cutlass.**
- **Memory after:** 2.57 GiB free (was 0.85 GiB) — enough for the expansion at 64K.
- **Costs:** prefill-overlap off (small prefill cost), decode eager (slightly
  slower decode), context capped at 64K. Trade-off is favorable for long-prefill
  workloads (the 2.9× at 61K dwarfs the small overlap/decode costs).
- **Bench (64K config, 5 reps, `results-baseline-64k.jsonl` vs
  `results-cutlass-64k.jsonl`):** prefill-4096 4805→4696 ms (1.02×), mixed-2048
  total 15199→14014 ms (1.08×), decode 12.1→12.1 tok/s, prefix-cached flat, 0
  errors. (The ≤4096 numbers are flat because attention is a small slice there;
  the long-context win is the 61K A/B above.)

### H-6 — Decode: MoE expert-cache size is the lever (LANDED in `qwen.sh`: 256 → 512)

The decode slowdown was **config-driven, not kernel-driven**. The decode kernels
were checked for the same sm75 mis-tuning that bit the prefill GDN kernels and
are **fine**: the fp8 dense GEMV (`_gemv_splitk_kernel`) and NVFP4 Marlin GEMV
(`_decode_nvfp4_marlin_kernel`) are bandwidth-bound (read each weight once — the
correct shape for M=1), and the GDN decode recurrence uses `num_warps=1,
num_stages=3` (sensible for batch=1). No tile change needed.

The real cause: the **unified MoE slot cache** (one LRU pool shared across all 40
MoE layers) was at the 256-slot floor (set in H-5 to make room for the 64K +
CUTLASS config). Each decode step touches 40 layers × 8 experts = **320
expert-rows**, but there are 10,240 distinct expert-rows (256 experts × 40
layers), so a 256-slot pool holds only ~0.8× the per-step working set and
**thrashes** — evicting and re-streaming experts over PCIe nearly every step.

**Measured A/B (live server, runtime `POST /v1/cache/rebuild`, no restart,
299 out-tokens, prefill included):**

| slots | decode tok/s | free VRAM | 61K CUTLASS prefix path |
|---|---|---|---|
| 256 (H-5 floor) | 14.33 | 2.55 GiB | OK |
| **512 (landed)** | **18.23** | **2.01 GiB** | **OK (peak 1.195 < 0.6×2.01)** |
| 600 | 19.31 | 1.85 GiB | **BAILS to Triton** |
| 800 | 20.30 | 1.48 GiB | BAILS to Triton |
| 1024 | 21.38 | 1.13 GiB | BAILS to Triton |
| 1500 (original) | 24.69 | 0.31 GiB | BAILS to Triton |
| 2000 | rejected (503, needs 4.53 GiB) | – | – |

**512 is the sweet spot, not 1500.** The CUTLASS prefix path (the 2.9× 61K win
from H-5) has a memory guard that bails to the Triton extend kernel if the
GQA→MHA expansion peak (≈1.195 GiB at 61K) exceeds 0.6× free VRAM — i.e. it
needs **~2.0 GiB free**. Bigger caches (≥600) win more decode but leave <2.0 GiB
free and **lose the 61K prefill win**. 512 keeps 2.01 GiB free (61K CUTLASS
stays alive) *and* recovers most of the decode (+27% vs 256). The two wins are
VRAM-antagonistic; 512 is the max cache that keeps both.

- **Router is a non-issue (already pure-torch):** `triton_kernels` is hard-gated
  to sm80+ (`kernel/backend.py::_supports_sm80`), so on sm75 the MoE router
  *already* runs the pure-torch `_torch_fused_topk` (the log warning just
  confirms it). Measured 88 µs/call × 40 layers = **3.5 ms/step ≈ 9%** of a
  40.5 ms decode step — not worth changing.
- **`qwen.sh`:** `--moe-cache-size` 256 → **512** (kept
  `--disable-moe-prefill-overlap`, `--cuda-graph-max-bs 1`, 64K, CUTLASS).
  `scripts/sm75/serve-candidate-cutlass.sh` synced to 512.
- **Net decode (graphs on + 512 cache): ~14.3 → 18.2 tok/s** (the earlier ~1.8
  tok/s was the no-graphs probe). Restoring graphs (H-5 follow-up, "yes A")
  removed the per-kernel launch overhead; the cache bump removed the PCIe
  thrash. Together they take decode from the 1.8 tok/s floor to ~18 tok/s.

## Watch items / caveats

- **`--max-prefill-length 4096`** (raised from 2048 in Phase H-1). The radix cache
  only hits for prompts ≤ the cap, so the cache-hit ceiling moved 2048 → 4096 full
  tokens. `prefill-8192` is deferred (regresses: GDN + attention grow
  super-linearly and overwhelm the fixed offload cost — see the 4096 sweep).
- **Memory is tight** — `--memory-ratio 0.95` leaves only ~0.84 GiB free after init (6.9 GiB used). Watch for OOM on long prompts.
- **`triton_kernels` not installed** → pure-torch router fallback (numerically equivalent, slower). This matches the original SM75 setup, so it's part of the baseline.
- **Offload MoE backend** → decode does CPU↔GPU expert transfers; decode is slow. Mixed workloads (128 out) take minutes.
- **First-request cold start** — the very first request after server start includes Triton JIT + first CUDA-graph replay (~34s). The `--warmup 1` per workload absorbs this; reported medians exclude warmup.

## Next steps

1. **Phase G — DONE, ACCEPTED** (see Phase G section above). All 12 deliverables
   produced; clean-restart reproducibility re-run, 24-request sweep,
   chunked-vs-whole, partial-cache-hit, non-SM75 dispatch gate, and
   NaN/CPU-fallback/crash checks all pass.
2. **Make E1+E3 the default on SM75** (flip `FREETOKEN_E1_CUBLAS` and
   `FREETOKEN_E3_CUBLAS` to on-by-default for cap < (8,0)) now that acceptance
   passed — env-gated today for a clean A/B. (Post-acceptance step.)
3. **E2 — DONE, dead end** (see Phase E2): no Triton config emits `mma` on SM75.
4. **NVFP4 expert GEMM — DONE** (Phase E expert branch, E3): 136 ms/layer → 24 ms;
   the remaining prefill floor (~1.5–2 s: GDN, H2D offload, attention, routing) is
   the only further target (Phase D native GDN / Phase F).
5. Add `prefill-4096` only if `--max-prefill-length` is deliberately raised (logged)
   for both baseline and candidate.

## Commands (exact)

```bash
# install (re-runnable)
bash ~/src/freetoken-sm75-native-prefill/scripts/sm75/setup-venv.sh

# verify
~/src/freetoken-sm75-native-prefill/.venv/bin/python -c \
  'import sys,freetoken,torch;print(sys.executable);print(freetoken.__file__);print(torch.__version__,torch.version.cuda,torch.cuda.is_available())'

# serve baseline (loopback:1920)
bash ~/src/freetoken-sm75-native-prefill/scripts/sm75/serve-baseline.sh

# benchmark
~/src/freetoken-sm75-native-prefill/.venv/bin/python \
  ~/src/freetoken-sm75-native-prefill/scripts/sm75/bench.py \
  --host 127.0.0.1 --port 1920 \
  --model ~/Models/Qwen3.6-35B-A3B-NVFP4 \
  --variant baseline --reps 5 --warmup 1 \
  --out ~/src/freetoken-sm75-native-prefill/artifacts/sm75-prefill/results.jsonl \
  --corpus-out ~/src/freetoken-sm75-native-prefill/artifacts/sm75-prefill/benchmark-corpus.json
```

## H-7: MoE decode cache study (C-coding / spec-reading workload)

Question: is the landed 512-slot unified MoE expert cache sufficient for the user's
real workload (C-coding + reading specs), or is that workload skewier (fewer misses)
than the general benchmark assumed?

Method: wired a `--moe-collect-stats` flag (new, off by default) + a trigger-file dump
in the scheduler (`touch $FREETOKEN_MOE_STATS_FILE.trigger` writes the counters to JSON).
Ran the study server with `--cuda-graph-max-bs 0` (the routing histogram is only
accurate with graphs off — the captured decode graph does not re-run the host-side
`decode_freq.scatter_add_`). Fed 12 C-coding/spec prompts × 299 out-tokens = **3588
decode steps** (ring buffers, pthread, ELF parsing, io_uring, memory pools, stdatomic,
CRC-32C, RFC 793, hash tables, x86-64 ABI, merge sort, realloc).

**Result: the workload is NOT skewier than general. Routing is genuinely hard to cache.**

- Realized miss rate (unified 512-slot LRU): **60.4%** (hit 39.6%).
- Per-layer realized miss spans **47.8% (layer 20) … 90.1% (layer 0)** — early layers
are the least cacheable.
- Routing skew (per-layer histogram): working set **mean 235 / max 256** distinct
experts (of 256); only **110 experts carry 90%** of the traffic; normalized entropy
**0.844** (1.0 = perfectly uniform). The model spreads each token's 8 chosen experts
across almost the whole expert set — there is no small hot set to pin.

Oracle hit ceiling (a perfect cache holding the hottest per-layer set, static):

| slots | /layer | oracle hit |
|------:|-------:|-----------:|
| 256   | 6      | 21.6%      |
| 512   | 13     | 33.7%      |
| 1024  | 26     | 49.0%      |
| 1500  | 38     | 59.4%      |
| 2000  | 50     | 67.5%      |
| 2663  | 67     | 76.3%      |

Interpretation:
- The realized 512 hit (39.6%) already **exceeds** the static per-layer oracle at the
same budget (33.7%) — the unified pool sharing 512 slots dynamically across layers
beats the "pin the hottest per layer" model. Sharing is worth ~6 pts.
- Ceiling gain 512→1500 slots is +25.7 pts of hit (i.e. −25.7 pts of miss), matching the
+29% decode A/B measured earlier (256→18.23 tok/s at 512, 24.69 at 1500). So the A/B
speedup tracks the cache ceiling directly.
- **512 stays the sweet spot**: it keeps the 61K CUTLASS prefill guard (needs >2.0 GiB
free) while taking most of the available cache win. A bigger cache keeps paying up to
1500, but only by trading away the prefill win (bails the CUTLASS guard at ≥600).
- Bottom line for the user's recollection ("there wasn't that many misses"): the *general*
benchmark and this C-coding workload are about the same — routing is near-uniform
(entropy 0.84) and the cache is fundamentally limited, not mis-sized. The earlier
"few misses" memory was most likely the radix/KV prefix cache (H-3, ~7% hit), a
different mechanism.

Artifacts: `~/.hermes/cache/scratch/moe_cache_study.json` (full 40×256 histogram +
per-layer realized stats), `analyze_cache_study.py`, `cache_study_run.py`.

## H-8: Hybrid decode A/B (GPU offload + CPU overflow) — NOT a win on this box

Question: the engine has a `--moe-backend hybrid` mode — GPU fetches a capped
subset of misses over PCIe, the CPU computes the overflow experts in parallel
(overlapped), partials merge. Could that hide the ~20.7 ms/step PCIe cost?

Setup: identical to production `qwen.sh` (512 slots, cutlass, graphs on, 64K)
except `--moe-backend hybrid`. Auto fetch-fraction from the `ft bench bw`
profile = 42.2% (CPU computes 57.8% of each step's misses).

| backend | tok/s (299 out-tok, 3 runs) |
|---------|------------------------------|
| offload (production, LANDED) | 17.68 / 18.06 / 18.12 → **18.1** |
| hybrid, auto 42% fetch        | 16.43 / 16.40 / 17.30 → **16.7** |

Hybrid is **~8% SLOWER**, not faster. Root cause (found in `~/.cache/freetoken/benchbw.json`):
the profile that drives the auto fetch-fraction was benched on a **23-core** CPU
(same RTX 2080, different machine) where CPU MoE = 16.6 GB/s > PCIe gather 8.2 GB/s
(ratio 2.03 → "hybrid recommended"). This box has only **6 physical cores**
(12 threads), so the CPU side can only use ~5-6 cores — it is the bottleneck,
not the PCIe bus. The auto 42% fetch-fraction is miscalibrated for this CPU.
A fetch-fraction sweep (0.8/1.0) was started to confirm the trend but the
multi-server-restart sweep exceeded the tool timeout; the single-point hybrid
measurement above is conclusive enough: offload is optimal here.

**Conclusion: keep `--moe-backend offload` (the default). Do NOT enable hybrid
on this 6-core box.** Hybrid only pays off when the CPU is much faster than the
PCIe bus (≥2×), which requires many more cores than this machine has.

## H-9: Python engine overhead — NOT a decode bottleneck

Question: the engine is largely Python — does it add per-step latency, or is
there a tight Python spin loop feeding the PCIe expert transfers that could be
optimized / replaced with high-speed DMA?

Findings (code-level, not speculation):
- **CUDA graphs remove the per-step Python loop from the hot path.** The
per-layer driver (`moe.py:315` `cache.ensure_experts(layer, ids);
cache.copy_missing()`) is a Python loop over 40 layers, but it runs **once at
graph-capture time**. In production (`--cuda-graph-max-bs 1`) each decode step
is a single `graph.replay()` — the 40 copy launches + GEMMs are baked into the
graph. Python is not re-executed per step.
- **The expert transfer is already DMA, not a Python spin loop.** The CPU expert
bank is **pinned** host memory (`offload_cache.py:268` "host (CPU pinned) expert
source banks"), so the H2D copy is DMA-eligible — the DMA engine moves the
bytes, the CPU is not in the data path.
- **It's already fused**: `fast_index_copy_multi_jit` does "one launch copies the
missing rows for every bank" instead of one launch per layer. That is the
high-speed path (pinned memory + single fused kernel), not a per-layer Python
`memcpy`.
- Python only does bookkeeping (routing, slot assignment) at capture time and a
graph-replay call per step. There is no Python overhead to reclaim on decode.

**The real ceiling is the PCIe bus itself.** Measured on this GPU (benchbw.json):
PCIe gather (the actual scattered 1.70 MB expert transfer rate) = **8.19 GB/s**,
vs linear H2D ceiling 10.34 GB/s. That is a bus limit — no Python or kernel
tuning makes it faster. The only levers are: fewer misses (workload-inherent,
H-7), smaller per-expert transfer (already 4-bit NVFP4, minimal), or offloading
some compute to the CPU in parallel (hybrid, H-8 — needs a faster CPU). None
available on this 6-core / PCIe-3.0 box.
