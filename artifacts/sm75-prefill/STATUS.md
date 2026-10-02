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

## Watch items / caveats

- **`--max-prefill-length 2048`** caps prefill. The `prefill-4096` workload is **deferred** (will hit the cap / OOM on 8 GB). Baseline measured under the same cap for a fair A/B; 4096 to be added only if the cap is deliberately raised (logged) for both baseline and candidate.
- **Memory is tight** — `--memory-ratio 0.95` leaves only ~0.84 GiB free after init (6.9 GiB used). Watch for OOM on long prompts.
- **`triton_kernels` not installed** → pure-torch router fallback (numerically equivalent, slower). This matches the original SM75 setup, so it's part of the baseline.
- **Offload MoE backend** → decode does CPU↔GPU expert transfers; decode is slow. Mixed workloads (128 out) take minutes.
- **First-request cold start** — the very first request after server start includes Triton JIT + first CUDA-graph replay (~34s). The `--warmup 1` per workload absorbs this; reported medians exclude warmup.

## Next steps (after benchmark completes)

1. **A-12 Finalize** — compute per-workload medians (TTFT, decode tok/s, total) from `results.jsonl`; write `environment-candidate.txt`; commit the reproduced baseline on `baseline/local-sm75`; tag `baseline`; create `perf/sm75-prefill`.
2. Optionally add `prefill-4096` if the cap is raised.

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
