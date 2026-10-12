# Launching Ornith-1.0-35B-NVFP4 on 8 GB consumer GPUs (sm75)

This fork adds first-class support for running **RedHat (Neural Magic)
[Ornith-1.0-35B-NVFP4](https://huggingface.co/RedHatAI/Ornith-1.0-35B-NVFP4)**
(a 35B-parameter Qwen3.5-MoE, W4A4 NVFP4) on **8 GB Turing/older GPUs
(RTX 2080/2080 Ti, sm75)** with a 64K context window — verified on an RTX 2080.
The model is pulled straight from Hugging Face; no local checkpoint is needed.

## Install

```bash
git clone <this-repo> && cd <this-repo>
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

## Launch

```bash
export FREETOKEN_E1_CUBLAS=1        # prefill dense GEMM -> cuBLAS fp16 tensor cores
export FREETOKEN_E3_CUBLAS=1        # prefill NVFP4 expert GEMM -> tensor-core cuBLAS
export FREETOKEN_E3_GROUP=16        # bounds the fp16 dequant transient to fit 8 GB
export FREETOKEN_GDN_SM75_TUNE=1    # sm75-tuned GDN recurrence tiles (~4.5x faster)
export FREETOKEN_LOAD_VISION=0      # text-only: skip the vision tower (~0.83 GiB)
export PATH="$PWD/.venv/bin:$PATH"  # JIT store kernel needs ninja from the venv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # cuts allocator fragmentation at the 8 GB edge

ft serve \
 --model RedHatAI/Ornith-1.0-35B-NVFP4 \
 --served-model-name Ornith-1.0-35B-PLAIN-NVFP4 \
 --dtype float16 \
 --attention-backend cutlass \
 --moe-backend offload \
 --nvfp4-backend triton \
 --expert-load serial \
 --moe-cache-size 970 \
 --disable-moe-prefill-overlap \
 --memory-ratio 0.95 \
 --num-tokens 65536 \
 --max-seq-len-override 65536 \
 --max-output-tokens 16000 \
 --max-prefill-length 4096 \
 --max-running-requests 1 \
 --cuda-graph-max-bs 1 \
 --cache-type radix \
 --enable-cache-report \
 --kv-q8 \
 --sampling-defaults model \
 --temperature 0.5 \
 --reasoning-parser auto \
 --host 0.0.0.0 \
 --port 8888
```

The server downloads the model (~23.7 GB, 2 shards) into the HF cache on first
launch and then serves an OpenAI-compatible API at `http://localhost:8888/v1`.

> **Run console-only (no desktop session).** A GUI session consumes ~0.3 GiB of
> VRAM and can OOM a long prefill at these settings. A terminal-only login
> leaves the full budget for the engine.

## What the important switches do

| Switch / env | Purpose |
|---|---|
| `--attention-backend cutlass` | Uses PyTorch's CUTLASS memory-efficient SDPA for prefill (~10–13x faster than the Triton extend kernel at production geometry). A built-in VRAM guard bails to the Triton kernel automatically if a long-context expansion would not fit — no OOMs. Decode always uses the Triton paged kernel. |
| `--moe-cache-size 970` | Number of MoE experts resident in GPU memory. **970 is the measured safe ceiling for `radix` at 64K on 8 GB** (16K-prefill peak 7772 MiB vs the 7782 MiB usable; 1024 boots fine but OOMs mid-prefill — startup "free memory" is NOT a safe fit indicator, gate bumps on a 16K-prefill peak). With `--cache-type naive` the ceiling is ~950 (955 is rejected; radix's 9-slot GDN state pool costs ~430 MiB more, funded by the expert cut). Lower it (e.g. 512) to free VRAM for a bigger context or longer outputs; raise it only on >8 GB cards. |
| `--num-tokens 65536` + `--max-seq-len-override 65536` | 64K context window. |
| `--max-prefill-length 4096` | Prefill chunk size. 4096 cuts time-to-first-token (~15% faster prefill than 2048, ~+190 MiB peak). Do not raise further on 8 GB — GDN + attention cost is super-linear in chunk length. |
| `--kv-q8` | Stores the paged K/V cache in Q8_0 (int8 + per-head scale) instead of fp16 — halves KV VRAM; dequantized in-kernel so compute stays fp16. Required for the 64K budget on 8 GB. |
| `--cache-type radix` | Radix-tree prefix caching — **recommended for chat/multi-turn**: a request whose prompt shares a prefix with a cached conversation skips the entire forward for the cached tokens (MoE FFN + attention, all 40 layers), ~6x end-to-end on a 16K-history turn. Cost: the GDN (Mamba-style) state pool grows 2 → 9 slots (~123 → ~553 MiB), funded by cutting ~108 experts (1078 → 970) ≈ −3 tok/s decode. Use `naive` for one-shot unique-prompt workloads (zero prefix reuse): max expert residency, ~30 tok/s decode. |
| `--expert-load serial` | Streams experts one at a time (lowest peak during load). |
| `--disable-moe-prefill-overlap` | Disables the prefill double-buffer (its extra buffers don't fit 8 GB at 64K). |
| `--memory-ratio 0.95` | Fraction of VRAM the engine may use. |
| `FREETOKEN_E1_CUBLAS=1` | Prefill dense (non-expert) GEMMs run on cuBLAS fp16 tensor cores instead of Triton. No-op on sm89+. |
| `FREETOKEN_E3_CUBLAS=1` | Prefill NVFP4 expert GEMMs run on tensor-core cuBLAS (dequant + fp16 GEMM). No-op on sm89+. |
| `FREETOKEN_E3_GROUP=16` | Processes experts in groups of 16 so the fp16 dequant working set fits 8 GB. |
| `FREETOKEN_GDN_SM75_TUNE=1` | sm75-tuned tile configs for the GDN (gated delta net) recurrence kernels — ~4.5x faster, bit-identical. No-op elsewhere. |
| `FREETOKEN_LOAD_VISION=0` | Skip loading the vision tower (Ornith is multimodal; text-only serving). |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Cuts allocator fragmentation; needed to land 970 experts under the 7782 MiB peak ceiling at the 8 GB edge. |

## Measured performance (RTX 2080, console-only, 2026-10-12)

| Config | Prefill (unique 16K) | Prefill (multi-turn, 16K history) | Decode (2048-token soak) | Peak VRAM (16K prefill) |
|---|---|---|---|---|
| `radix`, 970 experts, 64K (above) | ~400 tok/s | ~6x faster e2e (prefix hit skips the cached forward) | 26–27 tok/s | 7772 MiB |
| `naive`, 1078 experts, 72K | ~385 tok/s | no prefix reuse (full re-prefill) | ~30 tok/s | 7652 MiB |

Unique-prompt prefill is at parity between the two cache types; the radix win
exists only when requests share conversation history. Decode tracks expert
residency (~3.05 MiB/expert): 950 → 25–26, 970 → 26–27, 1078 → ~30 tok/s.

## Tuning for other GPUs

- **>8 GB (12/16 GB):** raise `--moe-cache-size` (e.g. 1100–1400) and drop
  `--kv-q8` if you prefer fp16 KV.
- **Longer context:** lower `--moe-cache-size` to free the cache budget, or
  enable `--kv-q8` (already on).
- **Newer GPUs (sm89+):** the `FREETOKEN_E1/E3_CUBLAS` and `FREETOKEN_GDN_SM75_TUNE`
  env vars are no-ops (gated to sm75 and lower); the rest of the config still applies.

## Smoke test

```bash
curl -s http://localhost:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Ornith-1.0-35B-PLAIN-NVFP4","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":8}'
```

## Launching Qwen3-Coder-Next-NVFP4 (80B-A3B)

The same sm75 path serves the larger **80B-A3B
[Qwen3-Coder-Next-NVFP4](https://huggingface.co/RedHatAI/Qwen3-Coder-Next-NVFP4)**
(Qwen3Next MoE: 48 layers — 12 full-attention + 36 GDN — 512 experts, top-10).
Everything above applies except:

- **Much more RAM is needed.** Coder-Next offloads **~43.5 GB of experts** to
  host memory (Ornith offloads ~18 GB). On a 46 GB-RAM machine that leaves only
  ~2–3 GB of headroom, so expert fetches will thrash the page cache under
  load — that, not VRAM, is the real decode-speed risk. A 64 GB RAM machine
  removes the risk.
- **Tighter VRAM budget.** The expert cache is larger, so the 64K ceiling
  moves: use `--moe-cache-size 1078` (not 950) with a **72K** context. That
  fits at init (~7.9 GiB used) but peaks right at the 8 GB edge — if it OOMs,
  step `--moe-cache-size` down to ~1000 or the context to ~64K.

```bash
export FREETOKEN_E1_CUBLAS=1
export FREETOKEN_E3_CUBLAS=1
export FREETOKEN_E3_GROUP=16
export FREETOKEN_GDN_SM75_TUNE=1
export FREETOKEN_LOAD_VISION=0      # Coder-Next is text-only; harmless no-op
export PATH="$PWD/.venv/bin:$PATH"

ft serve \
 --model RedHatAI/Qwen3-Coder-Next-NVFP4 \
 --served-model-name Qwen3-Coder-Next-NVFP4 \
 --dtype float16 \
 --attention-backend cutlass \
 --moe-backend offload \
 --nvfp4-backend triton \
 --expert-load serial \
 --moe-cache-size 1078 \
 --disable-moe-prefill-overlap \
 --memory-ratio 0.95 \
 --num-tokens 72000 \
 --max-seq-len-override 72000 \
 --max-output-tokens 16000 \
 --max-prefill-length 4096 \
 --max-running-requests 1 \
 --cuda-graph-max-bs 1 \
 --cache-type naive \
 --enable-cache-report \
 --kv-q8 \
 --kv-v-bits 5 \
 --sampling-defaults model \
 --temperature 0.5 \
 --reasoning-parser auto \
 --host 0.0.0.0 \
 --port 8888
```

The only new switch vs Ornith is `--kv-v-bits 5` (K8V5): pack the V slab to
5 bits on top of `--kv-q8` to claw back the extra KV the larger model needs.
Run **console-only** (no desktop session) as with Ornith.
