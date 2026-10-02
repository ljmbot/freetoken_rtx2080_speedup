#!/usr/bin/env bash
# Phase E3 (NVFP4 expert branch): same server as serve-e1.sh + FREETOKEN_E3_CUBLAS=1
# (tensor-core NVFP4 expert GEMM: fused single-launch dequant of a bounded expert group
# + cuBLAS fp16 bmm + vectorized sync-free scatter, SM75 only). E1 (dense GEMM) stays ON
# so this is the full prefill candidate. Flags are IDENTICAL to baseline for A/B
# comparability; only the env flags and cache dir differ.
# Usage: scripts/sm75/serve-e3.sh   (foreground; or background it yourself)
set -uo pipefail

export PROJECT="$HOME/src/freetoken-sm75-native-prefill"
cd "$PROJECT"
source .venv/bin/activate

# --- env: mirror the working env, but redirect ALL caches into the project ---
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="7.5"
export TVM_FFI_CUDA_ARCH_LIST="7.5"
export FREETOKEN_KERNEL_CACHE_ARCHES="7.5"
export MAX_JOBS=4
export PYTHONDONTWRITEBYTECODE=1

# Phase E1: prefill dense GEMM via per-forward fp8->fp16 dequant + cuBLAS fp16 (SM75 only).
export FREETOKEN_E1_CUBLAS=1
# Phase E3: prefill NVFP4 expert GEMM via tensor-core path (SM75 only).
export FREETOKEN_E3_CUBLAS=1
# group bounds the FP16 dequant transient (gu = group*1024*2048*2 B). The 8 GB card
# has only ~0.84 GiB free after load and E1's dense temps + base activations consume
# ~683 MB of it, so gu must stay <= ~140 MiB -> group=16 (gu=64 MiB). group=64 (256 MiB)
# OOMs at M>=512. Speed is ~independent of group (same total dequant/bmm work).
export FREETOKEN_E3_GROUP=16

# project-local cache dirs (E3 variant)
export XDG_CACHE_HOME="$PROJECT/.cache/e3"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p "$XDG_CACHE_HOME"

# --- endpoint: loopback only, unused port 1920 (NOT 0.0.0.0:8888) ---
HOST=127.0.0.1
PORT=1920

echo "[serve-e3] host=$HOST port=$PORT  E1=$FREETOKEN_E1_CUBLAS E3=$FREETOKEN_E3_CUBLAS group=$FREETOKEN_E3_GROUP"
echo "[serve-e3] caches -> $XDG_CACHE_HOME"
echo "[serve-e3] freetoken -> $(python -c 'import freetoken;print(freetoken.__file__)')"

exec ft serve \
  --model "$HOME/Models/Qwen3.6-35B-A3B-NVFP4" \
  --served-model-name Qwen3.6-35B-A3B-NVFP4 \
  --dtype float16 \
  --attention-backend triton \
  --moe-backend offload \
  --nvfp4-backend triton \
  --expert-load serial \
  --moe-cache-size 1500 \
  --memory-ratio 0.95 \
  --num-tokens 32768 \
  --max-seq-len-override 32768 \
  --max-output-tokens 4096 \
  --max-prefill-length 2048 \
  --max-running-requests 1 \
  --cuda-graph-max-bs 1 \
  --cache-type radix \
  --enable-cache-report \
  --sampling-defaults model \
  --reasoning-parser auto \
  --host "$HOST" \
  --port "$PORT"
