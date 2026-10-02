#!/usr/bin/env bash
# Phase A: launch the NEW project's baseline server on LOOPBACK:1920.
# Isolated from the original service (0.0.0.0:8888) and its caches.
# Usage: scripts/sm75/serve-baseline.sh   (foreground; or background it yourself)
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

# project-local cache dirs (baseline variant)
export XDG_CACHE_HOME="$PROJECT/.cache/baseline"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p "$XDG_CACHE_HOME"

# --- endpoint: loopback only, unused port 1920 (NOT 0.0.0.0:8888) ---
HOST=127.0.0.1
PORT=1920

echo "[serve-baseline] host=$HOST port=$PORT"
echo "[serve-baseline] caches -> $XDG_CACHE_HOME"
echo "[serve-baseline] freetoken -> $(python -c 'import freetoken;print(freetoken.__file__)')"

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
