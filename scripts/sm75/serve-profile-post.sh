#!/usr/bin/env bash
# Post-E1+E3 re-profile server: qwen.sh settings (E1+E3 GROUP=16, moe-cache-size 512,
# radix, 0.95) with context 64000 (per user) and the Phase B profiler armed.
# Loopback:1920 so it never collides with the production endpoint.
set -uo pipefail

export PROJECT="$HOME/src/freetoken-sm75-native-prefill"
cd "$PROJECT"
source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="7.5"
export TVM_FFI_CUDA_ARCH_LIST="7.5"
export FREETOKEN_KERNEL_CACHE_ARCHES="7.5"
export MAX_JOBS=4
export PYTHONDONTWRITEBYTECODE=1

# Phase E gates (identical to qwen.sh)
export FREETOKEN_E1_CUBLAS=1
export FREETOKEN_E3_CUBLAS=1
export FREETOKEN_E3_GROUP=16

# project-local cache (candidate dir: kernels already compiled there)
export XDG_CACHE_HOME="$PROJECT/.cache/candidate"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p "$XDG_CACHE_HOME"

# profiler: skip first 4 prefill forwards (warmup/autotune), capture the next 3
export FREETOKEN_PROFILE="${FREETOKEN_PROFILE:-1}"
export FREETOKEN_PROFILE_AFTER="${FREETOKEN_PROFILE_AFTER:-4}"
export FREETOKEN_PROFILE_COUNT="${FREETOKEN_PROFILE_COUNT:-3}"
export FREETOKEN_PROFILE_OUT="${FREETOKEN_PROFILE_OUT:-$PROJECT/artifacts/sm75-prefill/profile-post}"
mkdir -p "$FREETOKEN_PROFILE_OUT"

HOST=127.0.0.1
PORT="${PORT:-1920}"
MAX_PREFILL="${MAX_PREFILL:-2048}"

echo "[serve-profile-post] host=$HOST port=$PORT after=4 count=3"
echo "[serve-profile-post] out -> $FREETOKEN_PROFILE_OUT"

exec ft serve \
  --model "$HOME/Models/Qwen3.6-35B-A3B-NVFP4" \
  --served-model-name Qwen3.6-35B-A3B-NVFP4 \
  --dtype float16 \
  --attention-backend triton \
  --moe-backend offload \
  --nvfp4-backend triton \
  --expert-load serial \
  --moe-cache-size 512 \
  --memory-ratio 0.95 \
  --num-tokens 64000 \
  --max-seq-len-override 64000 \
  --max-output-tokens 4096 \
  --max-prefill-length "$MAX_PREFILL" \
  --max-running-requests 1 \
  --cuda-graph-max-bs 1 \
  --cache-type radix \
  --enable-cache-report \
  --sampling-defaults model \
  --reasoning-parser auto \
  --host "$HOST" \
  --port "$PORT"
