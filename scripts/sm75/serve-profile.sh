#!/usr/bin/env bash
# Phase B: launch the NEW project's server on LOOPBACK:1920 WITH the prefill
# profiler armed via env (the engine runs in a spawned subprocess, so the
# profiler must be triggered at launch, not over HTTP).
#
# Profiling env:
#   FREETOKEN_PROFILE=1            enable the profiler + backend report
#   FREETOKEN_PROFILE_AFTER=<n>    skip the first n prefill forwards (warmup)
#   FREETOKEN_PROFILE_COUNT=<c>    capture the next c prefill forwards
#   FREETOKEN_PROFILE_OUT=<dir>    trace/table output (relative to $PROJECT)
#
# Usage: scripts/sm75/serve-profile.sh [AFTER] [COUNT]
#   default: AFTER=1 COUNT=1  (skip 1 warmup prefill, capture the next 1)
set -uo pipefail

AFTER="${1:-1}"
COUNT="${2:-1}"

export PROJECT="$HOME/src/freetoken-sm75-native-prefill"
cd "$PROJECT"
source .venv/bin/activate

# --- env: mirror serve-baseline.sh exactly (A/B comparability) ---
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="7.5"
export TVM_FFI_CUDA_ARCH_LIST="7.5"
export FREETOKEN_KERNEL_CACHE_ARCHES="7.5"
export MAX_JOBS=4
export PYTHONDONTWRITEBYTECODE=1

# project-local cache dirs (same as baseline so kernels are already compiled)
export XDG_CACHE_HOME="$PROJECT/.cache/baseline"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p "$XDG_CACHE_HOME"

# --- Phase B profiler ---
export FREETOKEN_PROFILE=1
export FREETOKEN_PROFILE_AFTER="$AFTER"
export FREETOKEN_PROFILE_COUNT="$COUNT"
export FREETOKEN_PROFILE_OUT="$PROJECT/artifacts/sm75-prefill/profile"
mkdir -p "$FREETOKEN_PROFILE_OUT"

HOST=127.0.0.1
PORT=1920

echo "[serve-profile] host=$HOST port=$PORT profile=1 after=$AFTER count=$COUNT"
echo "[serve-profile] out -> $FREETOKEN_PROFILE_OUT"
echo "[serve-profile] freetoken -> $(python -c 'import freetoken;print(freetoken.__file__)')"

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
