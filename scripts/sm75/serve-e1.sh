#!/usr/bin/env bash
# Phase E1: same server as serve-baseline.sh + FREETOKEN_E1_CUBLAS=1 (per-forward fp8->fp16
# dequant + cuBLAS fp16 for the prefill dense GEMM on SM75). Flags are IDENTICAL to baseline
# for A/B comparability; only the env flag and cache dir differ.
# Usage: scripts/sm75/serve-e1.sh   (foreground; or background it yourself)
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

# project-local cache dirs (E1 variant)
export XDG_CACHE_HOME="$PROJECT/.cache/e1"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p "$XDG_CACHE_HOME"

# --- endpoint: loopback only, unused port 1920 (NOT 0.0.0.0:8888) ---
HOST=127.0.0.1
PORT=1920

echo "[serve-e1] host=$HOST port=$PORT  FREETOKEN_E1_CUBLAS=$FREETOKEN_E1_CUBLAS"
echo "[serve-e1] caches -> $XDG_CACHE_HOME"
echo "[serve-e1] freetoken -> $(python -c 'import freetoken;print(freetoken.__file__)')"

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
