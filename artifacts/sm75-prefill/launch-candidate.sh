#!/usr/bin/env bash
# launch-candidate.sh — Phase G candidate (E1+E3) launch, loopback:1920.
#
# This is the acceptance launch configuration for the FreeToken SM75 native-prefill
# candidate: the baseline serve flags UNCHANGED, plus the two Phase E env gates
# (E1 dense GEMM -> cuBLAS fp16, E3 NVFP4 expert GEMM -> tensor-core cuBLAS) and the
# E3 group cap that bounds the FP16 dequant transient to fit the 8 GB card.
#
# The ORIGINAL service (~/src/freetoken-sm75, 0.0.0.0:8888) is untouched and remains
# the default. This script only starts the independent test server on loopback:1920.
# See rollback.md to stop it and return to the original.
set -uo pipefail

PROJECT="$HOME/src/freetoken-sm75-native-prefill"
cd "$PROJECT"
source .venv/bin/activate

# --- env: mirror the working SM75 env, redirect ALL caches into the project ---
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="7.5"
export TVM_FFI_CUDA_ARCH_LIST="7.5"
export FREETOKEN_KERNEL_CACHE_ARCHES="7.5"
export MAX_JOBS=4
export PYTHONDONTWRITEBYTECODE=1

# --- Phase E candidate gates (SM75-only; no-op on sm_89+) ---
export FREETOKEN_E1_CUBLAS=1          # E1: prefill dense GEMM via per-forward fp8->fp16 dequant + cuBLAS fp16
export FREETOKEN_E3_CUBLAS=1          # E3: prefill NVFP4 expert GEMM via tensor-core cuBLAS
export FREETOKEN_E3_GROUP=16          # bounds the FP16 dequant transient (gu = group*4 MiB) to fit 8 GB

# project-local cache dirs (candidate variant)
XDG_CACHE_HOME="$PROJECT/.cache/candidate"
export XDG_CACHE_HOME
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p "$XDG_CACHE_HOME"

HOST=127.0.0.1
PORT=1920

echo "[launch-candidate] host=$HOST port=$PORT  E1=$FREETOKEN_E1_CUBLAS E3=$FREETOKEN_E3_CUBLAS group=$FREETOKEN_E3_GROUP"
echo "[launch-candidate] caches -> $XDG_CACHE_HOME"
echo "[launch-candidate] freetoken -> $(python -c 'import freetoken;print(freetoken.__file__)')"

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
