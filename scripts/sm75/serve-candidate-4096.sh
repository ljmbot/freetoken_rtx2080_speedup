#!/usr/bin/env bash
# serve-candidate-4096.sh — regression launch: IDENTICAL to launch-candidate.sh
# (Phase G acceptance config) except --max-prefill-length 2048 -> 4096.
# Used to verify the 4096 cap change does not regress decode / prefix-cached
# and to confirm the expected prefill-2048 improvement (single chunk, not 2).
set -uo pipefail

PROJECT="$HOME/src/freetoken-sm75-native-prefill"
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

export FREETOKEN_E1_CUBLAS=1
export FREETOKEN_E3_CUBLAS=1
export FREETOKEN_E3_GROUP=16

XDG_CACHE_HOME="$PROJECT/.cache/candidate"
export XDG_CACHE_HOME
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p "$XDG_CACHE_HOME"

HOST=127.0.0.1
PORT=1920

echo "[serve-candidate-4096] host=$HOST port=$PORT  E1=$FREETOKEN_E1_CUBLAS E3=$FREETOKEN_E3_CUBLAS group=$FREETOKEN_E3_GROUP max-prefill=4096"
echo "[serve-candidate-4096] freetoken -> $(python -c 'import freetoken;print(freetoken.__file__)')"

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
  --max-prefill-length 4096 \
  --max-running-requests 1 \
  --cuda-graph-max-bs 1 \
  --cache-type radix \
  --enable-cache-report \
  --sampling-defaults model \
  --reasoning-parser auto \
  --host "$HOST" \
  --port "$PORT"
