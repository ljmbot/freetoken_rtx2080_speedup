#!/usr/bin/env bash
# serve-candidate-hitd2d.sh — idea-3 A/B candidate: IDENTICAL to
# serve-candidate-4096.sh (Phase G config + cap 4096, moe-cache-size 1500)
# plus --moe-prefill-hit-d2d (prefill hit/miss split: resident experts served
# device-side via D2D gather, only misses cross PCIe).
# Isolates the hit-D2D effect against the candidate-4096 baseline.
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

echo "[serve-candidate-hitd2d] host=$HOST port=$PORT  E1=$FREETOKEN_E1_CUBLAS E3=$FREETOKEN_E3_CUBLAS group=$FREETOKEN_E3_GROUP max-prefill=4096 hit-d2d=ON cache=1500"
echo "[serve-candidate-hitd2d] freetoken -> $(python -c 'import freetoken;print(freetoken.__file__)')"

exec ft serve \
  --model "$HOME/Models/Qwen3.6-35B-A3B-NVFP4" \
  --served-model-name Qwen3.6-35B-A3B-NVFP4 \
  --dtype float16 \
  --attention-backend triton \
  --moe-backend offload \
  --nvfp4-backend triton \
  --expert-load serial \
  --moe-cache-size 1500 \
  --moe-prefill-hit-d2d \
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
