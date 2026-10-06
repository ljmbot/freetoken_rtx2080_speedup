#!/usr/bin/env bash
set -euo pipefail
cd "/home/lmeek/src/freetoken-sm75-native-prefill"
source "/home/lmeek/src/freetoken-sm75-native-prefill/.venv/bin/activate"
export FREETOKEN_E1_CUBLAS=1
export FREETOKEN_E3_CUBLAS=1
export FREETOKEN_E3_GROUP=16
export FREETOKEN_GDN_SM75_TUNE=1
# Ensure the JIT store kernel can find ninja (it's in the venv bin).
export PATH="$PWD/.venv/bin:$PATH"
exec ft serve \
 --model /home/lmeek/Models/Qwen3.6-35B-A3B-NVFP4 \
 --served-model-name Qwen3.6-35B-A3B-NVFP4 \
 --dtype float16 \
 --attention-backend cutlass \
 --moe-backend offload \
 --nvfp4-backend triton \
 --expert-load serial \
 --moe-cache-size 640 \
 --disable-moe-prefill-overlap \
 --memory-ratio 0.95 \
 --num-tokens 64000 \
 --max-seq-len-override 64000 \
 --max-output-tokens 4096 \
 --max-prefill-length 4096 \
 --max-running-requests 1 \
 --cuda-graph-max-bs 1 \
 --cache-type radix \
 --enable-cache-report \
 --kv-q8 \
 --sampling-defaults model \
 --reasoning-parser auto \
 --host 0.0.0.0 \
 --port 8888
