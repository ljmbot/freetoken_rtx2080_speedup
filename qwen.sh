#!/usr/bin/env bash
set -euo pipefail
cd "/home/lmeek/src/freetoken-sm75"
source "/home/lmeek/freetoken-sm75-env.sh"
source "/home/lmeek/src/freetoken-sm75/.venv/bin/activate"
exec ft serve \
 --model /home/lmeek/Models/Qwen3.6-35B-A3B-NVFP4 \
 --served-model-name Qwen3.6-35B-A3B-NVFP4 \
 --dtype float16 \
 --attention-backend triton \
 --moe-backend offload \
 --nvfp4-backend triton \
 --expert-load serial \
 --moe-cache-size 1500 \
 --memory-ratio 0.95 \
 --num-tokens 32768  \
 --max-seq-len-override 32768 \
 --max-output-tokens 4096 \
 --max-prefill-length 2048 \
 --max-running-requests 1 \
 --cuda-graph-max-bs 1 \
 --cache-type radix \
 --enable-cache-report \
 --sampling-defaults model \
 --reasoning-parser auto \
 --host 0.0.0.0 \
 --port 8888 

