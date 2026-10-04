#!/usr/bin/env bash
set -euo pipefail
cd "/home/lmeek/src/freetoken-sm75-native-prefill"
#source "/home/lmeek/freetoken-sm75-env.sh"
source "/home/lmeek/src/freetoken-sm75-native-prefill/.venv/bin/activate"
# Phase E SM75 prefill optimizations (acceptance passed, Phase G, commit c47f7c4):
# E1 = prefill dense GEMM -> cuBLAS fp16 (tensor cores)
# E3 = prefill NVFP4 expert GEMM -> tensor-core cuBLAS (GROUP bounds the fp16 transient to fit 8 GB)
# No-op on sm_89+ (gated by _sm75_or_lower); decode (M==1) is never touched.
export FREETOKEN_E1_CUBLAS=1
export FREETOKEN_E3_CUBLAS=1
export FREETOKEN_E3_GROUP=16
# Idea-4: sm75-tuned GDN recurrence tile configs (chunk_fwd_kernel_o + chunk_h).
# ~4.5x faster on the recurrence (147.5->31.8 and 54.8->13.3 ms/call at T=8192),
# bit-identical output, decode untouched. Opt-in: default 0 keeps the original
# Hopper/Ampere config, so this is a no-op on non-sm75 forks.
export FREETOKEN_GDN_SM75_TUNE=1
exec ft serve \
 --model /home/lmeek/Models/Qwen3.6-35B-A3B-NVFP4 \
 --served-model-name Qwen3.6-35B-A3B-NVFP4 \
 --dtype float16 \
 --attention-backend cutlass \
 --moe-backend offload \
 --nvfp4-backend triton \
 --expert-load serial \
 --moe-cache-size 512 \
 --disable-moe-prefill-overlap \
 --memory-ratio 0.95 \
 --num-tokens 64000  \
 --max-seq-len-override 64000 \
 --max-output-tokens 4096 \
 --max-prefill-length 4096 \
 --max-running-requests 1 \
 --cuda-graph-max-bs 1 \
 --cache-type radix \
 --enable-cache-report \
 --sampling-defaults model \
 --reasoning-parser auto \
 --host 0.0.0.0 \
 --port 8888 

