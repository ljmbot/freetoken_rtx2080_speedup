#!/usr/bin/env bash
# Phase A: rebuild the standalone FreeToken venv from the recorded baseline.
# Re-runnable / idempotent: safe to re-run after an interruption.
# Installs, in order:
#   [1/3] torch 2.11.0+cu130 (from the cu130 index, so the CUDA libs match)
#   [2/3] the remaining 88 pinned baseline deps (requirements-baseline.txt)
#   [3/3] editable install of THIS project (compiles _cpu_moe + _pinned_tensor for 3.12)
set -uo pipefail

export PROJECT="$HOME/src/freetoken-sm75-native-prefill"
cd "$PROJECT"
source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="7.5"
export MAX_JOBS=4
export PYTHONDONTWRITEBYTECODE=1

echo "=== [1/3] torch 2.11.0 from cu130 index ==="
pip install --disable-pip-version-check "torch==2.11.0" \
    --index-url https://download.pytorch.org/whl/cu130
echo "torch exit: $?"

echo "=== [2/3] remaining baseline deps (88 pkgs) ==="
pip install --disable-pip-version-check -r requirements-baseline.txt
echo "deps exit: $?"

echo "=== [3/3] editable install (compiles 2 C extensions for py3.12) ==="
pip install --disable-pip-version-check --no-deps --no-build-isolation -e .
echo "editable exit: $?"

echo "=== verify ==="
python - <<'PY'
import sys, importlib.metadata as md
import torch
print("python:", sys.executable)
print("torch:", torch.__version__, "cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))
    print("arch_list:", torch.cuda.get_arch_list())
import freetoken
print("freetoken.__file__:", freetoken.__file__)
for n in ("triton","apache-tvm-ffi","transformers","flashlib","nvidia-cublas"):
    try: print(n, md.version(n))
    except Exception: print(n, "NOT INSTALLED")
PY
echo "pip check:"
pip check
echo "=== DONE ==="
