#!/bin/bash
# Build the fork into a fresh venv. Run from the repository root: glm53/setup_env.sh
# Needs CUDA 12.x with nvcc (CUDA_HOME), Python 3.12 and uv. Tested with torch 2.8.0+cu128 on RTX 3090 (sm_86).
set -e
cd "$(dirname "$0")/.."
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}
export PATH=$CUDA_HOME/bin:$PATH
[ -d venv ] || uv venv --python 3.12 venv
uv pip install --python venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python venv/bin/python -r requirements.txt
uv pip install --python venv/bin/python fastapi uvicorn transformers huggingface_hub ninja setuptools wheel jinja2 llguidance marisa_trie
venv/bin/python setup.py build_ext --inplace
uv pip install --python venv/bin/python --no-deps -e .
venv/bin/python -c "import torch, exllamav3; from exllamav3.ext import exllamav3_ext as e; print('OK', torch.__version__, torch.version.cuda)"
