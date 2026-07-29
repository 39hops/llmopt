#!/bin/bash
# reverse-pairs controls on the cuda line (spec: slack-restoration
# cell 2). Comparators: cuda d64 n=3 (raw 51.3 / EMA 58.7).
set -e
cd ~/code/llmopt
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_DISABLE_NATIVE_JIT=1
ARM=dense TAG=_cu_half SEED=1 D=64 FFN=256 HEADS=4 HALF=1 EMA=0.999 \
  $PY scratch/sym_birth.py
ARM=dense TAG=_cu_rev SEED=1 D=64 FFN=256 HEADS=4 REV=1 EMA=0.999 \
  $PY scratch/sym_birth.py
echo NIGHT30-COMPLETE
