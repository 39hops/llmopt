#!/bin/bash
# Leg C: marginal-value ladder (spec: escalation-engine cell 2).
# Per-level dose cuts, cuda line. Comparator: cuda d64 EMA 58.7
# (n=3, night-29). 8 births: levels {1,2,3,5} x keep {25,50}%.
set -e
cd ~/code/llmopt
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_DISABLE_NATIVE_JIT=1
for lv in 1 2 3 5; do
  for cut in 0.25 0.50; do
    ARM=dense TAG=_cu_L${lv}c${cut/0./} SEED=1 D=64 FFN=256 \
      HEADS=4 CUTLV=$lv CUT=$cut EMA=0.999 \
      $PY scratch/sym_birth.py
  done
done
echo NIGHT31-COMPLETE
