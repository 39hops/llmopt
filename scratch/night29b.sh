#!/bin/bash
# night-29b: ffn cliff at d56 (pre-reg 2026-07-29 overnight)
set -e
cd ~/code/llmopt
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_DISABLE_NATIVE_JIT=1
for f in 96 64 48; do
  ARM=dense TAG=_cu56_f$f D=56 FFN=$f HEADS=4 EMA=0.999 \
    $PY scratch/sym_birth.py
done
echo NIGHT29B-COMPLETE
