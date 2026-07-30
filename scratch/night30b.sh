#!/bin/bash
# NIGHT-30b 3080: chained behind night30's marker -> seed-3 births.
set -e
cd ~/code/llmopt
until [ -f logs/night30_done.marker ]; do sleep 60; done
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 TORCH_DISABLE_NATIVE_JIT=1
ARM=dense TAG=_s3_h8 SEED=3 D=64 FFN=256 HEADS=8 EMA=0.999 \
  .venv/bin/python scratch/sym_birth.py > logs/night30_h8s3.log 2>&1
ARM=dense TAG=_s3_L4 SEED=3 D=56 FFN=224 HEADS=4 LAYERS=4 EMA=0.999 \
  .venv/bin/python scratch/sym_birth.py > logs/night30_L4s3.log 2>&1
touch logs/night30b_done.marker
