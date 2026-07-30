#!/bin/bash
# NIGHT-30 (pre-reg 2026-07-30): C1 seed-2 replication births.
set -e
cd ~/code/llmopt
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 TORCH_DISABLE_NATIVE_JIT=1
ARM=dense TAG=_s2_h8 SEED=2 D=64 FFN=256 HEADS=8 EMA=0.999 \
  .venv/bin/python scratch/sym_birth.py > logs/night30_h8s2.log 2>&1
ARM=dense TAG=_s2_L4 SEED=2 D=56 FFN=224 HEADS=4 LAYERS=4 EMA=0.999 \
  .venv/bin/python scratch/sym_birth.py > logs/night30_L4s2.log 2>&1
touch logs/night30_done.marker
