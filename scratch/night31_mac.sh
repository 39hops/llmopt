#!/bin/bash
# THE DEPTH LADDER (pre-reg 2026-07-29 day): layers {4,8,12,16}
# at d56/f224 + params-matched 4x d80/f320. Mac/MPS, sequential.
set -e
cd ~/code/llmopt
PY=.venv/bin/python
for L in 4 8 12 16; do
  ARM=dense TAG=_mps_L${L} SEED=1 D=56 FFN=224 HEADS=4 \
    LAYERS=$L EMA=0.999 $PY scratch/sym_birth.py
done
ARM=dense TAG=_mps_L4d80 SEED=1 D=80 FFN=320 HEADS=4 LAYERS=4 \
  EMA=0.999 $PY scratch/sym_birth.py
echo NIGHT31-MAC-COMPLETE
