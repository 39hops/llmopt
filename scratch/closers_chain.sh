#!/bin/bash
set -e
cd ~/code/llmopt
for A in S4 Z1 Z1S; do
  BIRTH_SEED=1 TF32=1 .venv/bin/python scripts/tournament_birth.py --alpha $A --epochs 3 > logs/tourn_${A}_birth.log 2>&1
  .venv/bin/python scratch/gate_ckpt.py checkpoints/tourn_${A}.pt 384 8 1536 6 tourn_${A} > logs/tourn_${A}_gate.log 2>&1
done
echo CLOSERS_DONE > logs/closers_done.marker
