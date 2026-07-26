#!/bin/bash
cd ~/code/llmopt
until [ -f logs/v5s2_done.marker ]; do sleep 120; done
BIRTH_SEED=1 TF32=1 .venv/bin/python scripts/tournament_birth.py --alpha Z1S --epochs 3 --lr 1e-3 --tag _hot > logs/tourn_Z1S_hot_birth.log 2>&1
.venv/bin/python scratch/gate_ckpt.py checkpoints/tourn_Z1S_hot.pt 384 8 1536 6 tourn_Z1S_hot > logs/tourn_Z1S_hot_gate.log 2>&1
echo Z1SHOT_DONE > logs/z1s_hot_done.marker
