#!/bin/bash
# Night 2026-07-28 (3080, Artin GO): audits -> greedy-first ->
# 6ep dose pair -> P2@45M. Jobs independent (failures logged,
# chain continues); markers success-only per job.
cd ~/code/llmopt
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 GRAD_CKPT=1

# A1: d2 endpoint verification (deps verified at arm time)
if [ -f checkpoints/metab_d2_fp64.pt ] && [ -f checkpoints/metab_d2_dd.pt ]; then
  $PY scratch/d2_verify.py > logs/n28_d2_verify.log 2>&1 \
    && touch logs/n28_d2_verify.marker
else
  echo "D2_CKPTS_MISSING" > logs/n28_d2_verify.log
fi

# A2: production crown re-baseline (drift audit)
$PY scratch/gate_ckpt_cuda.py checkpoints/mathnative_gen6_grown.pt \
  512 12 2304 8 CHAMPION-rebaseline > logs/n28_rebaseline.log 2>&1 \
  && touch logs/n28_rebaseline.marker

# D1: greedy-first adoption cell (same battery class)
$PY scratch/greedy_first_gate.py checkpoints/mathnative_gen6_grown.pt \
  512 12 2304 8 CHAMPION > logs/n28_gfirst.log 2>&1 \
  && touch logs/n28_gfirst.marker

# C1/C2: the 6ep dose pair (zero's born value at parity dose)
$PY scripts/tournament_birth.py --alpha S4 --epochs 6 --tag _6ep \
  > logs/n28_s4_6ep.log 2>&1 \
  && $PY scratch/gate_ckpt_cuda.py checkpoints/tourn_S4_6ep.pt \
     384 8 1536 6 S4-6ep >> logs/n28_s4_6ep.log 2>&1 \
  && touch logs/n28_s4_6ep.marker
$PY scripts/tournament_birth.py --alpha M4 --epochs 6 --tag _6ep \
  > logs/n28_m4_6ep.log 2>&1 \
  && $PY scratch/gate_ckpt_cuda.py checkpoints/tourn_M4_6ep.pt \
     384 8 1536 6 M4-6ep >> logs/n28_m4_6ep.log 2>&1 \
  && touch logs/n28_m4_6ep.marker

# B1: P2 at 45M-class (the alphabet winner at width), 6ep dose
$PY scripts/tournament_birth.py --alpha P2 --epochs 6 \
  --d 512 --layers 12 --ffn 2048 --heads 8 --tag _45m \
  > logs/n28_p2_45m.log 2>&1 \
  && $PY scratch/gate_ckpt_cuda.py checkpoints/tourn_P2_45m.pt \
     512 12 2048 8 P2-45m-6ep >> logs/n28_p2_45m.log 2>&1 \
  && touch logs/n28_p2_45m.marker

touch logs/night_28_done.marker
