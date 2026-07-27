#!/bin/bash
# Quick exact battery on union_45m.pt (pre-reg 2026-07-27 late).
# Comparator: tonight's booked control gate 65/120, SAME device/env.
set -e
cd ~/code/llmopt
export VOCAB_EXTRA="in(,out(,Z(,X(,P(,H(,:"
export D=512 LAYERS=12 FFN=2048 HEADS=8
G="checkpoints/union_45m.pt"
[ -s "$G" ] || { echo "missing $G"; exit 1; }
.venv/bin/python scratch/rational_snap.py $G 64 checkpoints/u45_rat64.pt > logs/qx_prep.log 2>&1
.venv/bin/python scratch/rational_snap.py $G 16 checkpoints/u45_rat16.pt >> logs/qx_prep.log 2>&1
.venv/bin/python scratch/fixed_q_snap.py $G 512 checkpoints/u45_fq512.pt >> logs/qx_prep.log 2>&1
.venv/bin/python scratch/fixed_q_snap.py $G 128 checkpoints/u45_fq128.pt >> logs/qx_prep.log 2>&1
for arm in rat64 fq512 fq128 rat16; do
    .venv/bin/python scratch/gate_ckpt_cuda.py "checkpoints/u45_${arm}.pt" \
        512 12 2048 8 "u45_${arm}" > "logs/qx_${arm}_gate.log" 2>&1
done
.venv/bin/python scratch/rat_repair.py checkpoints/u45_rat16.pt \
    data/union_math_zx.jsonl 400 checkpoints/u45_rat16_rep.pt \
    > logs/qx_repair.log 2>&1
.venv/bin/python scratch/gate_ckpt_cuda.py checkpoints/u45_rat16_rep.pt \
    512 12 2048 8 u45_rat16_rep > logs/qx_rat16_rep_gate.log 2>&1
touch logs/qx_done.marker
