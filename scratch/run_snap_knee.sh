#!/bin/bash
# Knee localization (pre-reg 2026-07-27): Q in {24,32,48}, same
# instrument as the Q-sweep (control 49 already gated this session).
set -e
cd ~/code/llmopt
for Q in 24 32 48; do
    .venv/bin/python scratch/rational_snap.py \
        checkpoints/mathnative_19m.pt "$Q" \
        "checkpoints/snap19m_q${Q}.pt" >> logs/snap_prep.log 2>&1
    .venv/bin/python scratch/gate_ckpt.py "checkpoints/snap19m_q${Q}.pt" \
        384 8 1536 6 "snap_q${Q}_mps" > "logs/snap_q${Q}_gate.log" 2>&1
done
touch logs/snap_knee_done.marker
