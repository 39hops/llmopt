#!/bin/bash
# Rational-snap Q-sweep, Mac (pre-reg 2026-07-27): snap 19M crystal
# at Q in {4,16,64}, gate all four arms on the SAME MPS instrument.
set -e
cd ~/code/llmopt
for Q in 4 16 64; do
    .venv/bin/python scratch/rational_snap.py \
        checkpoints/mathnative_19m.pt "$Q" \
        "checkpoints/snap19m_q${Q}.pt" >> logs/snap_prep.log 2>&1
done
.venv/bin/python scratch/gate_ckpt.py checkpoints/mathnative_19m.pt \
    384 8 1536 6 snap_control_mps > logs/snap_control_gate.log 2>&1
for Q in 4 16 64; do
    .venv/bin/python scratch/gate_ckpt.py "checkpoints/snap19m_q${Q}.pt" \
        384 8 1536 6 "snap_q${Q}_mps" > "logs/snap_q${Q}_gate.log" 2>&1
done
touch logs/snap_sweep_done.marker
