#!/bin/bash
# Born-rational seed-2 replication (pre-reg 2026-07-27 late).
set -e
cd ~/code/llmopt
BIRTH_SEED=2 .venv/bin/python scripts/train_mathnative.py \
    --gen4 --epochs 3 --out checkpoints/mathnative_19m_mac_fp32_s2.pt \
    > logs/rat_control_s2_birth.log 2>&1
.venv/bin/python scratch/gate_ckpt.py checkpoints/mathnative_19m_mac_fp32_s2.pt \
    384 8 1536 6 rat_control_s2_mps > logs/rat_control_s2_gate.log 2>&1
RAT_Q=6 BIRTH_SEED=2 .venv/bin/python scripts/train_mathnative.py \
    --gen4 --epochs 3 --out checkpoints/mathnative_19m_mac_ratq6_s2.pt \
    > logs/rat_q6_s2_birth.log 2>&1
grep -q "RAT_Q ACTIVE" logs/rat_q6_s2_birth.log || { echo "RAT flag never armed"; exit 1; }
.venv/bin/python scratch/rat_deploy.py checkpoints/mathnative_19m_mac_ratq6_s2.pt \
    6 checkpoints/mathnative_19m_mac_ratq6_s2_dep.pt >> logs/rat_q6_s2_birth.log 2>&1
.venv/bin/python scratch/gate_ckpt.py checkpoints/mathnative_19m_mac_ratq6_s2_dep.pt \
    384 8 1536 6 rat_q6_s2_dep_mps > logs/rat_q6_s2_dep_gate.log 2>&1
touch logs/night_rat_s2_done.marker
