#!/bin/bash
# Born-rational paired arms, Mac overnight (pre-reg 2026-07-27):
# (A) fp32 control birth -> gate; (B) RAT_Q=6 STE birth -> deploy
# -> gate. Same recipe/diet/seed; markers on success only.
set -e
cd ~/code/llmopt
BIRTH_SEED=1 .venv/bin/python scripts/train_mathnative.py \
    --gen4 --epochs 3 --out checkpoints/mathnative_19m_mac_fp32.pt \
    > logs/rat_control_birth.log 2>&1
.venv/bin/python scratch/gate_ckpt.py checkpoints/mathnative_19m_mac_fp32.pt \
    384 8 1536 6 rat_control_mps > logs/rat_control_gate.log 2>&1
RAT_Q=6 BIRTH_SEED=1 .venv/bin/python scripts/train_mathnative.py \
    --gen4 --epochs 3 --out checkpoints/mathnative_19m_mac_ratq6.pt \
    > logs/rat_q6_birth.log 2>&1
grep -q "RAT_Q ACTIVE" logs/rat_q6_birth.log || { echo "RAT flag never armed"; exit 1; }
.venv/bin/python scratch/rat_deploy.py checkpoints/mathnative_19m_mac_ratq6.pt \
    6 checkpoints/mathnative_19m_mac_ratq6_dep.pt >> logs/rat_q6_birth.log 2>&1
.venv/bin/python scratch/gate_ckpt.py checkpoints/mathnative_19m_mac_ratq6_dep.pt \
    384 8 1536 6 rat_q6_dep_mps > logs/rat_q6_dep_gate.log 2>&1
.venv/bin/python scratch/gate_ckpt.py checkpoints/mathnative_19m_mac_ratq6.pt \
    384 8 1536 6 rat_q6_latent_mps > logs/rat_q6_latent_gate.log 2>&1
touch logs/night_rat_done.marker
