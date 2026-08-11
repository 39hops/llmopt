#!/bin/bash
# SSM-STAR-1 driver, 3080. Pre-reg: RESULTS PRE-REG SSM-STAR-1.
# Two paired arms (SSM twin + attention twin), same device, fp32,
# BIRTH_SEED=0, budget 6144, 3ep, d64/L8/ffn256.
set -e
cd ~/code/llmopt
mkdir -p logs/ssm_star1

BIRTH_SEED=0 .venv/bin/python scratch/ssm_star.py train \
  checkpoints/ssmstar_ssm_d64.pt 3 6144 \
  > logs/ssm_star1/ssm_d64_birth.log 2>&1
BIRTH_SEED=0 .venv/bin/python scripts/train_mathnative.py \
  --v22 --gen4 --fp32 --epochs 3 --budget 6144 \
  --d 64 --layers 8 --ffn 256 --heads 4 \
  --out checkpoints/ssmstar_attn_d64.pt \
  > logs/ssm_star1/attn_d64_birth.log 2>&1

.venv/bin/python scratch/ssm_star.py gate \
  checkpoints/ssmstar_ssm_d64.pt ssmstar_ssm_d64 \
  > logs/ssm_star1/ssm_d64_gate.log 2>&1
grep -h "gate" logs/ssm_star1/ssm_d64_gate.log | tail -1
.venv/bin/python scratch/gate_ckpt.py checkpoints/ssmstar_attn_d64.pt \
  64 8 256 4 ssmstar_attn_d64 > logs/ssm_star1/attn_d64_gate.log 2>&1
grep -h "gate" logs/ssm_star1/attn_d64_gate.log | tail -1
echo SSM-STAR-1 COMPLETE
