#!/bin/bash
# 45M FEDERATION UNION (Artin GO 2026-07-27): does "union nearly
# free + grammar class does not bind" survive the 19M->45M scale
# rung? fp32 birth (doctrine), union diet, BOTH gates.
set -e
cd ~/code/llmopt
[ -s data/union_math_zx.jsonl ] || { echo "missing union diet"; exit 1; }
export VOCAB_EXTRA="in(,out(,Z(,X(,P(,H(,:"
export SEQ_CAP=1536
export TF32=1
export GRAD_CKPT=1
export BIRTH_BS=8

BIRTH_SEED=1 .venv/bin/python scripts/train_mathnative.py \
    --diet data/union_math_zx.jsonl --epochs 3 \
    --d 512 --layers 12 --ffn 2048 --heads 8 \
    --out checkpoints/union_45m.pt > logs/union_45m_birth.log 2>&1
.venv/bin/python scratch/gate_ckpt_cuda.py checkpoints/union_45m.pt \
    512 12 2048 8 union_45m_math > logs/union_45m_math_gate.log 2>&1
ZX_D=512 ZX_LAYERS=12 ZX_HEADS=8 ZX_FFN=2048 \
.venv/bin/python scratch/gate_zx.py checkpoints/union_45m.pt \
    real union_45m_zx > logs/union_45m_zx_gate.log 2>&1
