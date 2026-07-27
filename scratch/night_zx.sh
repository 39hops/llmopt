#!/bin/bash
# 3080 night queue (specs/2026-07-26-next-session-2.md, fired on
# Artin's GO 2026-07-26 night). Two jobs, self-marking via wsl.sh
# launch (marker on success only):
#   1. math+ZX union birth (vocab-47, gen-4 + zx_farm1_train, real
#      19M-class d384/8L, fp32, seed 1) + BOTH gates (math + ZX).
#   2. ZX seed-2 confirmation (cplx_none recipe, BIRTH_SEED=2).
cd ~/code/llmopt
export VOCAB_EXTRA="in(,out(,Z(,X(,P(,H(,:"
export SEQ_CAP=1536
export TF32=1
export GRAD_CKPT=1
export BIRTH_BS=8

.venv/bin/python scratch/make_union_diet.py > logs/union_diet.log 2>&1 && \
BIRTH_SEED=1 .venv/bin/python scripts/train_mathnative.py \
    --diet data/union_math_zx.jsonl --epochs 3 \
    --out checkpoints/union_math_zx.pt > logs/union_birth.log 2>&1 && \
.venv/bin/python scratch/gate_ckpt_cuda.py checkpoints/union_math_zx.pt \
    384 8 1536 6 union_math > logs/union_math_gate.log 2>&1 && \
.venv/bin/python scratch/gate_zx.py checkpoints/union_math_zx.pt \
    real union_zx > logs/union_zx_gate.log 2>&1 && \
BIRTH_SEED=2 CPLX_ALPHA=none .venv/bin/python scratch/complex_birth.py \
    --epochs 3 --tag _zx_s2 --diet data/zx_farm1_train.jsonl \
    > logs/zx_none_s2_birth.log 2>&1 && \
.venv/bin/python scratch/gate_zx.py checkpoints/cplx_none_zx_s2.pt \
    cplx none_zx_s2 > logs/zx_none_s2_gate.log 2>&1
