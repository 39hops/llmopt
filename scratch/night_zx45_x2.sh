#!/bin/bash
# HARDENING-P4 row 2 (ZX 45M seed ladder): two fresh union_45m
# births (seeds 2, 3) + both gates each, envs verbatim from the
# frozen night_45m_union.sh. Pooled read vs the 19M ZX seed fence
# (pre-reg REVIVE... HARDENING-P4-2 in RESULTS, booked before
# launch). Logs land in logs/zx45/ (unique per seed).
set -e
cd ~/code/llmopt
[ -s data/union_math_zx.jsonl ] || { echo "missing union diet"; exit 1; }
mkdir -p logs/zx45
export VOCAB_EXTRA="in(,out(,Z(,X(,P(,H(,:"
export SEQ_CAP=1536
export TF32=1
export GRAD_CKPT=1
export BIRTH_BS=8

for S in 2 3; do
    SEED=$S .venv/bin/python scratch/rev4_zx45.py \
        > logs/zx45/birth_s$S.log 2>&1
    .venv/bin/python scratch/gate_ckpt_cuda.py \
        checkpoints/union_45m_s$S.pt 512 12 2048 8 \
        union_45m_s${S}_math > logs/zx45/math_gate_s$S.log 2>&1
    ZX_D=512 ZX_LAYERS=12 ZX_HEADS=8 ZX_FFN=2048 \
    .venv/bin/python scratch/gate_zx.py \
        checkpoints/union_45m_s$S.pt real union_45m_s${S}_zx \
        > logs/zx45/zx_gate_s$S.log 2>&1
done
echo ALL-DONE
