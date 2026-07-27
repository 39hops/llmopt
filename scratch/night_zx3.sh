#!/bin/bash
# JOB 5 — ZX seed-3 (queued 2026-07-27 under Artin's open-queue
# GO): third sigma point on the cplx_none/zx_farm1 arm (36, 28,
# ?) — the seed-2 verdict's fence stands "until an n>2 sigma
# exists"; this is the cheapest way to firm it. Chained behind
# job 4 (SR birth); honest-death waiter, success-only marker.
set -e
cd ~/code/llmopt
dead=0
while [ ! -f logs/night_sr_done.marker ]; do
    if pgrep -f 'train_mathnativ[e]|gate_ckpt_cud[a]|night_s[r]' > /dev/null; then
        dead=0
    else
        dead=$((dead+1))
        [ "$dead" -ge 3 ] && { echo "job 4 died without marker"; exit 1; }
    fi
    sleep 120
done
[ -s data/zx_farm1_train.jsonl ] || { echo "missing diet"; exit 1; }
export VOCAB_EXTRA="in(,out(,Z(,X(,P(,H(,:"
export SEQ_CAP=1536
export TF32=1
export GRAD_CKPT=1
export BIRTH_BS=8

BIRTH_SEED=3 CPLX_ALPHA=none .venv/bin/python scratch/complex_birth.py \
    --epochs 3 --tag _zx_s3 --diet data/zx_farm1_train.jsonl \
    > logs/zx_none_s3_birth.log 2>&1
.venv/bin/python scratch/gate_zx.py checkpoints/cplx_none_zx_s3.pt \
    cplx none_zx_s3 > logs/zx_none_s3_gate.log 2>&1
