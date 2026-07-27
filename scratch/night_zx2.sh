#!/bin/bash
# 3080 night queue, job 3 (Artin GO 2026-07-26 ~11:12PM): the
# fp32-real ZX-ONLY control — the missing comparator that decides
# whether the union's ZX 40/120 is recipe (unquantized fp32-real)
# or math->ZX transfer. Same recipe as the union arm in every way
# except diet: d384/8L fp32, vocab-47, seed 1, 3 epochs, ZX rows
# only (zx_farm1_train, 97,036). Pre-reg in RESULTS 2026-07-27.
# WAITS for the running night queue's success marker; if the queue
# died (no marker, no python jobs for 3 consecutive polls), exits
# 1 so this job's marker honestly never fires.
cd ~/code/llmopt

dead=0
while [ ! -f logs/night_zx_done.marker ]; do
    if pgrep -f 'complex_birth.py|train_mathnative.py|gate_zx.py' > /dev/null; then
        dead=0
    else
        dead=$((dead + 1))
        [ "$dead" -ge 3 ] && { echo "night queue died without marker"; exit 1; }
    fi
    sleep 120
done

[ -f data/zx_farm1_train.jsonl ] || { echo "missing diet"; exit 1; }
export VOCAB_EXTRA="in(,out(,Z(,X(,P(,H(,:"
export SEQ_CAP=1536
export TF32=1
export GRAD_CKPT=1
export BIRTH_BS=8

BIRTH_SEED=1 .venv/bin/python scripts/train_mathnative.py \
    --diet data/zx_farm1_train.jsonl --epochs 3 \
    --out checkpoints/zx_real_fp32.pt > logs/zx_real_fp32_birth.log 2>&1 && \
.venv/bin/python scratch/gate_zx.py checkpoints/zx_real_fp32.pt \
    real zx_real_fp32 > logs/zx_real_fp32_gate.log 2>&1
