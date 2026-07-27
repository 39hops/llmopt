#!/bin/bash
# JOB 4 — SR-bf16 birth (Artin GO 2026-07-27, hold released):
# stochastic-rounding bf16 weight cast on the gen-4 std --fast
# recipe. One variable v the booked pair (RESULTS 2026-07-17:
# fp32 69/120 v bf16-RNE 66/120): SR_BF16=1, same data/seed/gate.
# WAITS on job 3's success marker; if job 3 died (no marker, no
# procs for 3 consecutive polls), exits 1 — marker never fires.
set -e
cd ~/code/llmopt
dead=0
while [ ! -f logs/night_zx2_launcher_done.marker ]; do
    if pgrep -f 'train_mathnative.py|gate_zx.py|night_zx2.sh' > /dev/null; then
        dead=0
    else
        dead=$((dead+1))
        [ "$dead" -ge 3 ] && { echo "job 3 died without marker"; exit 1; }
    fi
    sleep 120
done

for f in data/micromodel_gen4_sidecar.jsonl data/micromodel_algebra_shard0.jsonl; do
    [ -s "$f" ] || { echo "MISSING $f"; exit 1; }
done

SR_BF16=1 .venv/bin/python scripts/train_mathnative.py \
    --gen4 --fast --epochs 3 \
    --out checkpoints/mathnative_45m_gen4_sr.pt \
    > logs/sr_birth.log 2>&1
grep -q "SR_BF16 ACTIVE" logs/sr_birth.log || { echo "SR flag never armed"; exit 1; }
.venv/bin/python scratch/gate_ckpt_cuda.py \
    checkpoints/mathnative_45m_gen4_sr.pt 384 8 1536 6 45m_gen4_sr \
    > logs/sr_gate.log 2>&1
