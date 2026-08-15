#!/bin/bash
# SOFT-SPEED-1 driver (pre-reg RESULTS 2026-08-15, commit be3b8e4):
# two serial Mac arms, control (stock recipe, must reproduce booked
# stock s3 bit-exactly) then soft (collapsed diet, matched epochs).
# Receipts: logs/softspeed1/arms.jsonl. Marker on SUCCESS only.
. "$(dirname "$0")/lib/driver.sh"
llmopt_cd
mkdir -p logs/softspeed1

for arm in control soft; do
  echo "[driver] arm=$arm start $(date +%H:%M:%S)"
  ARM=$arm SEED=3 .venv/bin/python scratch/birth19m_softspeed.py \
    2>&1 | tee logs/softspeed1/${arm}_s3.log
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "[driver] arm=$arm FAILED rc=$rc"; exit "$rc"
  fi
done
mark_done logs/softspeed1.DONE 0
echo "[driver] both arms done"
