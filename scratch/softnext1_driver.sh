#!/bin/bash
# SOFT-NEXT-1 driver (pre-reg RESULTS 2026-08-15): two serial arms
# on the 3080 — control (one-hot) then soft (branch-distribution
# targets) — each gated + calibration-probed, receipts streamed to
# logs/softnext1/arms.jsonl.
. "$(dirname "$0")/lib/driver.sh"
llmopt_cd
export TORCH_DISABLE_NATIVE_JIT=1

mkdir -p logs/softnext1
for ARM in control soft; do
  echo "=== arm=$ARM $(date +%H:%M:%S) ==="
  ARM=$ARM .venv/bin/python scratch/birth19m_softnext.py \
    2>&1 | tee "logs/softnext1/${ARM}_s2.log"
done
mark_done logs/softnext1.DONE 0
