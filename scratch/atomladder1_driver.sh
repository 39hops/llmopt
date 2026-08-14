#!/bin/bash
# ATOM-DIET-LADDER-1 driver (pre-reg RESULTS 2026-08-14): four serial
# arms on the Mac — (stock, atoms) x SEED {3,4} — each gated and its
# receipt streamed to logs/atomladder1/arms.jsonl as it lands.
. "$(dirname "$0")/lib/driver.sh"
llmopt_cd

mkdir -p logs/atomladder1
for SEED in 3 4; do
  for ARM in stock atoms; do
    echo "=== arm=$ARM seed=$SEED $(date +%H:%M:%S) ==="
    ARM=$ARM SEED=$SEED .venv/bin/python scratch/birth19m_atoms_ladder.py \
      2>&1 | tee "logs/atomladder1/${ARM}_s${SEED}.log"
  done
done
mark_done logs/atomladder1.DONE 0
