#!/bin/bash
# GRAVMOE-BRUTE-B arms (pre-reg 2026-08-01 night): waits for the
# first brute leg's job to exit, re-runs the default regression
# (post ACLAMP/SCHED knobs), then the decay and corrected-width arms.
set -u
PY=.venv/bin/python
until [[ -f jobs/brute0801.rc ]]; do sleep 30; done
echo "=== ARM BR-REG2 (post-knob regression: must reproduce 1fcfd187) ==="
GATE=1 COND=1 QK=1 $PY scratch/detbwd_gravmoe.py 2>&1 | tee logs/brute_reg2.tmp
SHA=$(grep "FINAL trajectory sha" logs/brute_reg2.tmp | awk '{print $NF}')
rm -f logs/brute_reg2.tmp
if [[ "$SHA" != 1fcfd187* ]]; then
  echo "REGRESSION FAIL: sha $SHA — LEG VOID, aborting"; exit 1
fi
echo "REGRESSION PASS: $SHA"
echo "=== ARM BR-S4D (STEPS=8000 SCHED=1) ==="
GATE=1 COND=1 QK=1 STEPS=8000 SCHED=1 $PY scratch/detbwd_gravmoe.py 2>&1
echo "=== ARM BR-W4b (DIM=128 DHEAD=32 FFN=256 ACLAMP=32768) ==="
GATE=1 COND=1 QK=1 DIM=128 DHEAD=32 FFN=256 ACLAMP=32768 $PY scratch/detbwd_gravmoe.py 2>&1
echo "=== BRUTE-B ARMS COMPLETE ==="
