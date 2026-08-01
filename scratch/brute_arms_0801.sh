#!/bin/bash
# GRAVMOE-BRUTE arms (pre-reg 2026-08-01): sha regression gate, then
# the steps axis and the params axis. Mac, sequential.
set -u
PY=.venv/bin/python
echo "=== ARM BR-REG (regression: must reproduce G-RB1 1fcfd187) ==="
GATE=1 COND=1 QK=1 $PY scratch/detbwd_gravmoe.py 2>&1 | tee /tmp/brute_reg.$$ 
SHA=$(grep "FINAL trajectory sha" /tmp/brute_reg.$$ | awk '{print $NF}')
rm -f /tmp/brute_reg.$$
if [[ "$SHA" != 1fcfd187* ]]; then
  echo "REGRESSION FAIL: sha $SHA != 1fcfd187... — LEG VOID, aborting"
  exit 1
fi
echo "REGRESSION PASS: $SHA"
echo "=== ARM BR-S4 (STEPS=8000, same params) ==="
GATE=1 COND=1 QK=1 STEPS=8000 $PY scratch/detbwd_gravmoe.py 2>&1
echo "=== ARM BR-W4 (DIM=128 DHEAD=32 FFN=256, STEPS=2000) ==="
GATE=1 COND=1 QK=1 DIM=128 DHEAD=32 FFN=256 $PY scratch/detbwd_gravmoe.py 2>&1
echo "=== BRUTE ARMS COMPLETE ==="
