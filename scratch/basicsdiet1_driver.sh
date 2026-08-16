#!/bin/bash
# BASICS-DIET-1 paired arms (pre-reg RESULTS L30355, amended L30429).
# Serial, same machine, same session: control then arith, seed 3.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/basicsdiet1
for ARM in control arith; do
  echo "=== $ARM $(date)"
  SEED=3 ARM_NAME=$ARM .venv/bin/python -u scratch/birth19m_arith.py \
    > logs/basicsdiet1/${ARM}_s3.log 2>&1
  rc=$?
  echo "=== $ARM rc=$rc $(date)"
  [ $rc -eq 0 ] || exit $rc
done
echo "=== BOTH ARMS DONE $(date)"
