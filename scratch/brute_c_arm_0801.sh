#!/bin/bash
# BR-W4c (pre-reg GRAVMOE-BRUTE-C): clamp-matched width arm.
set -u
until [[ -f jobs/bruteb0801.rc ]]; do sleep 30; done
echo "=== ARM BR-W4c (DIM=128 DHEAD=32 FFN=256 ACLAMP=49152) ==="
GATE=1 COND=1 QK=1 DIM=128 DHEAD=32 FFN=256 ACLAMP=49152 \
  .venv/bin/python scratch/detbwd_gravmoe.py 2>&1
echo "=== BRUTE-C ARM COMPLETE ==="
