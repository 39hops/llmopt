#!/bin/bash
# XTERM-DIET-1 paired arms (pre-reg RESULTS L30565). Serial, same
# machine, same session: control then xterm, seed 3; probe follows
# each arm so partial results stream.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/xtermdiet1
for ARM in control xterm; do
  echo "=== birth $ARM $(date)"
  SEED=3 ARM_NAME=$ARM .venv/bin/python -u scratch/birth19m_xterm.py \
    > logs/xtermdiet1/${ARM}_s3.log 2>&1
  echo "=== probe $ARM $(date)"
  .venv/bin/python -u scratch/xterm_probe.py \
    checkpoints/gallery19m_xterm${ARM}_s3.pt \
    > logs/xtermdiet1/probe_${ARM}_s3.log 2>&1
  echo "=== $ARM done $(date)"
done
echo "=== BOTH ARMS + PROBES DONE $(date)"
