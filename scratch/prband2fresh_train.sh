#!/usr/bin/env bash
# RENDER-ATLAS-FRESH-SEED-0 execution: the four frozen seeds in
# frozen order, each = path-isolated smoke (skipped if its receipt
# exists) -> double-build init -> the ONE fixed 3-epoch paired
# production birth. No validation of any kind runs here. Marker on
# success only.
set -eo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for s in 21001 22001 23001 24001; do
  echo "=== seed $s $(date) ==="
  if [ ! -f logs/mathworld1/prband2fresh_smoke/smoke_s$s.json ]; then
    SVPFF_SEED=$s $PY scratch/mathworld1_svpfofresh.py
  fi
  SVPFF_SEED=$s SVPFF_MAKE_INIT=1 $PY scratch/mathworld1_svpfofresh.py
  SVPFF_SEED=$s SVPFF_PRODUCTION=1 $PY scratch/mathworld1_svpfofresh.py
done
echo DONE > logs/mathworld1/prband2fresh_train/ALL4.DONE
echo "=== ALL FOUR SEEDS COMPLETE $(date) ==="
