#!/usr/bin/env bash
# NIGHT-31b cuda queue [HOLD — launch only on Artin's GO]:
# the evening remainder moved off the box to free it (2026-07-31):
#   1. lambda-merge reviews (pre-reg: review-adoption amendment #6)
#   2. R0 TF32-off lb ladder n=3 (pre-reg: deterministic-birth spec)
# Launch: scratch/wsl.sh launch "bash scratch/night31b_cuda.sh" \
#           "logs/night31b.log" "logs/night31b.DONE"
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || PY=python
echo "=== $(date '+%F %T') lambda-merge reviews ==="
"$PY" scratch/lam_merge_review.py
for s in 1 2 3; do
  echo "=== $(date '+%F %T') R0 TF32OFF lb seed $s ==="
  TF32OFF=1 ARM=lb SEED=$s OTAG=_cuda_tf32off \
    "$PY" scratch/umoe_conserve.py
done
echo "=== $(date '+%F %T') NIGHT-31b complete ==="
