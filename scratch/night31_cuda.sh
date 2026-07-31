#!/usr/bin/env bash
# NIGHT-31 cuda chain (pre-reg RUNG-2/3-CUDA): internally paired
# scaffold family + gravmoe lambda-sweep, ALL on the 3080, seed 1,
# OTAG=_cuda namespace. Never compare these gates to Mac numbers.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || PY=python
run() {
  echo "=== $(date '+%F %T') ARM=$1 ${2:-} ==="
  # NOTE friendly-fire #10: ${2:+GRAV_LAM=$2} expanded AFTER the
  # shell parses assignment prefixes -> "command not found". env(1)
  # accepts assignments as arguments, so this form is expansion-safe.
  ARM="$1" SEED=1 OTAG=_cuda env GRAV_LAM="${2:-0.5}" \
    "$PY" scratch/umoe_conserve.py
}
run lb
run tree
run treegrav
run gravmoe
run channel
run chantree
run gravmoe 0.1
run gravmoe 0.25
run gravmoe 1.0
echo "=== $(date '+%F %T') NIGHT-31 chain complete ==="
