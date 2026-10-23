#!/bin/bash
# QWEN-LOSSY-GEOMETRY-0 chain (GO 2026-10-22), one process per stage, in
# order; a failed stage stops the chain with a FAILED line. Stages:
#   capture     scratch/qwen_geometry_capture.py (originals, arm C payloads,
#               canonical decode, activations of the 24 rollouts; receipt
#               logs/qwen_geometry/capture.json; GPU)
#   transform   scratch/qwen_geometry_transform.py (rotation cost on Metal;
#               receipt logs/qwen_geometry/transform.json; GPU)
#   dev         scratch/qwen_geometry_quant.py MODE=dev (candidates on the
#               development split; receipt logs/qwen_geometry/dev.json)
#   eval        MODE=eval (the registered comparison on the evaluation
#               split; receipt logs/qwen_geometry/eval.json)
#   adjudicate  scratch/qwen_geometry_obs.py then scripts/adjudicate.py
#               against docs/preregs/qwen-lossy-geometry-0.json
# ONLY="a b" runs a subset (the registration commit sits between dev and
# eval). Every receipt refuses to overwrite. Run under scripts/liverun.py.
set -uo pipefail
cd "$(dirname "$0")/.."
LOGDIR=logs/qwen_geometry
mkdir -p "$LOGDIR"
COMMIT=$(git rev-parse --short HEAD)
LOG=$LOGDIR/study_$COMMIT.log
PY=.venv/bin/python
ONLY=${ONLY:-}
run() {
  local name=$1; shift
  if [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " $name "; then echo "== skip $name (ONLY=$ONLY)" >> "$LOG"; return 0; fi
  echo "== $name $(date -u +%H:%M:%S) $COMMIT" >> "$LOG"
  TMP=$(mktemp)
  "$@" > "$TMP" 2>&1
  rc=$?
  grep -v Warning "$TMP" | grep -E "^\[" >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "== FAILED $name rc=$rc $(date -u +%H:%M:%S)" >> "$LOG"
    tail -20 "$TMP" >> "$LOG"
    rm -f "$TMP"
    exit "$rc"
  fi
  rm -f "$TMP"
}
run capture   $PY scratch/qwen_geometry_capture.py
run transform $PY scratch/qwen_geometry_transform.py
run dev       env MODE=dev $PY scratch/qwen_geometry_quant.py
run eval      env MODE=eval $PY scratch/qwen_geometry_quant.py
adjudicate() {
  if [ -e logs/qwen_geometry/observations.json ] || [ -e logs/qwen_geometry/adjudication.txt ]; then
    echo "REFUSING: observations or adjudication exists"; return 3; fi
  $PY scratch/qwen_geometry_obs.py logs/qwen_geometry/eval.json logs/qwen_geometry/transform.json \
      docs/preregs/qwen-lossy-geometry-0.json > logs/qwen_geometry/observations.json || return $?
  $PY scripts/adjudicate.py docs/preregs/qwen-lossy-geometry-0.json logs/qwen_geometry/observations.json \
      > logs/qwen_geometry/adjudication.txt
  rc=$?
  cat logs/qwen_geometry/adjudication.txt
  [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ]
}
run adjudicate adjudicate
echo "== DONE $(date -u +%H:%M:%S)" >> "$LOG"
