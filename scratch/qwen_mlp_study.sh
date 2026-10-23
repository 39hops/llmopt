#!/bin/bash
# QWEN-MLP-COUPLED-0 chain (GO 2026-10-23), one process per stage, in
# order; a failed stage stops the chain with a FAILED line. Stages:
#   capture     scratch/qwen_mlp_capture.py (the new collection's prefill
#               and free-running inputs of layer 22's MLP from the unchanged
#               arm C model; receipt logs/qwen_mlp/capture.json; GPU)
#   dev         scratch/qwen_mlp_coupled.py MODE=dev (the candidate fit on
#               the dev split and scored on it; receipt logs/qwen_mlp/dev.json)
#   eval        MODE=eval (the registered comparison on the eval split;
#               receipt logs/qwen_mlp/eval.json)
#   adjudicate  scratch/qwen_mlp_obs.py then scripts/adjudicate.py against
#               docs/preregs/qwen-mlp-coupled-0.json, into
#               logs/qwen_mlp/adjudicate_<ADJ_TAG>/ (default 1; a new run
#               belongs at a new path)
# ONLY="a b" runs a subset. Every receipt refuses to overwrite. Run under
# scripts/liverun.py.
set -uo pipefail
cd "$(dirname "$0")/.."
LOGDIR=logs/qwen_mlp
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
run capture $PY scratch/qwen_mlp_capture.py
run dev     env MODE=dev $PY scratch/qwen_mlp_coupled.py
run eval    env MODE=eval $PY scratch/qwen_mlp_coupled.py
adjudicate() {
  local d=logs/qwen_mlp/adjudicate_${ADJ_TAG:-1}
  if [ -e "$d/observations.json" ] || [ -e "$d/adjudication.txt" ]; then
    echo "REFUSING: $d exists"; return 3; fi
  mkdir -p "$d"
  $PY scratch/qwen_mlp_obs.py logs/qwen_mlp/eval.json logs/qwen_mlp/dev.json \
      docs/preregs/qwen-mlp-coupled-0.json > "$d/observations.json" || return $?
  $PY scripts/adjudicate.py docs/preregs/qwen-mlp-coupled-0.json "$d/observations.json" > "$d/adjudication.txt"
  rc=$?
  cat "$d/adjudication.txt"
  [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ]
}
run adjudicate adjudicate
echo "== DONE $(date -u +%H:%M:%S)" >> "$LOG"
