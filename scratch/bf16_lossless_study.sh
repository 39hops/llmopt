#!/bin/bash
# Track B chain (LOSSLESS-BF16-FEASIBILITY-0, GO 2026-10-22), one process
# per stage, in order; a failed stage stops the chain with a FAILED line.
#   census      scratch/bf16_entropy_census.py (exact BF16 bit-pattern
#               entropies of the two retained shards; receipt
#               logs/bf16_lossless/entropy.json)
#   bench       scratch/bf16_lossless_bench.py (exact reversible encoders
#               with byte-equal round trips; receipt
#               logs/bf16_lossless/bench.json)
#   adjudicate  scratch/bf16_lossless_obs.py then scripts/adjudicate.py
#               against docs/preregs/lossless-bf16-feasibility-0.json
# ONLY="a b" runs a subset. Every receipt refuses to overwrite. Run under
# scripts/liverun.py (clean tree, no commits while live). CPU and memory
# bound; never alongside a GPU timing stage on this machine (unified
# memory bandwidth is shared).
set -uo pipefail
cd "$(dirname "$0")/.."
LOGDIR=logs/bf16_lossless
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
run census $PY scratch/bf16_entropy_census.py
run bench  $PY scratch/bf16_lossless_bench.py
adjudicate() {
  if [ -e logs/bf16_lossless/observations.json ] || [ -e logs/bf16_lossless/adjudication.txt ]; then
    echo "REFUSING: observations or adjudication exists"; return 3; fi
  $PY scratch/bf16_lossless_obs.py logs/bf16_lossless/entropy.json logs/bf16_lossless/bench.json \
      docs/preregs/lossless-bf16-feasibility-0.json > logs/bf16_lossless/observations.json || return $?
  $PY scripts/adjudicate.py docs/preregs/lossless-bf16-feasibility-0.json logs/bf16_lossless/observations.json \
      > logs/bf16_lossless/adjudication.txt
  rc=$?
  cat logs/bf16_lossless/adjudication.txt
  [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ]
}
run adjudicate adjudicate
echo "== DONE $(date -u +%H:%M:%S)" >> "$LOG"
