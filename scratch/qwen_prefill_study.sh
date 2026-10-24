#!/bin/bash
# QWEN-PREFILL-WALL-0 chain (GO 2026-10-23), one process per stage, in order;
# a failed stage stops the chain with a FAILED line. Each stage of
# scratch/qwen_prefill_wall.py writes logs/qwen_prefill/<stage>.json and
# refuses to overwrite. The held-out stage runs last and is read once.
# ONLY="a b" runs a subset. Run under scripts/liverun.py (clean tree).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/qwen_prefill
COMMIT=$(git rev-parse --short HEAD)
LOG=logs/qwen_prefill/study_$COMMIT.log
PY=.venv/bin/python
ONLY=${ONLY:-}
run() {
  local name=$1; shift
  if [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " $name "; then echo "== skip $name (ONLY=$ONLY)" >> "$LOG"; return 0; fi
  echo "== $name $(date -u +%H:%M:%S) $COMMIT" >> "$LOG"
  TMP=$(mktemp)
  "$@" > "$TMP" 2>&1
  rc=$?
  grep -v Warning "$TMP" | grep -E "^\[|passed|failed" >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "== FAILED $name rc=$rc $(date -u +%H:%M:%S)" >> "$LOG"
    tail -20 "$TMP" >> "$LOG"
    rm -f "$TMP"
    exit "$rc"
  fi
  rm -f "$TMP"
}
run tests    $PY -m pytest tests/test_qchat_summary.py tests/test_qdefaults.py -q -p no:cacheprovider
run baseline env STAGE=baseline $PY scratch/qwen_prefill_wall.py
run decomp   env STAGE=decomp $PY scratch/qwen_prefill_wall.py
run design   env STAGE=design $PY scratch/qwen_prefill_wall.py
run heldout  env STAGE=heldout $PY scratch/qwen_prefill_wall.py
echo "== DONE $(date -u +%H:%M:%S)" >> "$LOG"
