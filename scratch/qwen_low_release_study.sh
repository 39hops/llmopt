#!/bin/bash
# QWEN-LOW-RELEASE-0 chain (GO 2026-10-23), one process per stage, in order;
# a failed stage stops the chain with a FAILED line.
#   tests   tests/test_qtemplate.py and the live serve tests on the real tower
#   A       scratch/qwen_battery_run.py on the v3 collection, no thinking, 384
#           (receipt logs/qwen_battery/results_v3_A.json)
#   B       no thinking, 768 (results_v3_B.json)
#   C       THINK=1 EFFORT=low BUDGET=768 (results_v3_C.json)
#   chat    scratch/qwen_lowchat.py (receipt logs/qwen_serve/lowchat_v1.json)
# ONLY="a b" runs a subset. Every receipt refuses to overwrite. Run under
# scripts/liverun.py (clean tree, no commits while live).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/qwen_serve logs/qwen_battery
COMMIT=$(git rev-parse --short HEAD)
LOG=logs/qwen_serve/lowrelease_$COMMIT.log
PY=.venv/bin/python
ONLY=${ONLY:-}
run() {
  local name=$1; shift
  if [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " $name "; then echo "== skip $name (ONLY=$ONLY)" >> "$LOG"; return 0; fi
  echo "== $name $(date -u +%H:%M:%S) $COMMIT" >> "$LOG"
  TMP=$(mktemp)
  "$@" > "$TMP" 2>&1
  rc=$?
  grep -v Warning "$TMP" | grep -E "^\[|passed|failed|skipped" >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "== FAILED $name rc=$rc $(date -u +%H:%M:%S)" >> "$LOG"
    tail -20 "$TMP" >> "$LOG"
    rm -f "$TMP"
    exit "$rc"
  fi
  rm -f "$TMP"
}
run tests env LLMOPT_QWEN_LIVE=1 $PY -m pytest tests/test_qtemplate.py tests/test_qserve.py -q -p no:cacheprovider -k "live or template or effort or render"
run A     env COLLECTION=qwen_battery_v3.json OUT=logs/qwen_battery/results_v3_A.json $PY scratch/qwen_battery_run.py
run B     env COLLECTION=qwen_battery_v3.json BUDGET=768 OUT=logs/qwen_battery/results_v3_B.json $PY scratch/qwen_battery_run.py
run C     env COLLECTION=qwen_battery_v3.json THINK=1 EFFORT=low BUDGET=768 OUT=logs/qwen_battery/results_v3_C.json $PY scratch/qwen_battery_run.py
run chat  $PY scratch/qwen_lowchat.py
echo "== DONE $(date -u +%H:%M:%S)" >> "$LOG"
