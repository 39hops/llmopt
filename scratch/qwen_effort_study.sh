#!/bin/bash
# QWEN-EFFORT-AND-DEEPCHAT-0 chain (GO 2026-10-23), one process per stage,
# in order; a failed stage stops the chain with a FAILED line.
#   template  tests/test_qtemplate.py (the pinned template's reasoning_effort
#             variants in fresh subprocesses) and the live serve tests
#   low       scratch/qwen_battery_run.py THINK=1 BUDGET=768 EFFORT=low
#             (receipt logs/qwen_battery/results_v2_think768_low.json)
#   medium    the same with EFFORT=medium (results_v2_think768_medium.json)
#   deepchat  scratch/qwen_deepchat_diag.py (receipt logs/qwen_serve/deepchat_v1.json)
# ONLY="a b" runs a subset. Every receipt refuses to overwrite. Run under
# scripts/liverun.py (clean tree, no commits while live).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/qwen_serve logs/qwen_battery
COMMIT=$(git rev-parse --short HEAD)
LOG=logs/qwen_serve/effort_$COMMIT.log
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
run template env LLMOPT_QWEN_LIVE=1 $PY -m pytest tests/test_qtemplate.py tests/test_qserve.py -q -p no:cacheprovider -k "live or template or effort or render"
run low      env THINK=1 BUDGET=768 EFFORT=low OUT=logs/qwen_battery/results_v2_think768_low.json $PY scratch/qwen_battery_run.py
run medium   env THINK=1 BUDGET=768 EFFORT=medium OUT=logs/qwen_battery/results_v2_think768_medium.json $PY scratch/qwen_battery_run.py
run deepchat $PY scratch/qwen_deepchat_diag.py
echo "== DONE $(date -u +%H:%M:%S)" >> "$LOG"
