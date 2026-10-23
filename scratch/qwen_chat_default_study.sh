#!/bin/bash
# QWEN-CHAT-DEFAULT-AND-THINKING-0 chain (GO 2026-10-23), one process per
# stage, in order; a failed stage stops the chain with a FAILED line.
#   live      tests/test_qserve.py live tests on the real tower (one turn, two
#             turns, incremental equals replay, thinking on in both modes, a
#             budget stop then an incremental follow-up), LLMOPT_QWEN_LIVE=1
#   incr      scratch/qwen_incr_perf.py: five conversations in replay and
#             incremental mode (receipt logs/qwen_serve/incr_v1.json)
#   think384  scratch/qwen_battery_run.py THINK=1 BUDGET=384
#             (receipt logs/qwen_battery/results_v2_think384.json)
#   think768  scratch/qwen_battery_run.py THINK=1 BUDGET=768
#             (receipt logs/qwen_battery/results_v2_think768.json)
# ONLY="a b" runs a subset. Every receipt refuses to overwrite. Run under
# scripts/liverun.py (clean tree, no commits while live).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/qwen_serve logs/qwen_battery
COMMIT=$(git rev-parse --short HEAD)
LOG=logs/qwen_serve/chatdefault_$COMMIT.log
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
run live     env LLMOPT_QWEN_LIVE=1 $PY -m pytest tests/test_qserve.py -q -p no:cacheprovider -k live
run incr     $PY scratch/qwen_incr_perf.py
run think384 env THINK=1 BUDGET=384 OUT=logs/qwen_battery/results_v2_think384.json $PY scratch/qwen_battery_run.py
run think768 env THINK=1 BUDGET=768 OUT=logs/qwen_battery/results_v2_think768.json $PY scratch/qwen_battery_run.py
echo "== DONE $(date -u +%H:%M:%S)" >> "$LOG"
