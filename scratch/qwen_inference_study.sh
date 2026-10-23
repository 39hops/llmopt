#!/bin/bash
# QWEN-INFERENCE-BASELINE-0 chain (GO 2026-10-23), one process per stage,
# in order; a failed stage stops the chain with a FAILED line. Stages:
#   live      tests/test_qserve.py live tests on the real tower (one turn,
#             two turns, incremental equals replay), under LLMOPT_QWEN_LIVE=1
#   perf      scratch/qwen_serve_perf.py (cold load, first generation, 512 /
#             1,024 / 2,048-token prompts, repeated turns in both modes,
#             parity against the direct qualified path, tokenizer round trips;
#             receipt logs/qwen_serve/perf_<TAG>.json, TAG default v2; the
#             first chain wrote perf.json and results.json with the v1 battery)
#   battery   scratch/qwen_battery_run.py (the frozen battery scored ONCE;
#             receipt logs/qwen_battery/results_<TAG>.json)
# ONLY="a b" runs a subset. Every receipt refuses to overwrite. Run under
# scripts/liverun.py (clean tree, no commits while live).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/qwen_serve logs/qwen_battery
COMMIT=$(git rev-parse --short HEAD)
LOG=logs/qwen_serve/study_$COMMIT.log
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
run live    env LLMOPT_QWEN_LIVE=1 $PY -m pytest tests/test_qserve.py -q -p no:cacheprovider -k live
run perf    env OUT=logs/qwen_serve/perf_${TAG:-v2}.json $PY scratch/qwen_serve_perf.py
run battery env OUT=logs/qwen_battery/results_${TAG:-v2}.json $PY scratch/qwen_battery_run.py
echo "== DONE $(date -u +%H:%M:%S)" >> "$LOG"
