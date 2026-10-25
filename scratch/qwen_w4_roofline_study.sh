#!/bin/bash
# QWEN-W4-PREFILL-ROOFLINE-0 chain (GO 2026-10-25: assessment and bounded
# measurement of the W4 prefill wall on the promoted kernel path), one
# process per stage, in order; a failed stage stops the chain with a FAILED
# line. Each stage of scratch/qwen_w4_roofline.py writes
# logs/qwen_w4_roofline/<stage>.json and refuses to overwrite. SMOKE=1 runs
# every stage shortened into logs/qwen_w4_roofline/smoke/. ONLY="a b" runs a
# subset. The real run goes under scripts/liverun.py (clean tree).
set -uo pipefail
cd "$(dirname "$0")/.."
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then DIR=logs/qwen_w4_roofline/smoke; else DIR=logs/qwen_w4_roofline; fi
mkdir -p "$DIR"
COMMIT=$(git rev-parse --short HEAD)
LOG=$DIR/study_$COMMIT.log
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
run decomp env SMOKE=$SMOKE STAGE=decomp $PY scratch/qwen_w4_roofline.py
run micro  env SMOKE=$SMOKE STAGE=micro  $PY scratch/qwen_w4_roofline.py
echo "== DONE $(date -u +%H:%M:%S)" >> "$LOG"
