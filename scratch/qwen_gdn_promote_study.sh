#!/bin/bash
# QWEN-GDN-PROMOTE-0 chain (GO 2026-10-25: the kernel as the serving default,
# conditional on every gate), one process per stage, in order; a failed stage
# stops the chain with a FAILED line. Each stage of scratch/qwen_gdn_promote.py
# writes logs/qwen_gdn_promote/<stage>.json and refuses to overwrite. SMOKE=1
# runs every stage shortened into logs/qwen_gdn_promote/smoke/. ONLY="a b" runs
# a subset. The real run goes under scripts/liverun.py (clean tree).
set -uo pipefail
cd "$(dirname "$0")/.."
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then DIR=logs/qwen_gdn_promote/smoke; else DIR=logs/qwen_gdn_promote; fi
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
  grep -v Warning "$TMP" | grep -E "^\[|passed|failed" >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "== FAILED $name rc=$rc $(date -u +%H:%M:%S)" >> "$LOG"
    tail -20 "$TMP" >> "$LOG"
    rm -f "$TMP"
    exit "$rc"
  fi
  rm -f "$TMP"
}
run tests    env SMOKE=$SMOKE STAGE=tests $PY scratch/qwen_gdn_promote.py
run heldout  env SMOKE=$SMOKE STAGE=heldout $PY scratch/qwen_gdn_promote.py
run cache    env SMOKE=$SMOKE STAGE=cache $PY scratch/qwen_gdn_promote.py
run battery  env SMOKE=$SMOKE STAGE=battery $PY scratch/qwen_gdn_promote.py
run cli      env SMOKE=$SMOKE STAGE=cli $PY scratch/qwen_gdn_promote.py
echo "== DONE $(date -u +%H:%M:%S)" >> "$LOG"
