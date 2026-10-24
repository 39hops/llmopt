#!/bin/bash
# QWEN-GDN-KERNEL-0 chain (GO 2026-10-24, qualification not promotion), one
# process per stage, in order; a failed stage stops the chain with a FAILED
# line. Each stage of scratch/qwen_gdn_kernel_qual.py writes
# logs/qwen_gdn/<stage>.json and refuses to overwrite. SMOKE=1 runs every
# stage shortened into logs/qwen_gdn/smoke/. ONLY="a b" runs a subset. The
# real run goes under scripts/liverun.py (clean tree).
set -uo pipefail
cd "$(dirname "$0")/.."
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then DIR=logs/qwen_gdn/smoke; else DIR=logs/qwen_gdn; fi
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
run selector  $PY -m pytest tests/test_qgdn.py -q -p no:cacheprovider
if grep -q "skipped" "$LOG"; then echo "== FAILED selector tests skipped (no Metal device?)" >> "$LOG"; exit 3; fi
run tests     $PY -m pytest tests/test_qprefill.py tests/test_qserve.py -q -p no:cacheprovider
run layers    env SMOKE=$SMOKE STAGE=layers $PY scratch/qwen_gdn_kernel_qual.py
run census    env SMOKE=$SMOKE STAGE=census $PY scratch/qwen_gdn_kernel_qual.py
run decode    env SMOKE=$SMOKE STAGE=decode $PY scratch/qwen_gdn_kernel_qual.py
run heldout   env SMOKE=$SMOKE STAGE=heldout $PY scratch/qwen_gdn_kernel_qual.py
run cachehist env SMOKE=$SMOKE STAGE=cachehist $PY scratch/qwen_gdn_kernel_qual.py
echo "== DONE $(date -u +%H:%M:%S)" >> "$LOG"
