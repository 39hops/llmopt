#!/bin/bash
# QWEN-HEAD-RESTORE-0 chain: preflight, opener, corpus, fetch, score,
# adjudicate (runtime only on an explicit RUNTIME=1 after the validation
# fires), one process per stage, in order; a failed stage stops the chain
# with a FAILED line. START=<stage> resumes from a stage; ONLY="a b" runs
# a subset. Run under scripts/liverun.py.
set -uo pipefail
cd "$(dirname "$0")/.."
if [ "${SMOKE:-0}" = "1" ]; then OUTDIR=logs/qwen_head_restore_smoke; else OUTDIR=${OUT:-logs/qwen_head_restore}; fi
export OUT="$OUTDIR"
LOG=$OUTDIR/study.log
mkdir -p "$OUTDIR"
PY=.venv/bin/python
START=${START:-}
ONLY=${ONLY:-}
started=0
run() {
  local name=$1; shift
  if [ -n "$START" ] && [ "$started" = 0 ]; then
    if [ "$name" = "$START" ]; then started=1; else echo "== skip $name (START=$START)" >> "$LOG"; return 0; fi
  fi
  if [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " $name "; then echo "== skip $name (ONLY=$ONLY)" >> "$LOG"; return 0; fi
  echo "== $name $(date -u +%H:%M:%S) $(git rev-parse --short HEAD)" >> "$LOG"
  TMP=$(mktemp)
  "$@" > "$TMP" 2>&1
  rc=$?
  grep -v Warning "$TMP" | grep -E "^\[|^BAR|^REFUTED" >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "== FAILED $name rc=$rc $(date -u +%H:%M:%S)" >> "$LOG"
    tail -20 "$TMP" >> "$LOG"
    rm -f "$TMP"
    exit "$rc"
  fi
  rm -f "$TMP"
}
run preflight   env MODE=preflight $PY scratch/qwen_head_restore.py
run opener      env MODE=opener    $PY scratch/qwen_head_restore.py
run corpus      env MODE=corpus    $PY scratch/qwen_head_restore.py
run fetch       env MODE=fetch     $PY scratch/qwen_head_restore.py
run score       env MODE=score     $PY scratch/qwen_head_restore.py
adjudicate() {
  if [ -e "$OUTDIR/adjudication.txt" ]; then echo "REFUSING: $OUTDIR/adjudication.txt exists"; return 3; fi
  $PY scripts/adjudicate.py docs/preregs/qwen-head-restore-0.json "$OUTDIR/observations.json" > "$OUTDIR/adjudication.txt"
  local rc=$?
  cat "$OUTDIR/adjudication.txt"
  return $rc
}
run adjudicate  adjudicate
if [ "${RUNTIME:-0}" = "1" ]; then
  run runtime   env MODE=runtime   $PY scratch/qwen_head_restore.py
fi
echo "== done $(date -u +%H:%M:%S)" >> "$LOG"
