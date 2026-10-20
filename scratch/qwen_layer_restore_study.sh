#!/bin/bash
# QWEN-LAYER-RESTORE-0 chain, one process per stage, in order; a failed
# stage stops the chain with a FAILED line. Stages:
#   preflight, capture, oracle_mlx_armc, oracle_torch_armc   (no vendor bytes)
#   fetch, oracle_mlx_vendor, oracle_torch_vendor            (FETCH=1 only)
#   score x 5 arms (control and group with CAPTURE=1), summarize,
#   sensitivity, adjudicate (an observation under every class, so it
#   runs before the adjudicator can halt the chain)          (SCORE=1 only)
# START=<stage> resumes from a stage; ONLY="a b" runs a subset. SMOKE=1
# writes under logs/qwen_layer_restore_smoke with the stand-in weights and
# runs every stage. Run real stages under scripts/liverun.py.
set -uo pipefail
cd "$(dirname "$0")/.."
if [ "${SMOKE:-0}" = "1" ]; then OUTDIR=logs/qwen_layer_restore_smoke; else OUTDIR=${OUT:-logs/qwen_layer_restore}; fi
export OUT="$OUTDIR"
LOG=$OUTDIR/study.log
mkdir -p "$OUTDIR"
PY=.venv/bin/python
PYT=.venv_teacher/bin/python
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
run preflight         env MODE=preflight  $PY  scratch/qwen_layer_restore.py
run capture           env MODE=capture    $PY  scratch/qwen_layer_restore.py
run oracle_mlx_armc   env MODE=oracle_mlx WEIGHTS=armc $PY scratch/qwen_layer_restore.py
run oracle_torch_armc env WEIGHTS=armc    $PYT scratch/qwen_layer_restore_torch.py
if [ "${FETCH:-0}" = "1" ] || [ "${SMOKE:-0}" = "1" ]; then
  run fetch             env MODE=fetch      $PY  scratch/qwen_layer_restore.py
  run oracle_mlx_vendor env MODE=oracle_mlx WEIGHTS=vendor $PY scratch/qwen_layer_restore.py
  run oracle_torch_vendor env WEIGHTS=vendor $PYT scratch/qwen_layer_restore_torch.py
fi
if [ "${SCORE:-0}" = "1" ] || [ "${SMOKE:-0}" = "1" ]; then
  run score_control   env MODE=score ARM=control CAPTURE=1 $PY scratch/qwen_layer_restore.py
  run score_dense_c   env MODE=score ARM=dense_c           $PY scratch/qwen_layer_restore.py
  run score_l22       env MODE=score ARM=l22               $PY scratch/qwen_layer_restore.py
  run score_l23       env MODE=score ARM=l23               $PY scratch/qwen_layer_restore.py
  run score_group     env MODE=score ARM=group CAPTURE=1   $PY scratch/qwen_layer_restore.py
  run summarize       env MODE=summarize $PY scratch/qwen_layer_restore.py
  adjudicate() {
    if [ -e "$OUTDIR/adjudication.txt" ]; then echo "REFUSING: $OUTDIR/adjudication.txt exists"; return 3; fi
    $PY scripts/adjudicate.py docs/preregs/qwen-layer-restore-0.json "$OUTDIR/observations.json" > "$OUTDIR/adjudication.txt"
    local rc=$?
    cat "$OUTDIR/adjudication.txt"
    if [ "${SMOKE:-0}" = "1" ] && [ "$rc" -eq 2 ]; then echo "[smoke] UNRESOLVED adjudication (stand-in weights are inadmissible): expected"; return 0; fi
    return $rc
  }
  run sensitivity     env MODE=sensitivity $PY scratch/qwen_layer_restore.py
  run adjudicate      adjudicate
fi
echo "== done $(date -u +%H:%M:%S)" >> "$LOG"
