#!/bin/bash
# S16 cached-decoding study chain (GO 2026-10-22), one process per stage,
# in order; a failed stage stops the chain with a FAILED line. Stages:
#   breakdown   scratch/qwen_metal_breakdown.py as frozen (the per-family
#               decode ablation on the current W4 v4 + S16 runtime; receipt
#               logs/qmetal/breakdown_<commit>.json)
#   s16_probe   scratch/qmetal_s16_probe.py (S16 lookup diagnostics and
#               candidates on real matrices and the head; receipt
#               logs/qmetal/s16_probe_<commit>.json)
#   ab          scratch/bench_qmetal_ab_s16.py (generation 4 against
#               generation 5, matrices and the full cached decode, ABBA
#               twice per context; receipt logs/qmetal/s16_v5/ab.json)
#   qual        scratch/qwen_metal_qual.py under QMETAL_VERSION=5 (the E0
#               census, cached decode and long context; receipts under
#               logs/qwen_metal_qual/C_v5/)
#   adjudicate  scratch/qmetal_s16_obs.py (observations from the two
#               receipts) then scripts/adjudicate.py against
#               docs/preregs/qwen-metal-s16-v5-0.json
# ONLY="a b" runs a subset. Every receipt refuses to overwrite. Run under
# scripts/liverun.py (clean tree, no commits while live). Timing stages
# must not share the GPU with anything else.
set -uo pipefail
cd "$(dirname "$0")/.."
LOGDIR=logs/qmetal/s16_study
mkdir -p "$LOGDIR"
LOG=$LOGDIR/study_$(git rev-parse --short HEAD).log
PY=.venv/bin/python
ONLY=${ONLY:-}
run() {
  local name=$1; shift
  if [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " $name "; then echo "== skip $name (ONLY=$ONLY)" >> "$LOG"; return 0; fi
  echo "== $name $(date -u +%H:%M:%S) $(git rev-parse --short HEAD)" >> "$LOG"
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
run breakdown  $PY scratch/qwen_metal_breakdown.py
run s16_probe  $PY scratch/qmetal_s16_probe.py
run ab         env OUT=logs/qmetal/s16_v5/ab.json $PY scratch/bench_qmetal_ab_s16.py
run qual       env QMETAL_VERSION=5 OUT=logs/qwen_metal_qual/C_v5 $PY scratch/qwen_metal_qual.py
adjudicate() {
  if [ -e logs/qmetal/s16_v5/observations.json ] || [ -e logs/qmetal/s16_v5/adjudication.txt ]; then
    echo "REFUSING: observations or adjudication exists"; return 3; fi
  $PY scratch/qmetal_s16_obs.py logs/qmetal/s16_v5/ab.json logs/qwen_metal_qual/C_v5/summary.json \
      docs/preregs/qwen-metal-s16-v5-0.json > logs/qmetal/s16_v5/observations.json || return $?
  $PY scripts/adjudicate.py docs/preregs/qwen-metal-s16-v5-0.json logs/qmetal/s16_v5/observations.json \
      > logs/qmetal/s16_v5/adjudication.txt
  rc=$?
  cat logs/qmetal/s16_v5/adjudication.txt
  # exit 2 (UNRESOLVED) is an honest outcome, booked as such; only a contract error stops the chain
  [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ]
}
run adjudicate adjudicate
echo "== DONE $(date -u +%H:%M:%S)" >> "$LOG"
