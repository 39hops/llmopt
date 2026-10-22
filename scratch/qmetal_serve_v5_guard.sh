#!/bin/bash
# Serving-path guard chain for the generation 5 default flip (GO 2026-10-22,
# items 1 and 3), one process per stage, in order; a failed stage stops
# the chain with a FAILED line. Stages:
#   ab          scratch/bench_qmetal_serve_v5.py MODE=ab (generation 4
#               against 5 through the serving configuration at contexts
#               1,024 and 2,048, ABBA twice; receipt
#               logs/qmetal/s16_v5_default/ab.json)
#   adjudicate  scratch/qmetal_serve_v5_obs.py then scripts/adjudicate.py
#               against docs/preregs/qwen-metal-s16-v5-default-0.json
#   release     after the flip: MODE=release in the process default
#               (QMETAL_VERSION unset), then MODE=release under
#               QMETAL_VERSION=4 (the control), then MODE=compare
#   profile     scratch/qwen_metal_breakdown.py as frozen under the new
#               default (receipt logs/qmetal/breakdown_<commit>.json)
# ONLY="a b" runs a subset. Every receipt refuses to overwrite. Run under
# scripts/liverun.py (clean tree, no commits while live). Timing stages
# must not share the GPU with anything else.
set -uo pipefail
cd "$(dirname "$0")/.."
LOGDIR=logs/qmetal/s16_v5_default
mkdir -p "$LOGDIR"
LOG=$LOGDIR/guard_$(git rev-parse --short HEAD).log
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
run ab env MODE=ab $PY scratch/bench_qmetal_serve_v5.py
adjudicate() {
  if [ -e logs/qmetal/s16_v5_default/observations.json ] || [ -e logs/qmetal/s16_v5_default/adjudication.txt ]; then
    echo "REFUSING: observations or adjudication exists"; return 3; fi
  $PY scratch/qmetal_serve_v5_obs.py logs/qmetal/s16_v5_default/ab.json \
      docs/preregs/qwen-metal-s16-v5-default-0.json > logs/qmetal/s16_v5_default/observations.json || return $?
  $PY scripts/adjudicate.py docs/preregs/qwen-metal-s16-v5-default-0.json logs/qmetal/s16_v5_default/observations.json \
      > logs/qmetal/s16_v5_default/adjudication.txt
  rc=$?
  cat logs/qmetal/s16_v5_default/adjudication.txt
  # exit 2 (UNRESOLVED) is an honest outcome, booked as such; only a contract error stops the chain
  [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ]
}
run adjudicate adjudicate
release() {
  if [ -n "${QMETAL_VERSION:-}" ]; then echo "REFUSING: QMETAL_VERSION is set in the release environment"; return 3; fi
  env -u QMETAL_VERSION MODE=release $PY scratch/bench_qmetal_serve_v5.py || return $?
  env QMETAL_VERSION=4 MODE=release $PY scratch/bench_qmetal_serve_v5.py || return $?
  local def
  def=$(ls logs/qmetal/s16_v5_default/release_v*_default.json)
  env MODE=compare A=logs/qmetal/s16_v5_default/release_v4_env.json B="$def" $PY scratch/bench_qmetal_serve_v5.py
}
run release release
run profile env -u QMETAL_VERSION $PY scratch/qwen_metal_breakdown.py
echo "== DONE $(date -u +%H:%M:%S)" >> "$LOG"
