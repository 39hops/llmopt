#!/bin/bash
# Prefill performance study, one stage after another, each stage its own
# process so allocator state never carries between measurements:
#   1 kernel study (fused tiles, tile-8 ablation, decode-once alternatives)
#   2 non-matrix floor of a 256-token chunk (tower with stand-ins)
#   3 final-position head: three arms at 1,024 tokens, parity, isolated head
#   4 per-process memory for full / final / none at 1,024 and 4,096 tokens
#   5 4,096-token chunk-size comparison (128, 256, 512 full; 256 final)
# A failed stage stops the study with its exit code and a FAILED line; the
# "study done" marker is written only when every stage exited 0.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/qmetal/prefill_head/study.log
mkdir -p logs/qmetal/prefill_head
run_stage() {
  local label=$1; shift
  echo "== $label $(date -u +%H:%M:%S)" >> "$LOG"
  TMP=$(mktemp)
  "$@" > "$TMP" 2>&1
  local rc=$?
  grep -v Warning "$TMP" | grep "^\[" >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    grep -v Warning "$TMP" | tail -12 >> "$LOG"
    echo "== study FAILED at $label rc=$rc $(date -u +%H:%M:%S)" >> "$LOG"
    rm -f "$TMP"
    exit "$rc"
  fi
  rm -f "$TMP"
}
# START=<label prefix> skips the stages before it (a rerun after a failed
# stage); ONLY="<prefix> <prefix> ..." runs just the stages whose label
# starts with one of the prefixes
START=${START:-kernels}
ONLY=${ONLY:-}
started=0
run() {
  if [ "$started" -eq 0 ]; then
    case "$1" in "$START"*) started=1;; *) echo "== skip $1 (START=$START)" >> "$LOG"; return 0;; esac
  fi
  if [ -n "$ONLY" ]; then
    local hit=0 pfx
    for pfx in $ONLY; do case "$1" in "$pfx"*) hit=1;; esac; done
    if [ "$hit" -eq 0 ]; then echo "== skip $1 (ONLY=$ONLY)" >> "$LOG"; return 0; fi
  fi
  run_stage "$@"
}
PY=.venv/bin/python
run kernels env MODE=kernels $PY scratch/qmetal_prefill_kernel_study.py
run floor env MODE=floor CHUNK=256 ROUNDS=3 $PY scratch/qmetal_prefill_kernel_study.py
run head_time env MODE=time CTX=1024 CHUNK=256 ROUNDS=2 $PY scratch/qwen_metal_prefill_head.py
for cfg in "1024 full" "1024 final" "1024 none" "4096 full" "4096 final"; do
  set -- $cfg
  run "mem ctx $1 head $2" env MODE=mem CTX=$1 CHUNK=256 HEAD=$2 $PY scratch/qwen_metal_prefill_head.py
done
run ctx4k env MODE=ctx4k CTX=4096 $PY scratch/qwen_metal_prefill_head.py
echo "== study done $(date -u +%H:%M:%S)" >> "$LOG"
