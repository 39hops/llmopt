#!/usr/bin/env bash
# OPTIMIZER-MEMORY-ABLATION-1 launcher: one registered mode under the liverun
# interlock, detached (nohup), stdout/stderr into logs/oma1/<mode>.log
# (refuses if it exists); writes logs/oma1/<id>.DONE on success only.
# Usage: bash scratch/oma1_launch.sh desk-bar1 | stage0 | stage1
# Each mode needs its own Artin GO (desk-bar1 = zero training; stage0 =
# native-state preconditions; stage1 = the Z / E legs).
set -eo pipefail
cd "$(dirname "$0")/.."
MODE="${1:?mode: desk-bar1 | stage0 | stage1}"
case "$MODE" in
  desk-bar1) ID=oma1bar1 ;;
  stage0)    ID=oma1s0 ;;
  stage1)    ID=oma1s1 ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
LOG="logs/oma1/${MODE}.log"
[ -e "$LOG" ] && { echo "REFUSING: $LOG exists"; exit 1; }
[ -e "logs/oma1/${ID}.DONE" ] && { echo "REFUSING: stale logs/oma1/${ID}.DONE"; exit 1; }
mkdir -p logs/oma1
nohup bash -c "MODE=$MODE .venv/bin/python scripts/liverun.py run $ID -- .venv/bin/python scratch/optimizer_memory_ablation.py && echo DONE > logs/oma1/${ID}.DONE" > "$LOG" 2>&1 &
echo "launched $MODE as $ID pid $!"
