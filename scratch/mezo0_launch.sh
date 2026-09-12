#!/usr/bin/env bash
# MEZO-SIGNAL-DESK-0 launcher: the registered desk under the liverun
# interlock (id mezo0), detached (nohup), stdout/stderr into the registered
# logs/mezo0/desk.log (refuses if it exists); writes logs/mezo0/mezo0.DONE on
# success only (a detached waiter can poll it). Usage: bash scratch/mezo0_launch.sh
set -eo pipefail
cd "$(dirname "$0")/.."
[ -e logs/mezo0/desk.log ] && { echo "REFUSING: logs/mezo0/desk.log exists"; exit 1; }
[ -e logs/mezo0/mezo0.DONE ] && { echo "REFUSING: stale logs/mezo0/mezo0.DONE"; exit 1; }
mkdir -p logs/mezo0
nohup bash -c '.venv/bin/python scripts/liverun.py run mezo0 -- .venv/bin/python scratch/mezo_signal_desk.py && echo DONE > logs/mezo0/mezo0.DONE' > logs/mezo0/desk.log 2>&1 &
echo "launched pid $!"
