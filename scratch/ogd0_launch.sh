#!/usr/bin/env bash
# OPTIMIZER-GEOMETRY-DESK-0 launcher: the registered zero-training optimizer-
# geometry desk under the liverun interlock (id ogd0), detached (nohup),
# stdout/stderr into logs/ogd0/desk.log (refuses if it exists); writes
# logs/ogd0/ogd0.DONE on success only. Usage: bash scratch/ogd0_launch.sh
set -eo pipefail
cd "$(dirname "$0")/.."
[ -e logs/ogd0/desk.log ] && { echo "REFUSING: logs/ogd0/desk.log exists"; exit 1; }
[ -e logs/ogd0/ogd0.DONE ] && { echo "REFUSING: stale logs/ogd0/ogd0.DONE"; exit 1; }
mkdir -p logs/ogd0
nohup bash -c '.venv/bin/python scripts/liverun.py run ogd0 -- .venv/bin/python scratch/optimizer_geometry_desk.py && echo DONE > logs/ogd0/ogd0.DONE' > logs/ogd0/desk.log 2>&1 &
echo "launched pid $!"
