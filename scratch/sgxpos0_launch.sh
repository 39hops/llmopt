#!/usr/bin/env bash
# SG-CROSSPOS-REPRESENTABILITY-0 launcher: the registered desk under the
# liverun interlock, detached (nohup), stdout/stderr into the registered
# desk.log (refuses if it exists). Usage: bash scratch/sgxpos0_launch.sh
set -eo pipefail
cd "$(dirname "$0")/.."
[ -e logs/sgxpos0/desk.log ] && { echo "REFUSING: logs/sgxpos0/desk.log exists"; exit 1; }
mkdir -p logs/sgxpos0
nohup .venv/bin/python scripts/liverun.py run sgxpos0 -- .venv/bin/python scratch/sg_crosspos_desk.py > logs/sgxpos0/desk.log 2>&1 &
echo "launched pid $!"
