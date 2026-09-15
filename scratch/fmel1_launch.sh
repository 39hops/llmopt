#!/usr/bin/env bash
# FIRST-MOMENT-ERASURE-LADDER-1 launcher: the single registered mode (ladder,
# writer A only, five fresh continuous CPU legs 7201..15420) under the liverun
# interlock (id fmel1), detached (nohup), stdout/stderr into
# logs/fmel1/ladder.log (refuses if it exists); writes logs/fmel1/fmel1.DONE on
# success only. Needs its own Artin GO. Usage: bash scratch/fmel1_launch.sh
set -eo pipefail
cd "$(dirname "$0")/.."
[ -e logs/fmel1/ladder.log ] && { echo "REFUSING: logs/fmel1/ladder.log exists"; exit 1; }
[ -e logs/fmel1/fmel1.DONE ] && { echo "REFUSING: stale logs/fmel1/fmel1.DONE"; exit 1; }
mkdir -p logs/fmel1
nohup bash -c 'MODE=ladder .venv/bin/python scripts/liverun.py run fmel1 -- .venv/bin/python scratch/first_moment_erasure_ladder.py && echo DONE > logs/fmel1/fmel1.DONE' > logs/fmel1/ladder.log 2>&1 &
echo "launched fmel1 pid $!"
