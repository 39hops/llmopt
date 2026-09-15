#!/usr/bin/env bash
# FIRST-MOMENT-ERASURE-LADDER-2 launcher: the single registered mode (ladder,
# writer A only, four fresh continuous CPU legs 7201..15420 (C + eps 1e-2 / 1e-1 / 1)) under the liverun
# interlock (id fmel2), detached (nohup), stdout/stderr into
# logs/fmel2/ladder.log (refuses if it exists); writes logs/fmel2/fmel2.DONE on
# success only. Needs its own Artin GO. Usage: bash scratch/fmel2_launch.sh
set -eo pipefail
cd "$(dirname "$0")/.."
[ -e logs/fmel2/ladder.log ] && { echo "REFUSING: logs/fmel2/ladder.log exists"; exit 1; }
[ -e logs/fmel2/fmel2.DONE ] && { echo "REFUSING: stale logs/fmel2/fmel2.DONE"; exit 1; }
mkdir -p logs/fmel2
nohup bash -c 'MODE=ladder .venv/bin/python scripts/liverun.py run fmel2 -- .venv/bin/python scratch/first_moment_erasure_ladder2.py && echo DONE > logs/fmel2/fmel2.DONE' > logs/fmel2/ladder.log 2>&1 &
echo "launched fmel2 pid $!"
