#!/usr/bin/env bash
# RANDOM-DIRECTION-CONTROL-1 launcher: the single registered mode (control,
# writer A only, four fresh continuous CPU legs 7201..15420 (C + three seeded random
# exp_avg directions R1 / R2 / R3) against the locked FMEL2 eps = 1e-1 comparison
# vectors) under the liverun interlock (id rdc1), detached (nohup), stdout/stderr
# into logs/rdc1/control.log (refuses if it exists); writes logs/rdc1/rdc1.DONE on
# success only. The registered 7 h wall cap (25200 s, stop law (a)) is enforced
# INSIDE the instrument (SIGALRM -> NOT-RUN receipt, exit 3), so a timeout never
# writes the DONE marker. Needs its own Artin GO. Usage: bash scratch/rdc1_launch.sh
set -eo pipefail
cd "$(dirname "$0")/.."
[ -e logs/rdc1/control.log ] && { echo "REFUSING: logs/rdc1/control.log exists"; exit 1; }
[ -e logs/rdc1/rdc1.DONE ] && { echo "REFUSING: stale logs/rdc1/rdc1.DONE"; exit 1; }
mkdir -p logs/rdc1
nohup bash -c 'MODE=control .venv/bin/python scripts/liverun.py run rdc1 -- .venv/bin/python scratch/random_direction_control.py && echo DONE > logs/rdc1/rdc1.DONE' > logs/rdc1/control.log 2>&1 &
echo "launched rdc1 pid $!"
