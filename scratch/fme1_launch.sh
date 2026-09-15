#!/usr/bin/env bash
# FIRST-MOMENT-ERASURE-1 launcher: the single registered mode (treat, writer A
# only) under the liverun interlock (id fme1), detached (nohup), stdout/stderr
# into logs/fme1/treat.log (refuses if it exists); writes logs/fme1/fme1.DONE
# on success only. Needs its own Artin GO. Usage: bash scratch/fme1_launch.sh
set -eo pipefail
cd "$(dirname "$0")/.."
[ -e logs/fme1/treat.log ] && { echo "REFUSING: logs/fme1/treat.log exists"; exit 1; }
[ -e logs/fme1/fme1.DONE ] && { echo "REFUSING: stale logs/fme1/fme1.DONE"; exit 1; }
mkdir -p logs/fme1
nohup bash -c 'MODE=treat .venv/bin/python scripts/liverun.py run fme1 -- .venv/bin/python scratch/first_moment_erasure.py && echo DONE > logs/fme1/fme1.DONE' > logs/fme1/treat.log 2>&1 &
echo "launched fme1 pid $!"
