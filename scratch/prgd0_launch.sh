#!/usr/bin/env bash
# PERTURBATION-RESPONSE-GRAM-DESK-0 launcher: the single registered zero-training
# desk (writer A, the booked RDC1 C / R1 / R2 / R3 snapshots and the locked FMEL2
# eps = 1e-1 snapshots at the 12 registered horizons) under the liverun interlock
# (id prgd0), detached (nohup), stdout/stderr into logs/prgd0/desk.log (refuses if
# it exists); writes logs/prgd0/prgd0.DONE on success only. The 3600 s wall cap is
# enforced INSIDE the instrument (SIGALRM -> NOT-RUN receipt, exit 3), so a timeout
# never writes the DONE marker. Needs its own Artin GO. Usage: bash scratch/prgd0_launch.sh
set -eo pipefail
cd "$(dirname "$0")/.."
[ -e logs/prgd0/desk.log ] && { echo "REFUSING: logs/prgd0/desk.log exists"; exit 1; }
[ -e logs/prgd0/prgd0.DONE ] && { echo "REFUSING: stale logs/prgd0/prgd0.DONE"; exit 1; }
mkdir -p logs/prgd0
nohup bash -c '.venv/bin/python scripts/liverun.py run prgd0 -- .venv/bin/python scratch/perturbation_response_gram_desk.py && echo DONE > logs/prgd0/prgd0.DONE' > logs/prgd0/desk.log 2>&1 &
echo "launched prgd0 pid $!"
