#!/usr/bin/env bash
# UPDATE-GEOMETRY-CENSUS-0 launcher: the registered zero-training loss-gradient
# geometry census under the liverun interlock (id ugc0), detached (nohup),
# stdout/stderr into logs/ugc0/census.log (refuses if it exists); writes
# logs/ugc0/ugc0.DONE on success only. Usage: bash scratch/ugc0_launch.sh
set -eo pipefail
cd "$(dirname "$0")/.."
[ -e logs/ugc0/census.log ] && { echo "REFUSING: logs/ugc0/census.log exists"; exit 1; }
[ -e logs/ugc0/ugc0.DONE ] && { echo "REFUSING: stale logs/ugc0/ugc0.DONE"; exit 1; }
mkdir -p logs/ugc0
nohup bash -c '.venv/bin/python scripts/liverun.py run ugc0 -- .venv/bin/python scratch/update_geometry_census.py && echo DONE > logs/ugc0/ugc0.DONE' > logs/ugc0/census.log 2>&1 &
echo "launched pid $!"
