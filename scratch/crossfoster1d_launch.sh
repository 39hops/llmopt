#!/usr/bin/env bash
# VERIFIED-ENDOGENOUS-DATA-CROSSFOSTER-1 donor stage launcher: the registered
# donor-library builder under the liverun interlock (id crossfoster1d),
# detached (nohup), stdout/stderr into the registered logs/crossfoster1/donor.log
# (refuses if it exists); writes logs/crossfoster1/crossfoster1d.DONE on
# success only (refuses on a stale marker). Usage: bash scratch/crossfoster1d_launch.sh
set -eo pipefail
cd "$(dirname "$0")/.."
[ -e logs/crossfoster1/donor.log ] && { echo "REFUSING: logs/crossfoster1/donor.log exists"; exit 1; }
[ -e logs/crossfoster1/crossfoster1d.DONE ] && { echo "REFUSING: stale logs/crossfoster1/crossfoster1d.DONE"; exit 1; }
mkdir -p logs/crossfoster1
nohup bash -c '.venv/bin/python scripts/liverun.py run crossfoster1d -- .venv/bin/python scratch/crossfoster_donor.py && echo DONE > logs/crossfoster1/crossfoster1d.DONE' > logs/crossfoster1/donor.log 2>&1 &
echo "launched pid $!"
