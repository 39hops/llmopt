#!/usr/bin/env bash
# VERIFIED-ENDOGENOUS-DATA-CROSSFOSTER-1-CHAIN-DESK launcher: the registered
# paired-chain stage under the liverun interlock (id crossfoster1c), detached
# (nohup), stdout/stderr into logs/crossfoster1/chain.log (refuses if it
# exists); writes logs/crossfoster1/crossfoster1c.DONE on success only.
# Usage: bash scratch/crossfoster1c_launch.sh
set -eo pipefail
cd "$(dirname "$0")/.."
[ -e logs/crossfoster1/chain.log ] && { echo "REFUSING: logs/crossfoster1/chain.log exists"; exit 1; }
[ -e logs/crossfoster1/crossfoster1c.DONE ] && { echo "REFUSING: stale logs/crossfoster1/crossfoster1c.DONE"; exit 1; }
mkdir -p logs/crossfoster1
nohup bash -c '.venv/bin/python scripts/liverun.py run crossfoster1c -- .venv/bin/python scratch/crossfoster_chain.py && echo DONE > logs/crossfoster1/crossfoster1c.DONE' > logs/crossfoster1/chain.log 2>&1 &
echo "launched pid $!"
