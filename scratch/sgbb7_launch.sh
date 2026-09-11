#!/usr/bin/env bash
# SG-BOUNDARY-BLOCK7-1 launcher: the sealed qualification driver under the
# liverun interlock (id sgbb7q), detached (nohup); the driver's own tee
# produces logs/sgbb7/driver.log (refuse-if-exists inside the driver), the
# launcher's stdout goes to logs/sgbb7/launch.out.
# Usage: bash scratch/sgbb7_launch.sh   (Artin GO only)
set -eo pipefail
cd "$(dirname "$0")/.."
[ -e logs/sgbb7/driver.log ] && { echo "REFUSING: logs/sgbb7/driver.log exists"; exit 1; }
mkdir -p logs/sgbb7
nohup .venv/bin/python scripts/liverun.py run sgbb7q -- bash scratch/sgbb7_qual_driver.sh > logs/sgbb7/launch.out 2>&1 &
echo "launched pid $!"
