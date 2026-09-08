#!/bin/bash
# WRITER-TRAJECTORY-CENSUS-0 driver (pre-reg RESULTS L67576, sealed
# L67810): STAGE 0 trajectory census first (its census.json is written
# and preserved before any gate), then STAGE 0B dependence and swap
# gates, then the independent verifier. Zero training. Marker on
# success only with the real rc.
. "$(dirname "$0")/lib/driver.sh"
llmopt_cd
[ -z "$(git status --porcelain)" ] || { echo "driver: dirty tree"; git status --porcelain; exit 2; }
mkdir -p logs/writertraj0
rc=0
echo "=== stage 0 census $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python scratch/writertraj_census.py 2>&1 | tee logs/writertraj0/census.log || rc=$?
[ "$rc" -eq 0 ] || { mark_done logs/writertraj0.DONE "$rc"; exit "$rc"; }
[ -f logs/writertraj0/census.json ] || { echo "driver: census.json missing"; exit 3; }
shasum -a 256 logs/writertraj0/census.json > logs/writertraj0/census.json.sha256
echo "=== stage 0B dependence + swaps $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python scratch/writertraj_depend.py 2>&1 | tee logs/writertraj0/depend.log || rc=$?
[ "$rc" -eq 0 ] || { mark_done logs/writertraj0.DONE "$rc"; exit "$rc"; }
shasum -a 256 -c logs/writertraj0/census.json.sha256 || { echo "driver: census.json changed during stage 0B"; exit 4; }
echo "=== verifier $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python scratch/writertraj_verify.py 2>&1 | tee logs/writertraj0/verify.log || rc=$?
mark_done logs/writertraj0.DONE "$rc"
