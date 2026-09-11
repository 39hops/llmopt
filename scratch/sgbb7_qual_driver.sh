#!/bin/bash
# SG-BOUNDARY-BLOCK7-1 qualification driver (PRE-REG SG-BOUNDARY-BLOCK7-1):
# at seed 28 the CONTROL (MODE=zero K_BP=4, the FROZEN-BACKBONE-1 arm) is
# born and gated first (c); if ADEQUATE (c >= 24) the one SG7 cell is born
# and gated; the law books FUNCTION-MATCH (c - 7 <= g <= c + 7), SG7-MISS
# or SG7-UNSTABLE. NOT LAUNCHED without a separate Artin GO. Under the
# liverun interlock; marker on success only.
. "$(dirname "$0")/lib/driver.sh"
llmopt_cd
[ -z "$(git status --porcelain)" ] || { echo "driver: dirty tree"; git status --porcelain; exit 2; }
mkdir -p logs/sgbb7
[ ! -e logs/sgbb7/driver.log ] || { echo "driver: logs/sgbb7/driver.log exists (a booked receipt path); refusing"; exit 2; }
exec > >(tee logs/sgbb7/driver.log) 2>&1     # the registered driver.log receipt: every line of this run
rc=0
echo "=== integrity smoke on the launch commit $(git rev-parse --short HEAD) $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
INTEG_TAG=_launch .venv/bin/python scratch/sg7_integrity_smoke.py 2>&1 | tee logs/sgbb7/integrity_smoke_launch.log || rc=$?
[ "$rc" -eq 0 ] || { echo "driver: integrity smoke failed rc=$rc, no birth"; mark_done logs/sgbb7q.DONE "$rc"; exit "$rc"; }
run_gate() {
  .venv/bin/python scratch/sg7_qualgate.py 2>&1 | tee -a logs/sgbb7/gate.log || rc=$?
  [ "$rc" -eq 0 ] || { echo "driver: gate failed rc=$rc"; mark_done logs/sgbb7q.DONE "$rc"; exit "$rc"; }
}
stop_now() {
  .venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('logs/sgbb7/ladder.json'))['stop'] else 1)"
}
echo "=== control seed 28 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
SG7=1 MODE=zero K_BP=4 SEED=28 LR=3e-4 .venv/bin/python scratch/birth19m_sg7.py 2>&1 | tee logs/sgbb7/train_control.log || rc=$?
[ "$rc" -eq 0 ] || { echo "driver: control birth failed rc=$rc"; mark_done logs/sgbb7q.DONE "$rc"; exit "$rc"; }
run_gate
if stop_now; then echo "driver: stopped after the control (inadequate)"; mark_done logs/sgbb7q.DONE 0; exit 0; fi
echo "=== sg7 cell seed 28 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
SG7=1 MODE=sg7 SEED=28 LR=3e-4 .venv/bin/python scratch/birth19m_sg7.py 2>&1 | tee logs/sgbb7/train_sg7.log || rc=$?
[ "$rc" -eq 0 ] || { echo "driver: sg7 birth failed rc=$rc"; mark_done logs/sgbb7q.DONE "$rc"; exit "$rc"; }
run_gate
mark_done logs/sgbb7q.DONE "$rc"
