#!/bin/bash
# SYNTHETIC-GRADIENT-WRITER-1 qualification driver (AMENDMENT -SEAL): at
# seed 27 the CONTROL (MODE=zero K_BP=4, the FROZEN-BACKBONE-1 arm) is born
# and gated first (c); then the SG cells in the sealed order LINEAR 3e-4,
# LINEAR 3e-5, MLP-256 3e-4, MLP-256 3e-5, each gated as it lands; the
# ladder STOPS at the first FUNCTION-MATCH (c - 7 <= g <= c + 7) or when
# the control is inadequate; unborn cells stay unborn. NOT LAUNCHED without
# a separate Artin GO. Under the liverun interlock; marker on success only.
. "$(dirname "$0")/lib/driver.sh"
llmopt_cd
[ -z "$(git status --porcelain)" ] || { echo "driver: dirty tree"; git status --porcelain; exit 2; }
mkdir -p logs/sgwriter1
[ ! -e logs/sgwriter1/driver.log ] || { echo "driver: logs/sgwriter1/driver.log exists (a booked receipt path); refusing"; exit 2; }
exec > >(tee logs/sgwriter1/driver.log) 2>&1     # the registered driver.log receipt: every line of this run
rc=0
echo "=== integrity smoke on the launch commit $(git rev-parse --short HEAD) $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
INTEG_TAG=_launch .venv/bin/python scratch/sg_integrity_smoke.py 2>&1 | tee logs/sgwriter1/integrity_smoke_launch.log || rc=$?
[ "$rc" -eq 0 ] || { echo "driver: integrity smoke failed rc=$rc, no birth"; mark_done logs/sgwriter1q.DONE "$rc"; exit "$rc"; }
run_gate() {
  .venv/bin/python scratch/sg_qualgate.py 2>&1 | tee -a logs/sgwriter1/gate.log || rc=$?
  [ "$rc" -eq 0 ] || { echo "driver: gate failed rc=$rc"; mark_done logs/sgwriter1q.DONE "$rc"; exit "$rc"; }
}
stop_now() {
  .venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('logs/sgwriter1/ladder.json'))['stop'] else 1)"
}
echo "=== control seed 27 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
SG=1 MODE=zero K_BP=4 SEED=27 LR=3e-4 .venv/bin/python scratch/birth19m_sg.py 2>&1 | tee logs/sgwriter1/train_control.log || rc=$?
[ "$rc" -eq 0 ] || { echo "driver: control birth failed rc=$rc"; mark_done logs/sgwriter1q.DONE "$rc"; exit "$rc"; }
run_gate
if stop_now; then echo "driver: ladder stopped after the control (inadequate)"; mark_done logs/sgwriter1q.DONE 0; exit 0; fi
for cell in linear:3e-4 linear:3e-5 mlp256:3e-4 mlp256:3e-5; do
  FAM="${cell%%:*}"; PLR="${cell##*:}"
  echo "=== sg $FAM plr $PLR seed 27 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  SG=1 MODE=sg FAMILY=$FAM PLR=$PLR SEED=27 LR=3e-4 .venv/bin/python scratch/birth19m_sg.py 2>&1 | tee "logs/sgwriter1/train_sg_${FAM}_plr${PLR}.log" || rc=$?
  [ "$rc" -eq 0 ] || { echo "driver: birth $cell failed rc=$rc"; mark_done logs/sgwriter1q.DONE "$rc"; exit "$rc"; }
  run_gate
  if stop_now; then echo "driver: ladder stopped at $cell"; break; fi
done
mark_done logs/sgwriter1q.DONE "$rc"
