#!/bin/bash
# FROZEN-BACKBONE-1 driver (PRE-REG FROZEN-BACKBONE-1): three fresh paired
# seeds 24 / 25 / 26, per seed FULL (MODE=bp) and FROZEN (MODE=zero K_BP=4)
# in the sealed counterbalanced order (24 FULL first, 25 FROZEN first,
# 26 FULL first), then the six post-hoc gates and the replication law.
# NOT LAUNCHED without Artin GO. Under the liverun interlock; marker on
# success only.
. "$(dirname "$0")/lib/driver.sh"
llmopt_cd
[ -z "$(git status --porcelain)" ] || { echo "driver: dirty tree"; git status --porcelain; exit 2; }
mkdir -p logs/frozenbb1
rc=0
ORDER=(24:bp 24:zero 25:zero 25:bp 26:bp 26:zero)
for cell in "${ORDER[@]}"; do
  SEED="${cell%%:*}"; MODE="${cell##*:}"
  echo "=== seed $SEED MODE=$MODE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  FB=1 MODE=$MODE K_BP=4 SEED=$SEED LR=3e-4 .venv/bin/python scratch/birth19m_fb.py \
    2>&1 | tee "logs/frozenbb1/train_s${SEED}_${MODE}.log" || rc=$?
  [ "$rc" -eq 0 ] || { echo "driver: birth $cell failed rc=$rc"; mark_done logs/frozenbb1.DONE "$rc"; exit "$rc"; }
done
echo "=== gates + replication law $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python scratch/fb_gate.py 2>&1 | tee logs/frozenbb1/gate.log || rc=$?
mark_done logs/frozenbb1.DONE "$rc"
