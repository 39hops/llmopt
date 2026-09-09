#!/bin/bash
# CREDIT-ANCHOR-FRONTIER-1 qualification driver (pre-reg RESULTS L69122,
# AMENDMENT -PRECISION L69223): for ONE k, the two hybrid cells
# (1, 3e-4) then (1, 1e-4) at seed 23, then the zero-credit control at
# lr 3e-4, then the gates for that k (GATE_ONLY). The ladder decision
# (next k or selection) is taken by the caller per the sealed law:
#   bash scratch/writercaf1_qual_driver.sh 1        (k = 1)
#   SELECT=1 bash scratch/writercaf1_qual_driver.sh   (selection only)
# NOT LAUNCHED under the implement/test/smoke GO. Under liverun; marker on success only.
. "$(dirname "$0")/lib/driver.sh"
llmopt_cd
[ -z "$(git status --porcelain)" ] || { echo "driver: dirty tree"; git status --porcelain; exit 2; }
[ -f logs/writercaf1/leakage.json ] || { echo "driver: leakage.json missing"; exit 3; }
mkdir -p logs/writercaf1
rc=0
if [ "${SELECT:-0}" = "1" ]; then
  .venv/bin/python scratch/caf_qualgate.py 2>&1 | tee logs/writercaf1/qualselect.log || rc=$?
  [ "$rc" -eq 0 ] && git add -f logs/writercaf1/qual_selection.json logs/writercaf1/qual.jsonl
  mark_done logs/writercaf1_qual.DONE "$rc"; exit "$rc"
fi
K="$1"; [ -n "$K" ] || { echo "driver: k required"; exit 3; }
for cell in 1:3e-4 1:1e-4; do
  S="${cell%%:*}"; LR="${cell##*:}"
  echo "=== k=$K hybrid S=$S LR=$LR $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  QUAL=1 MODE=hybrid K_BP=$K SEED=23 S=$S LR=$LR .venv/bin/python scratch/birth19m_caf.py \
    2>&1 | tee "logs/writercaf1/train_qual_k${K}_S${S}_lr${LR}.log" || rc=$?
  [ "$rc" -eq 0 ] || { mark_done logs/writercaf1_qual.DONE "$rc"; exit "$rc"; }
done
echo "=== k=$K zero-credit control LR=3e-4 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
QUAL=1 MODE=zero K_BP=$K SEED=23 LR=3e-4 .venv/bin/python scratch/birth19m_caf.py \
  2>&1 | tee "logs/writercaf1/train_qual_k${K}_zero.log" || rc=$?
[ "$rc" -eq 0 ] || { mark_done logs/writercaf1_qual.DONE "$rc"; exit "$rc"; }
echo "=== k=$K gates $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
GATE_ONLY=1 .venv/bin/python scratch/caf_qualgate.py 2>&1 | tee "logs/writercaf1/qualgate_k${K}.log" || rc=$?
mark_done logs/writercaf1_qual.DONE "$rc"
