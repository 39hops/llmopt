#!/bin/bash
# WRITER-DFA-1 qualification driver (pre-reg RESULTS L68321, sealed
# L68543 / L68644): the requested (S:LR) cells at seed 21 in order,
# MODE=dfa, one birth each with its own receipt row; a non-finite cell
# books UNSTABLE and the ladder continues. With GATE=1 the four cells are
# gated afterwards and the selection frozen (scratch/dfa_qualgate.py).
# Runs under the liverun interlock; marker on success only.
#   bash scratch/writerdfa1_qual_driver.sh 1:3e-4              (cell 1, then receipt-auditor)
#   GATE=1 bash scratch/writerdfa1_qual_driver.sh 1:1e-4 0.25:3e-4 4:1e-4
. "$(dirname "$0")/lib/driver.sh"
llmopt_cd
[ -z "$(git status --porcelain)" ] || { echo "driver: dirty tree"; git status --porcelain; exit 2; }
[ -f logs/writerdfa1/probe.json ] || { echo "driver: probe.json missing"; exit 3; }
[ -f logs/writerdfa1/leakage.json ] || { echo "driver: leakage.json missing"; exit 3; }
mkdir -p logs/writerdfa1
rc=0
for cell in "$@"; do
  S="${cell%%:*}"; LR="${cell##*:}"
  echo "=== qual cell S=$S LR=$LR $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  QUAL=1 MODE=dfa SEED=21 S=$S LR=$LR .venv/bin/python scratch/birth19m_dfa.py \
    2>&1 | tee "logs/writerdfa1/train_qual_S${S}_lr${LR}.log" || rc=$?
  [ "$rc" -eq 0 ] || { echo "driver: qual cell $cell failed rc=$rc"; mark_done logs/writerdfa1_qual.DONE "$rc"; exit "$rc"; }
done
if [ "${GATE:-0}" = "1" ]; then
  echo "=== qualification gates + selection $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  .venv/bin/python scratch/dfa_qualgate.py 2>&1 | tee logs/writerdfa1/qualgate.log || rc=$?
fi
mark_done logs/writerdfa1_qual.DONE "$rc"
