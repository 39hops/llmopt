#!/bin/bash
# ATOM-DIET-TRAJECTORY-1 driver (pre-reg RESULTS L66546, sealed L66822):
# the six registered births in the frozen counterbalanced order, with
# NO gates during or between trainings; then the 24 post-hoc gates and
# the census. The verifier runs separately. Each birth streams its own
# receipt row; the marker fires on success only.
. "$(dirname "$0")/lib/driver.sh"
llmopt_cd

mkdir -p logs/atomtraj1
ORDER=(stock_s5 atoms_s5 stock_s6 atoms_s6 atoms_s7 stock_s7)
i=0
for cell in "${ORDER[@]}"; do
  ARM="${cell%_s*}"; SEED="${cell#*_s}"
  echo "=== birth $i: arm=$ARM seed=$SEED $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  ARM=$ARM SEED=$SEED ORDER_INDEX=$i .venv/bin/python scratch/birth19m_atoms_traj.py \
    2>&1 | tee "logs/atomtraj1/train_${cell}.log"
  i=$((i + 1))
done
echo "=== all six trainings complete $(date -u +%Y-%m-%dT%H:%M:%SZ); post-hoc gates ==="
.venv/bin/python scratch/birth19m_atoms_trajgate.py 2>&1 | tee logs/atomtraj1/gates.log
echo "=== census $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python scratch/atomtraj_census.py 2>&1 | tee logs/atomtraj1/census.log
mark_done logs/atomtraj1.DONE 0
