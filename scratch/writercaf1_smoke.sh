#!/bin/bash
# CREDIT-ANCHOR-FRONTIER-1 path-isolated smokes (AMENDMENT -PRECISION F4):
# 300-step seed-11 hybrid births for k = 1, 2, 4 and one zero-credit
# birth (k = 2) on mps; receipts logs/writercaf1/smoke.jsonl only.
set -eo pipefail
cd "$(dirname "$0")/.."
for K in 1 2 4; do
  SMOKE=1 MODE=hybrid K_BP=$K SEED=11 S=1 LR=3e-4 .venv/bin/python scratch/birth19m_caf.py > "logs/writercaf1/smoke_hybrid_k${K}.log" 2>&1 && echo "hybrid k=$K rc=0"
done
SMOKE=1 MODE=zero K_BP=2 SEED=11 LR=3e-4 .venv/bin/python scratch/birth19m_caf.py > logs/writercaf1/smoke_zero_k2.log 2>&1 && echo "zero k=2 rc=0"
