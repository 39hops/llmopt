#!/bin/bash
# GT-6 arms 8-14 (resume; arms 1-7 in jobs/gt6_ladder_arms1-7.log)
set -u
cd "$(dirname "$0")/.."
for f in checkpoints/gt6_ladder_r75_d0.json checkpoints/gt6_ladder_r75_d1.json \
         checkpoints/gt6_ladder_r80_d0.json checkpoints/gt6_ladder_r80_d1.json \
         checkpoints/gt6_novrb_d0.json checkpoints/gt6_novrb_d1.json; do
  echo "=== ARM $f ==="
  SEED=1234 FRACS=0.5 KEEPSET=$f .venv/bin/python scratch/moe_gt1_arm2.py
done
echo "=== RIDER core-only seed 777 ==="
SEED=777 FRACS=0.29 KEEPSET=checkpoints/gt3_core_keep.json .venv/bin/python scratch/moe_gt1_arm2.py
echo "=== RIDER core-only seed 2026 ==="
SEED=2026 FRACS=0.29 KEEPSET=checkpoints/gt3_core_keep.json .venv/bin/python scratch/moe_gt1_arm2.py
echo "=== ALL ARMS DONE ==="
