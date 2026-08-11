#!/bin/bash
# MERGE-SPACE-5 (R2d) driver, 3080. Pre-reg: RESULTS PRE-REG
# MERGE-SPACE-5. Two order-twin births at seeds 2/3 + merges with
# the existing R1 parents + gates.
set -e
cd ~/code/llmopt
mkdir -p logs/merge_space5
G="scratch/gate_ckpt.py"
ARCH="64 8 256 4"
gate () {
  .venv/bin/python $G "$1" $ARCH "$2" > "logs/merge_space5/$2_gate.log" 2>&1
  grep -h "gate" "logs/merge_space5/$2_gate.log" | tail -1
}
for S in 2 3; do
  BIRTH_SEED=$S ORDER_SEED=7 .venv/bin/python scripts/train_mathnative.py \
    --v22 --gen4 --fast --epochs 3 --budget 6144 \
    --d 64 --layers 8 --ffn 256 --heads 4 \
    --out "checkpoints/msearch_s${S}o.pt" \
    > "logs/merge_space5/msearch_s${S}o_birth.log" 2>&1
  gate "checkpoints/msearch_s${S}o.pt" "msearch_s${S}o"
done
.venv/bin/python - << 'PYEOF'
from llmopt.lab.merge import average
for s in (2, 3):
    average(f"checkpoints/msearch_s{s}.pt", f"checkpoints/msearch_s{s}o.pt",
            f"checkpoints/msearch_avg{s}o.pt", shared_lineage=True,
            label=f"R2d seed-{s} same-init independent-order pair")
print("merges done")
PYEOF
gate checkpoints/msearch_avg2o.pt msearch_avg2o
gate checkpoints/msearch_avg3o.pt msearch_avg3o
echo MERGE-SPACE-5 COMPLETE
