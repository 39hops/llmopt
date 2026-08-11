#!/bin/bash
# MERGE-SPACE-4 (R2c) driver, 3080. Pre-reg: RESULTS PRE-REG
# MERGE-SPACE-4. Two order-twin births + 4-way soup via pairwise
# tree (equal weights exactly), gates for twins, inner averages,
# and the soup.
set -e
cd ~/code/llmopt
mkdir -p logs/merge_space4
G="scratch/gate_ckpt.py"
ARCH="64 8 256 4"

gate () {
  .venv/bin/python $G "$1" $ARCH "$2" > "logs/merge_space4/$2_gate.log" 2>&1
  grep -h "gate" "logs/merge_space4/$2_gate.log" | tail -1
}

for OS in 8 9; do
  BIRTH_SEED=1 ORDER_SEED=$OS .venv/bin/python scripts/train_mathnative.py \
    --v22 --gen4 --fast --epochs 3 --budget 6144 \
    --d 64 --layers 8 --ffn 256 --heads 4 \
    --out "checkpoints/msearch_s1o$OS.pt" \
    > "logs/merge_space4/msearch_s1o${OS}_birth.log" 2>&1
  gate "checkpoints/msearch_s1o$OS.pt" "msearch_s1o$OS"
done

.venv/bin/python - << 'EOF'
from llmopt.lab.merge import average
# pairwise tree = exact uniform 4-way (each leaf weight 1/4)
average("checkpoints/msearch_s1.pt", "checkpoints/msearch_s1o.pt",
        "checkpoints/msearch_pair_a.pt", shared_lineage=True,
        label="R2c inner pair a (s1+s1o)")
average("checkpoints/msearch_s1o8.pt", "checkpoints/msearch_s1o9.pt",
        "checkpoints/msearch_pair_b.pt", shared_lineage=True,
        label="R2c inner pair b (s1o8+s1o9)")
average("checkpoints/msearch_pair_a.pt", "checkpoints/msearch_pair_b.pt",
        "checkpoints/msearch_soup4.pt", shared_lineage=True,
        label="R2c soup4 = uniform 4-way order-twin soup")
print("merges done")
EOF
gate checkpoints/msearch_pair_a.pt msearch_pair_a
gate checkpoints/msearch_pair_b.pt msearch_pair_b
gate checkpoints/msearch_soup4.pt msearch_soup4
echo MERGE-SPACE-4 COMPLETE
