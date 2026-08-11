#!/bin/bash
# MERGE-SPACE-3 (R2b) driver, 3080. Pre-reg: RESULTS PRE-REG
# MERGE-SPACE-3. One birth (same init as s1, independent data
# order via ORDER_SEED=7), one merge, two gates.
set -e
cd ~/code/llmopt
mkdir -p logs/merge_space3
G="scratch/gate_ckpt.py"
ARCH="64 8 256 4"

gate () {
  .venv/bin/python $G "$1" $ARCH "$2" > "logs/merge_space3/$2_gate.log" 2>&1
  grep -h "gate" "logs/merge_space3/$2_gate.log" | tail -1
}

BIRTH_SEED=1 ORDER_SEED=7 .venv/bin/python scripts/train_mathnative.py \
  --v22 --gen4 --fast --epochs 3 --budget 6144 \
  --d 64 --layers 8 --ffn 256 --heads 4 \
  --out checkpoints/msearch_s1o.pt > logs/merge_space3/msearch_s1o_birth.log 2>&1
gate checkpoints/msearch_s1o.pt msearch_s1o

.venv/bin/python - << 'EOF'
from llmopt.lab.merge import average
average("checkpoints/msearch_s1.pt", "checkpoints/msearch_s1o.pt",
        "checkpoints/msearch_avg_ord.pt", shared_lineage=True,
        label="R2b same-init independent-order deconfound")
print("merge done")
EOF
gate checkpoints/msearch_avg_ord.pt msearch_avg_ord
echo MERGE-SPACE-3 COMPLETE
