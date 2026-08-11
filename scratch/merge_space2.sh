#!/bin/bash
# MERGE-SPACE-2 (R2) driver, 3080. Pre-reg: RESULTS PRE-REG
# MERGE-SPACE-2. Fork-point ladder: shared trajectory 0/3 (ep0
# cell) and 2/3 (ep2 cell); the 1/3 cell is R1's avg_shared,
# reused not rerun. Reuses R1 artifacts in place: msearch_s1.pt
# (ep0 parent, 6144), msearch_fork_base.pt (ep1 base).
set -e
cd ~/code/llmopt
mkdir -p logs/merge_space2
G="scratch/gate_ckpt.py"
ARCH="64 8 256 4"

gate () { # ckpt label
  .venv/bin/python $G "$1" $ARCH "$2" > "logs/merge_space2/$2_gate.log" 2>&1
  grep -h "gate" "logs/merge_space2/$2_gate.log" | tail -1
}

# ep0 cell: one new birth (seed1 init, budget 8192, 3 epochs)
BIRTH_SEED=1 .venv/bin/python scripts/train_mathnative.py --v22 --gen4 --fast \
  --epochs 3 --budget 8192 --d 64 --layers 8 --ffn 256 --heads 4 \
  --out checkpoints/msearch_s1w.pt > logs/merge_space2/msearch_s1w_birth.log 2>&1
gate checkpoints/msearch_s1w.pt msearch_s1w

# ep2 cell: fork base (ep1) + 1 epoch = base_e2, then two 1-epoch resumes
cp checkpoints/msearch_fork_base.pt checkpoints/msearch_base_e2.pt
printf '0' > checkpoints/msearch_base_e2.pt.ep
BIRTH_SEED=1 .venv/bin/python scripts/train_mathnative.py --v22 --gen4 --fast \
  --epochs 2 --budget 6144 --d 64 --layers 8 --ffn 256 --heads 4 \
  --out checkpoints/msearch_base_e2.pt > logs/merge_space2/msearch_base_e2_birth.log 2>&1
for V in a b; do
  cp checkpoints/msearch_base_e2.pt checkpoints/msearch_2$V.pt
  printf '1' > checkpoints/msearch_2$V.pt.ep
done
BIRTH_SEED=1 .venv/bin/python scripts/train_mathnative.py --v22 --gen4 --fast \
  --epochs 3 --budget 6144 --d 64 --layers 8 --ffn 256 --heads 4 \
  --out checkpoints/msearch_2a.pt > logs/merge_space2/msearch_2a_birth.log 2>&1
BIRTH_SEED=1 .venv/bin/python scripts/train_mathnative.py --v22 --gen4 --fast \
  --epochs 3 --budget 8192 --d 64 --layers 8 --ffn 256 --heads 4 \
  --out checkpoints/msearch_2b.pt > logs/merge_space2/msearch_2b_birth.log 2>&1
gate checkpoints/msearch_base_e2.pt msearch_base_e2
gate checkpoints/msearch_2a.pt msearch_2a
gate checkpoints/msearch_2b.pt msearch_2b

# merges
.venv/bin/python - << 'EOF'
from llmopt.lab.merge import average
average("checkpoints/msearch_s1.pt", "checkpoints/msearch_s1w.pt",
        "checkpoints/msearch_avg_e0.pt", shared_lineage=True,
        label="R2-e0 shared-init-only REGISTERED cell")
average("checkpoints/msearch_2a.pt", "checkpoints/msearch_2b.pt",
        "checkpoints/msearch_avg_e2.pt", shared_lineage=True,
        label="R2-e2 two-thirds-shared cell")
print("merges done")
EOF
gate checkpoints/msearch_avg_e0.pt msearch_avg_e0
gate checkpoints/msearch_avg_e2.pt msearch_avg_e2
echo MERGE-SPACE-2 COMPLETE
