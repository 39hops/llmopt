#!/bin/bash
# MERGE-SPACE-1 (R1) driver, 3080. Pre-reg: RESULTS PRE-REG
# MERGE-SPACE-1. Phases, streamed so partials book if the 17:00
# window closes:
#   1. four fresh d64 births, BIRTH_SEED 1..4 (micro-star recipe:
#      v22+gen4, 3ep, fast/bf16, budget 6144, L8 H4 ffn 256)
#   2. shared-init fork: seed1 trained 1 epoch = fork base; two
#      resumes to 3 epochs with budgets 6144 v 8192.
#      IMPLEMENTATION NOTE (books with the verdict): the pre-reg
#      said "shuffled order 1a/1b" but epoch order-shuffle is
#      seeded by epoch number alone (random.Random(ep)), identical
#      across forks — the fork variable is token-budget PACKING
#      (6144 v 8192), same data, different batch composition.
#   3. merges via llmopt/lab/merge.py (sidecars = provenance),
#      independent averages carry shared_lineage=True as the
#      REGISTERED OVERRIDE (this is the soups-crater replication).
#   4. gate every parent + child: scratch/gate_ckpt.py, d64 L8
#      ffn256 H4, receipts to logs/merge_space1/.
set -e
cd ~/code/llmopt
mkdir -p logs/merge_space1
G="scratch/gate_ckpt.py"
ARCH="64 8 256 4"

birth () { # seed out epochs budget
  BIRTH_SEED=$1 .venv/bin/python scripts/train_mathnative.py \
    --v22 --gen4 --fast --epochs $3 --budget $4 \
    --d 64 --layers 8 --ffn 256 --heads 4 \
    --out "$2" > "logs/merge_space1/$(basename $2 .pt)_birth.log" 2>&1
}
gate () { # ckpt label
  .venv/bin/python $G "$1" $ARCH "$2" > "logs/merge_space1/$2_gate.log" 2>&1
  grep -h "gate" "logs/merge_space1/$2_gate.log" | tail -1
}

# 1. four independent parents
for S in 1 2 3 4; do
  birth $S checkpoints/msearch_s$S.pt 3 6144
  gate checkpoints/msearch_s$S.pt msearch_s$S
done

# 2. shared-init fork
birth 1 checkpoints/msearch_fork_base.pt 1 6144
for V in a b; do
  cp checkpoints/msearch_fork_base.pt checkpoints/msearch_1$V.pt
  printf '0' > checkpoints/msearch_1$V.pt.ep
done
BIRTH_SEED=1 .venv/bin/python scripts/train_mathnative.py --v22 --gen4 --fast \
  --epochs 3 --budget 6144 --d 64 --layers 8 --ffn 256 --heads 4 \
  --out checkpoints/msearch_1a.pt > logs/merge_space1/msearch_1a_birth.log 2>&1
BIRTH_SEED=1 .venv/bin/python scripts/train_mathnative.py --v22 --gen4 --fast \
  --epochs 3 --budget 8192 --d 64 --layers 8 --ffn 256 --heads 4 \
  --out checkpoints/msearch_1b.pt > logs/merge_space1/msearch_1b_birth.log 2>&1
gate checkpoints/msearch_1a.pt msearch_1a
gate checkpoints/msearch_1b.pt msearch_1b

# 3+4. merges + gates
.venv/bin/python - << 'EOF'
from llmopt.lab.merge import average, task_vector
from itertools import combinations
# independent pairs: registered override — the crater replication
for i, j in combinations([1, 2, 3, 4], 2):
    average(f"checkpoints/msearch_s{i}.pt", f"checkpoints/msearch_s{j}.pt",
            f"checkpoints/msearch_avg_{i}{j}.pt", shared_lineage=True,
            label=f"R1-indep-{i}{j} REGISTERED-OVERRIDE crater-replication")
# shared-init pair — the informative cell
average("checkpoints/msearch_1a.pt", "checkpoints/msearch_1b.pt",
        "checkpoints/msearch_avg_shared.pt", shared_lineage=True,
        label="R1-shared-init")
# task-vector, probe-grade
task_vector("checkpoints/msearch_fork_base.pt", "checkpoints/msearch_1a.pt",
            "checkpoints/msearch_1b.pt", "checkpoints/msearch_tv.pt",
            label="R1-task-vector")
print("merges done")
EOF
for P in 12 13 14 23 24 34; do
  gate checkpoints/msearch_avg_$P.pt msearch_avg_$P
done
gate checkpoints/msearch_avg_shared.pt msearch_avg_shared
gate checkpoints/msearch_tv.pt msearch_tv
echo MERGE-SPACE-1 COMPLETE
