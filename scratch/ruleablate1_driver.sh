#!/bin/bash
# RULE-ABLATE-1 driver (pre-reg RESULTS 2026-08-15): two serial arms
# at BIRTH_SEED 3 — noheur (heurisch-ablated) and ctrl3218
# (dose-matched random control) — receipts to logs/ruleablate1/.
. "$(dirname "$0")/lib/driver.sh"
llmopt_cd

mkdir -p logs/ruleablate1
for spec in "noheur data/micromodel_atoms_noheur.jsonl" \
            "ctrl3218 data/micromodel_atoms_ctrl3218.jsonl"; do
  set -- $spec
  echo "=== arm=$1 $(date +%H:%M:%S) ==="
  SEED=3 LABEL=$1 NROWS=3218 SHARD=$2 \
    .venv/bin/python scratch/birth19m_atoms_rule.py \
    2>&1 | tee "logs/ruleablate1/$1_s3.log"
done
mark_done logs/ruleablate1.DONE 0
