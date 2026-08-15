#!/bin/bash
# ATOM-DOSE-LADDER-1 driver (pre-reg RESULTS 2026-08-14): farm the
# 12,000-row axiom shard, then three serial dose arms at BIRTH_SEED
# 3 — dose1 (1,700 rows), dose3p5 (6,000), dose7 (12,000) — each
# gated and its receipt streamed to logs/atomdose1/arms.jsonl.
. "$(dirname "$0")/lib/driver.sh"
llmopt_cd

mkdir -p logs/atomdose1
echo "=== farm $(date +%H:%M:%S) ==="
.venv/bin/python scratch/farm_atoms_axiom.py 2>&1 | tee logs/atomdose1/farm.log
rows=$(wc -l < data/micromodel_atoms_axiom_shard0.jsonl | tr -d ' ')
echo "=== shard rows: $rows ==="
if [ "$rows" -lt 10800 ]; then
  echo "FARM SHORT (<90% of 12000): NOT-RUN condition — stopping" >&2
  exit 3
fi

for spec in "dose1 1700" "dose3p5 6000" "dose7 12000"; do
  set -- $spec
  echo "=== arm=$1 nrows=$2 $(date +%H:%M:%S) ==="
  SEED=3 LABEL=$1 NROWS=$2 SHARD=data/micromodel_atoms_axiom_shard0.jsonl \
    .venv/bin/python scratch/birth19m_atoms_dose.py \
    2>&1 | tee "logs/atomdose1/$1_s3.log"
done
mark_done logs/atomdose1.DONE 0
