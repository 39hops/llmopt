#!/bin/bash
# GEN-9: solved-only-leak A/B (pre-registered 2026-07-24).
# Two paired 19M vocab-41 births; gen-8 diet + rations (L1-L3 base
# x2) + duo shard; ONE variable: practice rows solved-only (A) vs
# +failed-attempt steps (B). Graded: production gate + continent
# probes + paired rarity battery.
set -e
cd ~/code/llmopt

.venv/bin/python - << 'PYEOF'
import json, sys
sys.path.insert(0, "scripts")
from train_mathnative import load_rows
base = load_rows(gen4=True)
nb = len(base)

def emit_common(out):
    n = 0
    for r in base:
        row = json.dumps({"cur": r["cur"], "nxt": r["nxt"],
                          "level": r["level"]}) + "\n"
        out.write(row); n += 1
        if r["level"] <= 3:                    # the rations fix: x2
            out.write(row); n += 1
    def add(path, skip=0):
        nonlocal n
        for i, line in enumerate(open(path)):
            if i < skip: continue
            r = json.loads(line)
            if r["cur"].replace(" ", "") == r["nxt"].replace(" ", ""):
                continue
            out.write(json.dumps({"cur": r["cur"], "nxt": r["nxt"],
                                  "level": r.get("level", 4)}) + "\n")
            n += 1
    add("data/poly3_diet.jsonl", skip=nb)
    add("data/series_diet_1e.jsonl", skip=nb)
    add("data/phys_energy_diet.jsonl")
    add("data/duo_mined_shard1.jsonl")
    return n

for arm, keep in (("A", ("solved", "skip")), ("B", ("solved", "skip", "unsolved"))):
    out = open(f"data/gen9_diet_{arm}.jsonl", "w")
    n = emit_common(out)
    for line in open("data/practice_rows_v5.jsonl"):
        r = json.loads(line)
        if r["outcome"] in keep and \
           r["cur"].replace(" ", "") != r["nxt"].replace(" ", ""):
            out.write(json.dumps({"cur": r["cur"], "nxt": r["nxt"],
                                  "level": r.get("level", 5)}) + "\n")
            n += 1
    out.close()
    print(f"gen9 arm {arm}: {n} rows")
PYEOF

for ARM in A B; do
  BIRTH_SEED=1 VOCAB_EXTRA=t .venv/bin/python scripts/train_mathnative.py \
    --diet data/gen9_diet_$ARM.jsonl --epochs 3 \
    --out checkpoints/mathnative_19m_gen9$ARM.pt \
    > logs/gen9${ARM}_birth.log 2>&1
  for P in series_probe_1e phys_energy_probe poly3_probe; do
    sed "s/series_probe.jsonl/$P.jsonl/" scratch/series_probe.py > /tmp/gen9_$P.py
    VOCAB_EXTRA=t .venv/bin/python /tmp/gen9_$P.py \
      checkpoints/mathnative_19m_gen9$ARM.pt > logs/gen9${ARM}_$P.log 2>&1
  done
  VOCAB_EXTRA=t .venv/bin/python scratch/gate_rarity.py \
    checkpoints/mathnative_19m_gen9$ARM.pt 384 8 1536 6 gen9$ARM \
    > logs/gen9${ARM}_rarity.log 2>&1
done

echo GEN9_AB_DONE >> logs/gen9B_rarity.log
grep -hE "SERIES|gate|RARITY" logs/gen9A_*.log logs/gen9B_*.log | grep -v "ep[0-9]"
