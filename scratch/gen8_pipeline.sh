#!/bin/bash
# GEN-8 everything-diet birth (2026-07-23 evening, Mac/MPS, vocab-41).
# Diet = math base + poly3 train + series chain3 train + physics+energy
# + duo-mined shard1 + practice rows p1. Graded: three continent probes
# + production gate + rarity battery (paired vs the poly3 birth).
# Pre-registered (booked in RESULTS before launch):
#   (a) math gate 65 +/- 2  (b) series >=94 / energy >=99 / poly >=88
#   (c) rarity rare+unseen >= poly3-birth comparator, same battery/device
set -e
cd ~/code/llmopt

.venv/bin/python - << 'PYEOF'
import json, sys
sys.path.insert(0, "scripts")
from train_mathnative import load_rows
base = load_rows(gen4=True)
nb = len(base)
out = open("data/gen8_diet.jsonl", "w")
n = 0
for r in base:
    out.write(json.dumps({"cur": r["cur"], "nxt": r["nxt"],
                          "level": r["level"]}) + "\n"); n += 1
def emit(path, skip=0):
    global n
    for i, line in enumerate(open(path)):
        if i < skip: continue
        r = json.loads(line)
        if r["cur"].replace(" ", "") == r["nxt"].replace(" ", ""): continue
        out.write(json.dumps({"cur": r["cur"], "nxt": r["nxt"],
                              "level": r.get("level", 4)}) + "\n"); n += 1
emit("data/poly3_diet.jsonl", skip=nb)       # poly3 train only
emit("data/series_diet_1e.jsonl", skip=nb)   # series chain3 train only
emit("data/phys_energy_diet.jsonl")          # physics + energy (no base)
emit("data/duo_mined_shard1.jsonl")
emit("data/practice_rows_p1.jsonl")
out.close()
print(f"gen8 diet {n} rows (base {nb})")
PYEOF

BIRTH_SEED=1 VOCAB_EXTRA=t .venv/bin/python scripts/train_mathnative.py \
  --diet data/gen8_diet.jsonl --epochs 3 \
  --out checkpoints/mathnative_19m_gen8.pt > logs/gen8_birth.log 2>&1

for P in series_probe_1e phys_energy_probe poly3_probe; do
  sed "s/series_probe.jsonl/$P.jsonl/" scratch/series_probe.py > /tmp/gen8_$P.py
  VOCAB_EXTRA=t .venv/bin/python /tmp/gen8_$P.py \
    checkpoints/mathnative_19m_gen8.pt > logs/gen8_$P.log 2>&1
done

VOCAB_EXTRA=t .venv/bin/python scratch/gate_rarity.py \
  checkpoints/mathnative_19m_gen8.pt 384 8 1536 6 gen8 \
  > logs/gen8_rarity.log 2>&1
.venv/bin/python scratch/gate_rarity.py \
  checkpoints/mathnative_19m_poly3.pt 384 8 1536 6 poly3cmp \
  > logs/poly3_rarity.log 2>&1

echo GEN8_DONE >> logs/gen8_rarity.log
grep -hE "SERIES|gate|RARITY" logs/gen8_*.log logs/poly3_rarity.log | grep -v "ep[0-9]"
