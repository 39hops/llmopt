"""RULE-ABLATE-1 shard derivation (pre-reg RESULTS 2026-08-15):
selection-only split of the frozen sympy atom shard. Writes
data/micromodel_atoms_noheur.jsonl (all non-i_heurisch rows) and
data/micromodel_atoms_ctrl3218.jsonl (string-seeded random sample
of equal size from the full shard). The frozen source is read,
never modified; no new generation, no seed-band spend.

Usage: .venv/bin/python scratch/make_ruleablate_shards.py
"""
import json
import random
from pathlib import Path

SRC = Path("data/micromodel_atoms_shard0.jsonl")
rows = [json.loads(l) for l in SRC.open()]
assert len(rows) == 6000, len(rows)

noheur = [r for r in rows if r["rule"] != "i_heurisch"]
assert len(noheur) == 3218, len(noheur)
ctrl = rows[:]
random.Random("rule-ablate-1").shuffle(ctrl)
ctrl = ctrl[:3218]

for name, sel in (("noheur", noheur), ("ctrl3218", ctrl)):
    out = Path(f"data/micromodel_atoms_{name}.jsonl")
    with out.open("w") as f:
        for r in sel:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"{out}: {len(sel)} rows, levels "
          f"{dict(sorted(Counter(r['level'] for r in sel).items()))}, "
          f"heurisch {sum(1 for r in sel if r['rule']=='i_heurisch')}")
