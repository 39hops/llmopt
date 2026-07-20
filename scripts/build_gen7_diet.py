"""Gen-7 mass-targeted diet (Rung A of the epoch killer).

Rations by mastery (gate-saturated levels feed at maintenance;
light/new territory feeds full): L1/L2 15%, L3 30%, L4-L9 100%.
Deterministic seeded sampling; identity guard as always.

    python scripts/build_gen7_diet.py  # -> data/gen7_diet.jsonl
"""
from __future__ import annotations

import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RATION = {1: 0.15, 2: 0.15, 3: 0.30}  # else 1.0


def main() -> None:
    rows = []
    for f in sorted(glob.glob("data/micromodel_chains_shard*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    rows += [json.loads(l) for l in open("data/step_chains.jsonl")]
    for f in sorted(glob.glob("data/micromodel_v22_shard*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    for f in sorted(glob.glob("data/micromodel_l8_shard*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    rows += [json.loads(l)
             for l in open("data/micromodel_gen4_sidecar.jsonl")]
    for f in sorted(glob.glob("data/micromodel_l9a_shard*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    rng = random.Random("gen7-diet-0")
    kept = []
    for r in rows:
        if r["cur"].replace(" ", "") == r["nxt"].replace(" ", ""):
            continue
        if rng.random() < RATION.get(r.get("level", 9), 1.0):
            kept.append(r)
        elif r.get("level", 9) not in RATION:
            kept.append(r)
    with open("data/gen7_diet.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    c = Counter(r.get("level", 0) for r in kept)
    print(f"gen-7 diet: {len(kept)} rows (from {len(rows)}); "
          f"per level {dict(sorted(c.items()))}")


if __name__ == "__main__":
    main()
