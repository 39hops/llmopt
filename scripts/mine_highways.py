"""Highway mining (Artin's contraction-hierarchy analogy): recurring
rule n-grams in winning paths = macro-move candidates with traffic
data. Analysis only — promotion to actual macros is a future spec."""

import json
from collections import Counter

rows = [json.loads(l) for l in open("data/proposer_train.jsonl")]
rules = [r["moves"][r["answer"]].split("@")[0] for r in rows]

for n in (2, 3):
    grams = Counter(tuple(rules[i:i + n]) for i in range(len(rules) - n + 1))
    total = sum(grams.values())
    print(f"top {n}-grams ({total} total):")
    for g, c in grams.most_common(6):
        print(f"  {c:4d} ({100 * c / total:4.1f}%)  {' -> '.join(g)}")
