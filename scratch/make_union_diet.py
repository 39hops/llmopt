"""Build the math+ZX union diet (next-session-2 item 1): gen-4
math rows + zx_farm1_train, one jsonl. Shares are organic
(~133k math / ~97k ZX = 58/42); ZX provenance keys (kind, site)
kept. Output: data/union_math_zx.jsonl
"""
import json
import sys

sys.path.insert(0, "scripts")
from train_mathnative import load_rows

rows = load_rows(gen4=True)
n_math = len(rows)
rows += [json.loads(l) for l in open("data/zx_farm1_train.jsonl")]
with open("data/union_math_zx.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"union diet: {n_math} math + {len(rows)-n_math} zx "
      f"= {len(rows)} rows -> data/union_math_zx.jsonl", flush=True)
