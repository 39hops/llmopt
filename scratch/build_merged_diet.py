"""Build data/merged_diet.jsonl (schedule-law queue item 1):
gen-6 cumulative corpus (v22 + l8 + gen4 sidecar) + the L9a shard,
with L1-L3 rationed to 45% (the gen-7 lesson). Stable string seed."""
import sys, glob, json, random
from collections import Counter
sys.path.insert(0, "scripts")
from train_mathnative import load_rows

rows = load_rows(v2=True, v21=True, v22=True, gen4=True, l8=True)
print("gen-6 base:", len(rows))
l9a = []
for f in (glob.glob("data/micromodel_l9a_shard*.jsonl")
          + sorted(glob.glob("data/pull_l9a/*.jsonl"))):
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:
            continue  # torn tail line from a live worker file
        if r["cur"].replace(" ", "") != r["nxt"].replace(" ", ""):
            l9a.append(r)
seen = set()
l9a = [r for r in l9a
       if (k := (r["cur"].replace(" ", ""), r["nxt"].replace(" ", "")))
       not in seen and not seen.add(k)]
print("L9a shard (deduped):", len(l9a))
rng = random.Random("merged-diet-2026-07-22")
out = [r for r in rows
       if r.get("level", 9) > 3 or rng.random() < 0.45] + l9a
lv = Counter(r.get("level", "?") for r in out)
print("merged:", len(out), "levels:", dict(sorted(
    lv.items(), key=lambda x: str(x[0]))))
with open("data/merged_diet.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print("wrote data/merged_diet.jsonl")
