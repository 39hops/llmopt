"""Build the rung-3 paired diets (spec 2026-07-28, 3-arm design).

From the gen4 diet + data/dist_rows_d256.jsonl:
  - dose-control arm: each treated state's single pick row
    replicated 4x (exposure matched to the dist arm, label = pick)
  - dist arm: each treated state's 4 rows drawn as weighted
    replication of its verified-valid distribution (largest-
    remainder rounding of 4*w_i; the modal move always present)
Untreated rows pass through identically. dist vs dose-control =
one variable (the label distribution) at exactly matched rows.
Writes data/diet_dosectl_d256.jsonl + data/diet_dist_d256.jsonl.
"""
import json
import sys
from collections import defaultdict

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from train_mathnative import load_rows  # noqa: E402

REP = 4

dist = defaultdict(list)
with open("data/dist_rows_d256.jsonl") as f:
    for line in f:
        r = json.loads(line)
        dist[r["cur"].replace(" ", "")].append((r["nxt"], r["w"]))

rows = load_rows(gen4=True)
treated = set()
out_ctl = open("data/diet_dosectl_d256.jsonl", "w")
out_dst = open("data/diet_dist_d256.jsonl", "w")
n_ctl = n_dst = 0
for r in rows:
    key = r["cur"].replace(" ", "")
    base = {"cur": r["cur"], "nxt": r["nxt"]}
    if key in dist and key not in treated:
        treated.add(key)
        # dose-control: pick x4
        for _ in range(REP):
            out_ctl.write(json.dumps(base) + "\n")
            n_ctl += 1
        # dist: largest-remainder apportionment of REP over weights
        moves = dist[key]
        exact = [w * REP for _, w in moves]
        counts = [int(e) for e in exact]
        rem = REP - sum(counts)
        order = sorted(range(len(moves)),
                       key=lambda i: exact[i] - counts[i],
                       reverse=True)
        for i in order[:rem]:
            counts[i] += 1
        for (nxt, _), c in zip(moves, counts):
            for _ in range(c):
                out_dst.write(json.dumps(
                    {"cur": r["cur"], "nxt": nxt}) + "\n")
                n_dst += 1
    elif key in treated:
        continue  # duplicate cur rows of treated states collapse
    else:
        out_ctl.write(json.dumps(base) + "\n")
        out_dst.write(json.dumps(base) + "\n")
        n_ctl += 1
        n_dst += 1
print(f"treated {len(treated)} states; dosectl {n_ctl} rows, "
      f"dist {n_dst} rows (must match: {n_ctl == n_dst})", flush=True)
