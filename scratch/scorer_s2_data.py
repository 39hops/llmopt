"""S2 data farm (calibrated-scorer spec): training rows for the
listwise scorer. Stratified sample of unique corpus states ->
full legal enumeration -> per-child value labels (cache-aware
fork solves, budget 150, 8s walls, 6 workers, streamed shards)
+ the replayed true-move label where the corpus row matches a
legal child (the R1b 68%). Every solve extends the permanent
value cache.

    python scratch/scorer_s2_data.py [per_level=400]
Out: data/scorer_train_v1.jsonl
  {level, state, true_child_idx|null, children: [{rule, child,
   solved, plies, nodes, n_tok}]}
"""
import json
import multiprocessing as mp
import os
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp

from train_mathnative import load_rows
from llmopt.train.mathnative import MathTokenizer
from llmopt.search.derivation import State, successors

PER_LV = int(sys.argv[1]) if len(sys.argv) > 1 else 400
BUDGET, WALL, WORKERS = 150, 8, 6
CACHE = "data/value_cache.jsonl"
norm = lambda s: s.replace(" ", "")  # noqa: E731
tok = MathTokenizer()

rows = load_rows(gen4=True)
by_cur = {}
for r in rows:
    k = norm(r["cur"])
    if k not in by_cur:
        by_cur[k] = (r.get("level", 0), r["cur"], set())
    by_cur[k][2].add(norm(r["nxt"]))
by_lv = {}
for k, (lv, cur, nxts) in by_cur.items():
    by_lv.setdefault(lv, []).append(k)
rnd = random.Random("s2-data-1")
keys = []
for lv in sorted(by_lv):
    ks = sorted(by_lv[lv])
    rnd.shuffle(ks)
    keys += ks[:PER_LV]
print(f"{len(keys)} states sampled", flush=True)

enum = {}
for k in keys:
    lv, cur, nxts = by_cur[k]
    try:
        legal = [(n, sp.sstr(c.expr)) for n, c in
                 successors(State(sp.sympify(cur)), use_macros=True)]
    except Exception:
        continue
    if len(legal) >= 2:
        enum[k] = (lv, cur, nxts, legal)
todo = {norm(c): c for _, (_, _, _, legal) in enum.items()
        for _, c in legal}
cache = {}
if os.path.exists(CACHE):
    for r in map(json.loads, open(CACHE)):
        if r["budget"] == BUDGET:
            cache[r["key"]] = r
work = [c for k, c in todo.items() if k not in cache]
print(f"{len(enum)} states enumerable; {len(todo)} distinct children; "
      f"{len(cache)} cached; {len(work)} to solve", flush=True)


def _worker(idx, exprs):
    out = open(f"data/vcache_s2_shard{idx}.jsonl", "w")
    from llmopt.search.engine import solve

    def one(es, q):
        try:
            r = solve(sp.sympify(es), budget=BUDGET)
            q.put({"solved": bool(r.solved), "nodes": r.nodes,
                   "plies": r.state.plies})
        except Exception:
            q.put(None)
    for es in exprs:
        ctx = mp.get_context("fork")
        q = ctx.Queue()
        p = ctx.Process(target=one, args=(es, q))
        p.start()
        p.join(WALL)
        if p.is_alive():
            p.kill()
            p.join()
            v = None
        else:
            try:
                v = q.get(timeout=10)
            except Exception:
                v = None
        row = {"key": norm(es), "budget": BUDGET}
        row.update(v or {"solved": None})
        out.write(json.dumps(row) + "\n")
        out.flush()
    out.close()


procs = [mp.get_context("fork").Process(target=_worker,
                                        args=(i, work[i::WORKERS]))
         for i in range(WORKERS)]
[p.start() for p in procs]
[p.join() for p in procs]
with open(CACHE, "a") as cf:
    for i in range(WORKERS):
        f = f"data/vcache_s2_shard{i}.jsonl"
        if os.path.exists(f):
            for line in open(f):
                cf.write(line)
                cache[json.loads(line)["key"]] = json.loads(line)
            os.remove(f)
print(f"cache now {len(cache)}", flush=True)

n_true = n_out = 0
with open("data/scorer_train_v1.jsonl", "w") as out:
    for k, (lv, cur, nxts, legal) in enum.items():
        ch, true_idx = [], None
        for j, (rule, c) in enumerate(legal):
            v = cache.get(norm(c), {})
            try:
                ntok = len(tok.encode(c))
            except ValueError:
                ntok = None
            if norm(c) in nxts and true_idx is None:
                true_idx = j
            ch.append({"rule": rule, "child": c,
                       "solved": v.get("solved"),
                       "plies": v.get("plies"),
                       "nodes": v.get("nodes"), "n_tok": ntok})
        n_true += true_idx is not None
        n_out += 1
        out.write(json.dumps({"level": lv, "state": cur,
                              "true_child_idx": true_idx,
                              "children": ch}) + "\n")
print(f"{n_out} training states -> data/scorer_train_v1.jsonl "
      f"({n_true} with true-move label = {100*n_true/max(n_out,1):.0f}%)",
      flush=True)
