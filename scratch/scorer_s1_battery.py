"""S1: the frontier battery + persistent value cache (spec
2026-07-27-calibrated-scorer; pre-reg in RESULTS).

Builds the battery the saturated 43-state pool could not be:
states whose enumerated legal children have VALUE VARIANCE.
Sources: L6/L7 gate roots + stall-endpoint states from pp
sidecars. Fork-isolated engine solves (6 workers, 25s wall,
budget 150) stream into the persistent skeleton-hash cache
(data/value_cache.jsonl) — deterministic labels, never
recomputed.

    python scratch/scorer_s1_battery.py
Out: data/scorer_battery_v1.jsonl + cache + honest length-control
read on the kept battery.
"""
import json
import math
import multiprocessing as mp
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp

from llmopt.train.mathnative import MathTokenizer
from llmopt.search.derivation import State, successors
import step_grpo_micro as G
from bench_step_tokens import _gen_isolated

BUDGET, WALL, WORKERS = 150, 25, 6
CACHE = "data/value_cache.jsonl"
norm = lambda s: s.replace(" ", "")  # noqa: E731
tok = MathTokenizer()

# ---- candidate states ----
cands = []  # (level, state)
seen = set()


def add(lv, s):
    if norm(s) not in seen:
        seen.add(norm(s))
        cands.append((lv, s))


for lv in (6, 7):
    for i in range(G.GATE_N):
        p = _gen_isolated(lv, G.GATE_BAND + 1000 * lv + i)
        if p is not None:
            add(lv, f"Integral({sp.sstr(p._expr)}, x)")
import glob
for f in sorted(glob.glob("logs/pp_*.jsonl")):
    if any(k in f for k in ("r0", "r8", "dist", "backpairs")):
        continue
    for r in map(json.loads, open(f)):
        ch = r.get("chain", [])
        if not r.get("solved") and ch and ch[-1] != "SOLVED":
            add(r["level"], ch[-1])
print(f"{len(cands)} candidate states", flush=True)

# ---- enumeration pass ----
enum = {}  # state -> [(rule, child)]
for lv, s in cands:
    try:
        legal = [(name, sp.sstr(c.expr))
                 for name, c in successors(State(sp.sympify(s)),
                                           use_macros=True)]
        if len(legal) >= 2:
            enum[s] = (lv, legal)
    except Exception:
        continue
todo = {norm(c): c for _, (_, legal) in enum.items() for _, c in legal}
print(f"{len(enum)} states enumerable; {len(todo)} distinct children",
      flush=True)

# ---- cache-aware parallel value labels ----
cache = {}
if os.path.exists(CACHE):
    for r in map(json.loads, open(CACHE)):
        if r["budget"] == BUDGET:
            cache[r["key"]] = r
work = [c for k, c in todo.items() if k not in cache]
print(f"{len(cache)} cached; {len(work)} to solve", flush=True)


def _worker(idx, exprs):
    out = open(f"data/vcache_shard{idx}.jsonl", "w")
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
        f = f"data/vcache_shard{i}.jsonl"
        if os.path.exists(f):
            for line in open(f):
                cf.write(line)
                cache[json.loads(line)["key"]] = json.loads(line)
            os.remove(f)
print(f"cache now {len(cache)}", flush=True)

# ---- assemble battery: keep states with value variance ----
kept, all_rows = 0, 0
out = open("data/scorer_battery_v1.jsonl", "w")
lensp, lentop = [], [0, 0]
for s, (lv, legal) in enum.items():
    ch = []
    for rule, c in legal:
        v = cache.get(norm(c), {})
        try:  # engine moves can leave vocab-40 (e.g. fresnelc);
            # kept with n_tok None — excluded at crystal-scoring time
            ntok = len(tok.encode(c))
        except ValueError:
            ntok = None
        ch.append({"rule": rule, "child": c,
                   "solved": v.get("solved"),
                   "plies": v.get("plies"), "nodes": v.get("nodes"),
                   "n_tok": ntok})
    known = [c for c in ch if c["solved"] is not None]
    ns = sum(1 for c in known if c["solved"])
    all_rows += 1
    mixed = 0 < ns < len(known)
    graded = (ns == len(known) and ns >= 3 and
              len({c["plies"] for c in known if c["plies"]}) >= 3)
    if not (mixed or graded):
        continue
    kept += 1
    out.write(json.dumps({"level": lv, "state": s, "kind":
                          "mixed" if mixed else "graded",
                          "children": ch}) + "\n")
    # length-only control on this state
    known = [c for c in known if c["n_tok"] is not None]
    if mixed and len(known) >= 3:
        xs = [-c["n_tok"] for c in known]
        ys = [1.0 if c["solved"] else 0.0 for c in known]
        def rank(v):
            o = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            for j, i in enumerate(o):
                r[i] = j
            return r
        rx, ry = rank(xs), rank(ys)
        mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
        dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
        dy = math.sqrt(sum((b - my) ** 2 for b in ry))
        if dx and dy:
            lensp.append(sum((a - mx) * (b - my)
                             for a, b in zip(rx, ry)) / (dx * dy))
        best = max(range(len(known)), key=lambda i: xs[i])
        lentop[0] += bool(known[best]["solved"])
        lentop[1] += 1
out.close()
print(f"battery: {kept}/{all_rows} states kept -> "
      f"data/scorer_battery_v1.jsonl", flush=True)
if lensp:
    print(f"length-only control on mixed states: mean spearman "
          f"{sum(lensp)/len(lensp):.3f}; top1-solves "
          f"{lentop[0]}/{lentop[1]}", flush=True)
