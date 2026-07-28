"""Distribution rows (spec 2026-07-28 rung 3): for each diet cur,
enumerate the engine's verified-valid moves (successors: sympy-
verified, non-identity by construction), weight by MarkovPrior
(rule-name unigram, @site stripped, unseen = 0.5*median — the
proposer's own convention), emit ALL of them as weighted rows.
Rows STREAM out incrementally (the killed-worker doctrine).
sympify here runs on farm-certified diet strings, not model text.
"""
import json
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp  # noqa: E402

from llmopt.search.derivation import State, successors  # noqa: E402
from llmopt.search.engine import MarkovPrior  # noqa: E402
from train_mathnative import load_rows  # noqa: E402

N_STATES = 4000
OUT = "data/dist_rows_d256.jsonl"

prior = MarkovPrior.load()
med = (sorted(prior.unigram.values())[len(prior.unigram) // 2]
       if prior.unigram else 1)
rows = load_rows(gen4=True)
# dedupe cur states (many rows share cur via chains)
seen_cur = set()
pool = []
for r in rows:
    c = r["cur"].replace(" ", "")
    if c not in seen_cur:
        seen_cur.add(c)
        pool.append(r["cur"])
random.Random(99_200_000).shuffle(pool)

out = open(OUT, "a")
done = emitted = failed = 0
for cur in pool[:N_STATES]:
    try:
        expr = sp.sympify(cur)
    except Exception:
        failed += 1
        continue
    try:
        kids = list(successors(State(expr)))
    except Exception:
        failed += 1
        continue
    valid = []
    seen = set()
    for name, s in kids:
        nxt = sp.sstr(s.expr)
        if nxt.replace(" ", "") in seen:
            continue
        seen.add(nxt.replace(" ", ""))
        valid.append((name.split("@")[0], nxt))
    if not valid:
        continue
    ws = [prior.unigram.get(n, 0.5 * med) for n, _ in valid]
    tot = sum(ws)
    for (n, nxt), w in zip(valid, ws):
        out.write(json.dumps(
            {"cur": cur, "nxt": nxt, "w": w / tot, "src": "dist"})
            + "\n")
        emitted += 1
    out.flush()
    done += 1
    if done % 200 == 0:
        print(f"{done} states / {emitted} rows / {failed} skipped",
              flush=True)
print(f"done: {done} states, {emitted} rows "
      f"({emitted/max(done,1):.2f} moves/state), {failed} skipped",
      flush=True)
