"""Pincer R1b prep: (t, rule, child)-label recovery by engine
replay (spec 2026-07-26-reverse-llmue-pincer.md amendment 1/2).

Math chain rows carry only (cur, nxt, level) — no rule
annotation. B-b's training labels are recoverable WITHOUT an
axiom ask iff replaying successors(cur) finds nxt among the
legal children. This pilot measures that recovery rate on a
sample of gen-4 rows: UNIQUE (one rule matches), AMBIG (several
rules reach the same child — soft-label mass splits), MISS
(nxt not in the legal set — e.g. macro/multi-step or algebra-
normalized rows). Books the label-extraction economics.

    python scratch/pincer_r1b_labels.py [n_sample=300]
"""
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp

from train_mathnative import load_rows
from llmopt.search.derivation import State, successors

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
norm = lambda s: s.replace(" ", "")  # noqa: E731

rows = load_rows(gen4=True)
rows = [r for r in rows if norm(r["cur"]) != norm(r["nxt"])]
rnd = random.Random("r1b-labels-1")
sample = rnd.sample(rows, N)

uniq = ambig = miss = err = 0
rule_hist = {}
for r in sample:
    try:
        cur = sp.sympify(r["cur"])
        target = norm(sp.sstr(sp.sympify(r["nxt"])))
        names = [name for name, ch in successors(State(cur), use_macros=True)
                 if norm(sp.sstr(ch.expr)) == target]
    except Exception:
        err += 1
        continue
    if len(names) == 1:
        uniq += 1
        rule_hist[names[0]] = rule_hist.get(names[0], 0) + 1
    elif names:
        ambig += 1
        for n_ in names:
            rule_hist[n_] = rule_hist.get(n_, 0) + 1
    else:
        miss += 1

top = sorted(rule_hist.items(), key=lambda kv: -kv[1])[:8]
print(f"R1b label recovery on {N} rows: unique {uniq} ambig {ambig} "
      f"miss {miss} err {err} -> recoverable "
      f"{100*(uniq+ambig)/max(N,1):.1f}%")
print("top rules:", top)
