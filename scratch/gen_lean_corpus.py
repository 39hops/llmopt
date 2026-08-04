"""Generator for the Lean-tier smoke corpus (2026-08-03).

500 sympy identities in the tier-1 eligible family (polynomial
expansions, rational cancellations, fn-atom factorizations) fed to
axiom-oracle --lean-cert to produce data/lean_certs_500.jsonl
(443 emitted / 0 fenced; fence exercised separately by the sqrt smoke
row s4). Stable string seed per house convention.

Usage: .venv/bin/python scratch/gen_lean_corpus.py > data/lean_corpus_in.jsonl
       ~/code/axiom/build-rel/axiom-oracle data/lean_corpus_in.jsonl \
           /tmp/out.jsonl --lean-cert data/lean_certs_500.jsonl
"""
import json
import random

import sympy as sp

x = sp.Symbol('x')
rng = random.Random("lean-corpus-2026-08-03")
fns = [sp.exp(x), sp.sin(x), sp.cos(x), sp.log(x)]
for i in range(500):
    kind = rng.choice(["poly", "rat", "atom", "atom2"])
    a, b, c = [rng.randint(1, 9) for _ in range(3)]
    if kind == "poly":
        e = (a*x + b)**rng.choice([2, 3]) * rng.choice([1, x])
        lhs, rhs = e, sp.expand(e)
    elif kind == "rat":
        lhs, rhs = (x + a)*(x + b)/(x + a), x + b
    elif kind == "atom":
        f = rng.choice(fns)
        e = a*x*f + b*f
        lhs, rhs = e, sp.factor(e)
    else:
        f, g = rng.sample(fns, 2)
        e = (f + g)**2
        lhs, rhs = e, sp.expand(e)
    print(json.dumps({"id": f"c{i}", "task": "equiv", "var": "x",
                      "lhs": sp.sstr(lhs), "rhs": sp.sstr(rhs)}))
