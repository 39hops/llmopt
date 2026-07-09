"""Budget allocation: the magic estimator's first engine integration.

Flat budgets waste compute twice — easy problems don't need 200
nodes, hopeless ones shouldn't get 200 either. The estimator
(rho 0.855 vs measured cost) predicts per-problem hardness in
microseconds, so allocate a FIXED TOTAL node budget by predicted
need: easy -> less, hard-but-solvable -> more, predicted-hopeless ->
floor. Pre-registered bar: allocated solves strictly more than flat
at the same total budget (paired arms, one run, same problems).

Allocation: b_i proportional to predicted cost 2^chat_i, gated by
p(solved) (sigmoid of the solved head; p < 0.2 gets the floor),
clipped to [50, 600], rescaled to sum to n*200.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
from pathlib import Path

import sympy as sp
import torch

from llmopt.mathgen.problems import make_integrate
from llmopt.search.engine import solve
from llmopt.search.features import featurize

CKPT = Path("checkpoints/magic_estimator.pt")


class _Timeout(BaseException):
    pass


def load_estimator():
    import sys
    sys.path.insert(0, "scripts")
    from train_magic_estimator import Estimator
    payload = torch.load(CKPT, weights_only=False)
    m = Estimator()
    m.load_state_dict(payload["state_dict"])
    m.eval()
    return m, payload["mu"], payload["sd"]


def main(n_per: int, flat: int) -> None:
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    model, mu, sd = load_estimator()
    x = sp.Symbol("x")
    probs = []
    for level in (3, 4, 5):
        for i in range(n_per):
            p = make_integrate(level, 900_000 + i)  # fresh stream
            probs.append((level, sp.Integral(p._expr, x)))
    feats = torch.tensor([featurize(r) for _, r in probs],
                         dtype=torch.float32)
    with torch.no_grad():
        ls, lc = model((feats - mu) / sd)
    psolve = torch.sigmoid(ls).tolist()
    cost = lc.tolist()
    total = flat * len(probs)
    want = [50.0 if psolve[i] < 0.2 else
            min(max(2.0 ** cost[i] * 4, 50.0), 600.0)
            for i in range(len(probs))]
    scale = (total - 0) / sum(want)
    alloc = [max(25, int(w * scale)) for w in want]
    print(f"# budget race — n={len(probs)} total {total} nodes/arm; "
          f"alloc range [{min(alloc)}, {max(alloc)}]")
    res = {"flat": 0, "alloc": 0}
    spent = {"flat": 0, "alloc": 0}
    for i, (level, root) in enumerate(probs):
        row = {}
        for arm, b in (("flat", flat), ("alloc", alloc[i])):
            signal.alarm(240)
            try:
                r = solve(root, budget=b)
                row[arm] = r.solved
                spent[arm] += r.nodes
            except (_Timeout, Exception):
                row[arm] = False
                spent[arm] += b
            finally:
                signal.alarm(0)
            res[arm] += row[arm]
        if row["flat"] != row["alloc"]:
            print(f"{i:>3} L{level} flat={row['flat']} "
                  f"alloc={row['alloc']} (b={alloc[i]})", flush=True)
    print(f"TOTALS: flat {res['flat']}/{len(probs)} "
          f"(nodes {spent['flat']}), alloc {res['alloc']}/{len(probs)} "
          f"(nodes {spent['alloc']})")
    print("bar: alloc solves strictly more at equal total budget")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=40)
    ap.add_argument("--flat", type=int, default=200)
    a = ap.parse_args()
    main(a.n_per, a.flat)
