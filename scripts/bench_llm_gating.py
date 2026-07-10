"""LLM wall-time gating: the budget-allocation slot retargeted to
the currency that actually binds (RESULTS: node budget never binds —
the 5v3 timeout story says WALL TIME with LLM calls does).

Arms, each at the same fixed wall per problem:
  zero-nn : markov config (fast, no GPU)
  all-llm : hybrid config on every problem (best quality, slow)
  gated   : estimator decides — predicted-easy -> zero-nn,
            predicted-hard -> spend the LLM
Pre-registered bar: gated >= max(zero-nn, all-llm) solves at equal
wall. The estimator's job is to buy all-llm's wins on hard problems
without paying its timeouts on easy ones.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
import sys
import time

import sympy as sp
import torch

sys.path.insert(0, "scripts")

from llmopt.mathgen.problems import make_integrate
from llmopt.search.engine import solve
from llmopt.search.features import featurize

X = sp.Symbol("x")


class _Timeout(BaseException):
    pass


def main(n_per: int, wall: int, thresh: float) -> None:
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    from bench_hybrid import load_score_fn
    score_fn = load_score_fn()
    payload = torch.load("checkpoints/magic_estimator.pt",
                         weights_only=False)
    from train_magic_estimator import Estimator
    est = Estimator(d_in=len(payload["mu"]))
    est.load_state_dict(payload["state_dict"])
    est.eval()

    ctx = mp.get_context("fork")

    def _draw(chunk, q):
        q.put([(lv, sp.sstr(make_integrate(lv, sd)._expr))
               for lv, sd in chunk])

    jobs = [(lv, 990_100 + i) for lv in (3, 4, 5) for i in range(n_per)]
    probs = []
    for i in range(0, len(jobs), 20):
        q = ctx.Queue()
        pr = ctx.Process(target=_draw, args=(jobs[i:i + 20], q))
        pr.start()
        pr.join(120)
        if pr.is_alive():
            pr.kill()
            pr.join()
            continue
        try:
            probs += [(lv, sp.sympify(s)) for lv, s in q.get(timeout=10)]
        except Exception:
            continue

    feats = torch.tensor(
        [featurize(sp.Integral(g, X)) for _, g in probs],
        dtype=torch.float32)
    with torch.no_grad():
        _, cost = est((feats - payload["mu"]) / payload["sd"])
    cost = cost.tolist()
    n_llm = sum(c > thresh for c in cost)
    print(f"# llm-gating race — n={len(probs)} wall={wall}s "
          f"thresh={thresh} (gated sends {n_llm} to the LLM)")

    res = {a: {"solved": 0, "to": 0} for a in
           ("zero-nn", "all-llm", "gated")}
    for i, (lv, g) in enumerate(probs):
        root = sp.Integral(g, X)
        for arm in res:
            use_llm = (arm == "all-llm"
                       or (arm == "gated" and cost[i] > thresh))
            signal.alarm(wall)
            try:
                r = solve(root, budget=200,
                          llm_score_fn=score_fn if use_llm else None)
                ok = r.solved
            except (_Timeout, Exception):
                ok = False
                res[arm]["to"] += 1
            finally:
                signal.alarm(0)
            res[arm]["solved"] += ok
        if (i + 1) % 15 == 0:
            print(f"[{i+1}] " + " ".join(
                f"{a}={st['solved']}" for a, st in res.items()),
                flush=True)
    for arm, st in res.items():
        print(f"TOTALS {arm}: solved {st['solved']}/{len(probs)} "
              f"timeouts {st['to']}")
    print("bar: gated >= max(zero-nn, all-llm) at equal wall")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=20)
    ap.add_argument("--wall", type=int, default=30)
    ap.add_argument("--thresh", type=float, default=4.0)
    a = ap.parse_args()
    main(a.n_per, a.wall, a.thresh)
