"""Live race: syndrome-policy proposer vs markov prior (the current
zero-NN engine's brain) at identical beam config. Pre-registered
bar: policy arm solves >= markov arm on fresh problems (solve ties
broken by wall-time — the policy costs ms/node vs the prior's ~0,
so it must BUY something).

Problems: fresh streams (950_000+), L3/L4/L5. Both arms: width 2,
plies 24, budget 200, propose_k 3, verify_p 0.1, magic prune on.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import signal
import time
from pathlib import Path

import sympy as sp
import torch

from llmopt.mathgen.problems import make_integrate
from llmopt.search.derivation import beam_search
from llmopt.search.engine import MarkovPrior
from llmopt.search.features import featurize
from llmopt.search.magic import is_dead
from llmopt.search.rules import INT_RULES

X = sp.Symbol("x")


class _Timeout(BaseException):
    pass


def load_policy():
    p = torch.load("checkpoints/syndrome_policy.pt", weights_only=False)
    import sys
    sys.path.insert(0, "scripts")
    net = torch.nn.Sequential(
        torch.nn.Linear(len(p["mu"]), 96), torch.nn.ReLU(),
        torch.nn.Linear(96, 96), torch.nn.ReLU(),
        torch.nn.Linear(96, len(p["vocab"])))
    net.load_state_dict(p["state_dict"])
    net.eval()
    return net, p


def make_policy_proposer(net, p):
    vi = {r: i for i, r in enumerate(p["vocab"])}
    pidx = {r: i for i, r in enumerate(p["prevs"])}

    rule_names = p.get("synd_rules", [n for n, _ in INT_RULES])

    def proposer(state, kids):
        if not kids:
            return kids
        prev = (state.history[-1].split("@")[0] if state.history
                else "<start>")
        oh = [0.0] * len(p["prevs"])
        if prev in pidx:
            oh[pidx[prev]] = 1.0
        # syndromes FOR FREE: the search already evaluated every rule
        # to produce kids — a rule's bit is 1 iff it appears among the
        # children's labels (v1 recomputed all 14 rules per node and
        # ran 5x slower for the same solves)
        kid_rules = {lab.split("@")[0] for lab, _ in kids}
        synd = [1.0 if n in kid_rules else 0.0 for n in rule_names]
        feats = featurize(state.expr) + synd + oh
        x = (torch.tensor([feats], dtype=torch.float32) - p["mu"]) / p["sd"]
        with torch.no_grad():
            logits = net(x)[0]
        def score(lab):
            r = lab.split("@")[0]
            return logits[vi[r]].item() if r in vi else -50.0
        return sorted(kids, key=lambda c: -score(c[0]))

    return proposer


def main(n_per: int, budget: int) -> None:
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    net, payload = load_policy()
    pol = make_policy_proposer(net, payload)
    mk = MarkovPrior.load().proposer()
    # fresh problems, drawn fork-isolated (pathology #7 rule)
    ctx = mp.get_context("fork")

    def _draw(chunk, q):
        q.put([(lv, sp.sstr(make_integrate(lv, sd)._expr))
               for lv, sd in chunk])

    # 992_000: the old 950_000 band became the v5 estimator-label
    # sweep (2026-07-10) — keep race problems out of every label band
    jobs = [(lv, 992_000 + i) for lv in (3, 4, 5) for i in range(n_per)]
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
            probs += [(lv, sp.Integral(sp.sympify(s), X))
                      for lv, s in q.get(timeout=10)]
        except Exception:
            continue
    arms = {"markov": mk, "policy": pol}
    res = {a: {"solved": 0, "t": 0.0, "to": 0} for a in arms}
    print(f"# syndrome-policy race — n={len(probs)} budget={budget}")
    for i, (lv, root) in enumerate(probs):
        row = {}
        for arm, prop in arms.items():
            signal.alarm(180)
            t0 = time.time()
            try:
                r = beam_search(root, width=3, max_plies=24,
                                max_nodes=budget, proposer=prop,
                                propose_k=3, use_macros=True,
                                verify_p=0.1,
                                state_filter=lambda s: not is_dead(s))
                row[arm] = r.solved
            except (_Timeout, Exception):
                row[arm] = False
                res[arm]["to"] += 1
            finally:
                signal.alarm(0)
            res[arm]["solved"] += row[arm]
            res[arm]["t"] += time.time() - t0
        if row["markov"] != row["policy"]:
            print(f"{i:>3} L{lv} markov={row['markov']} "
                  f"policy={row['policy']}", flush=True)
    for arm, st in res.items():
        print(f"TOTALS {arm}: solved {st['solved']}/{len(probs)} "
              f"time {st['t']:.0f}s timeouts {st['to']}")
    print("bar: policy >= markov solves; ties -> less time")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=40)
    ap.add_argument("--budget", type=int, default=200)
    a = ap.parse_args()
    main(a.n_per, a.budget)
