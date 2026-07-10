"""Policy-gated expansion race: does skipping un-predicted rule
evaluations buy wall-time without costing solves?

Arms (identical except the gate): markov proposer, full expansion
vs markov proposer + expand_rules = gate's top-k rule names.
Gate misses fall back to full expansion inside beam_search (a miss
costs one wasted gated expansion, never a solution).
Pre-registered bar: gated wall-time < full at equal solves; any
solve regression kills it regardless of speed.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
import time

import sympy as sp
import torch

from llmopt.mathgen.problems import make_integrate
from llmopt.search.derivation import beam_search
from llmopt.search.engine import MarkovPrior
from llmopt.search.features import featurize
from llmopt.search.magic import is_dead

X = sp.Symbol("x")


class _Timeout(BaseException):
    pass


def make_gate(k: int):
    p = torch.load("checkpoints/rule_gate.pt", weights_only=False)
    net = torch.nn.Sequential(
        torch.nn.Linear(len(p["mu"]), 96), torch.nn.ReLU(),
        torch.nn.Linear(96, 96), torch.nn.ReLU(),
        torch.nn.Linear(96, len(p["vocab"])))
    net.load_state_dict(p["state_dict"])
    net.eval()
    pidx = {r: i for i, r in enumerate(p["prevs"])}

    def gate(state):
        prev = (state.history[-1].split("@")[0] if state.history
                else "<start>")
        oh = [0.0] * len(p["prevs"])
        if prev in pidx:
            oh[pidx[prev]] = 1.0
        feats = featurize(state.expr) + oh
        x = (torch.tensor([feats], dtype=torch.float32)
             - p["mu"]) / p["sd"]
        with torch.no_grad():
            logits = net(x)[0]
        top = logits.topk(min(k, len(p["vocab"]))).indices.tolist()
        return {p["vocab"][i] for i in top}

    return gate


def main(n_per: int, budget: int, k: int) -> None:
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    mk = MarkovPrior.load().proposer()
    gate = make_gate(k)
    ctx = mp.get_context("fork")

    def _draw(chunk, q):
        q.put([(lv, sp.sstr(make_integrate(lv, sd)._expr))
               for lv, sd in chunk])

    jobs = [(lv, 960_000 + i) for lv in (3, 4, 5) for i in range(n_per)]
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
    arms = {"full": None, "gated": gate}
    res = {a: {"solved": 0, "t": 0.0, "to": 0} for a in arms}
    print(f"# gated-expansion race — n={len(probs)} budget={budget} "
          f"k={k}")
    for i, (lv, root) in enumerate(probs):
        row = {}
        for arm, g in arms.items():
            signal.alarm(180)
            t0 = time.time()
            try:
                r = beam_search(root, width=2, max_plies=24,
                                max_nodes=budget, proposer=mk,
                                propose_k=3, use_macros=True,
                                verify_p=0.1,
                                state_filter=lambda s: not is_dead(s),
                                expand_rules=g)
                row[arm] = r.solved
            except (_Timeout, Exception):
                row[arm] = False
                res[arm]["to"] += 1
            finally:
                signal.alarm(0)
            res[arm]["solved"] += row[arm]
            res[arm]["t"] += time.time() - t0
        if row["full"] != row["gated"]:
            print(f"{i:>3} L{lv} full={row['full']} gated={row['gated']}",
                  flush=True)
    for arm, st in res.items():
        print(f"TOTALS {arm}: solved {st['solved']}/{len(probs)} "
              f"time {st['t']:.0f}s timeouts {st['to']}")
    print("bar: gated time < full at equal solves; regression kills it")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=40)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    main(a.n_per, a.budget, a.k)
