"""Dispatcher v4 adoption race: markov, policy, v3-routed, v4-routed
on a fresh L3-L8 band. Bar (the FA Law): v4 must match the best arm's
solves; wall breaks ties. Judgment-stack currency: v4 is the only
router trained on the post-orbital engine (v3 predates i_sqrt_basis's
log block and the trig(log) generators).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

CKPT = Path("checkpoints")


def _route(disp_path: Path, expr):
    """Replicates engine.solve's dispatcher gate (timeboxed probes)."""
    import sympy as sp
    import torch

    from llmopt.search.derivation import _timeboxed
    from llmopt.search.features import featurize
    from llmopt.search.rules import INT_RULES
    dp = torch.load(disp_path, weights_only=False)
    dnet = torch.nn.Sequential(
        torch.nn.Linear(len(dp["mu"]), 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 1))
    dnet.load_state_dict(dp["state_dict"])
    dnet.eval()
    node = max(expr.atoms(sp.Integral), key=sp.count_ops, default=None)
    by_name = dict(INT_RULES)
    synd = []
    for rname in dp.get("synd_rules", [n for n, _ in INT_RULES]):
        rule = by_name.get(rname)
        fired = (_timeboxed(rule, node, default=[])
                 if node is not None and rule else [])
        synd.append(1.0 if fired else 0.0)
    f = torch.tensor([featurize(expr) + synd], dtype=torch.float32)
    with torch.no_grad():
        return dnet((f - dp["mu"]) / dp["sd"]).item() > 0


def _worker(arm: str, level: int, seed: int, q: "mp.Queue") -> None:
    import sympy as sp

    from llmopt.mathgen.problems import make_integrate
    from llmopt.search.derivation import beam_search
    from llmopt.search.engine import MarkovPrior, SyndromePolicy
    from llmopt.search.magic import is_dead

    p = make_integrate(level, seed)
    root = sp.Integral(p._expr, sp.Symbol("x"))
    if arm == "markov":
        prop = MarkovPrior.load().proposer()
    elif arm == "policy":
        prop = SyndromePolicy.load().proposer()
    else:  # v3 / v4 routed
        path = CKPT / ("dispatcher_v3.pt" if arm == "v3"
                       else "dispatcher_v4.pt")
        prop = (SyndromePolicy.load().proposer() if _route(path, root)
                else MarkovPrior.load().proposer())
    t0 = time.monotonic()
    res = beam_search(root, width=3, max_plies=24, max_nodes=200,
                      proposer=prop, propose_k=3, use_macros=True,
                      verify_p=0.1,
                      state_filter=lambda s: not is_dead(s))
    q.put({"solved": bool(res.solved),
           "wall": round(time.monotonic() - t0, 2)})


def main(n_per: int, seed_base: int) -> None:
    ctx = mp.get_context("fork")
    arms = ("markov", "policy", "v3", "v4")
    tot = {a: [0, 0.0] for a in arms}  # solves, wall
    n = 0
    for level in (3, 4, 5, 6, 7, 8):
        for i in range(n_per):
            n += 1
            for arm in arms:
                q = ctx.Queue()
                pr = ctx.Process(target=_worker,
                                 args=(arm, level, seed_base + i, q))
                pr.start()
                pr.join(120)
                if pr.is_alive():
                    pr.kill()
                    pr.join()
                    tot[arm][1] += 120.0
                    continue
                try:
                    r = q.get(timeout=10)
                except Exception:
                    tot[arm][1] += 120.0
                    continue
                tot[arm][0] += r["solved"]
                tot[arm][1] += r["wall"]
            if n % 15 == 0:
                print("[%d] " % n + " ".join(
                    f"{a}={tot[a][0]}/{tot[a][1]:.0f}s"
                    for a in arms), flush=True)
    print(f"V4RACE n={n} (L3-L8 x {n_per})")
    for a in arms:
        print(f"  {a}: {tot[a][0]}/{n} in {tot[a][1]:.0f}s")
    print("bar: v4 >= all arms' solves; wall tiebreak (FA Law)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=7_500_000)
    a = ap.parse_args()
    main(a.n_per, a.seed_base)
