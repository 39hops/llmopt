"""Entropy-bonus beam selection (pre-registered, its own race).

Physics motivation (banked 2026-07-09): mimicking high-magic states
with low magic costs entropy — when the eval can't buy progress,
spend the beam on DIVERSITY. Distinct from the annealing null
(random temperature noise lost to greedy): this is structured
diversity pressure, greedy max-min in the 20-feature space —
slot 1 = best eval, each later slot = best eval among the
top-pool candidates FARTHEST from the already-selected beam.

Arms (paired, one run, same problems): plain top-width beam vs
diversity-select, both zero-NN markov config at width 4.
Pre-registered bar: diversity must solve strictly more; ties on
solves broken by total nodes (lower = better). If it loses, the
annealing null generalizes and we say so.
"""

from __future__ import annotations

import argparse
import math
import signal

import sympy as sp

from llmopt.mathgen.problems import make_integrate
from llmopt.search.derivation import beam_search
from llmopt.search.engine import MarkovPrior
from llmopt.search.features import featurize
from llmopt.search.magic import is_dead


class _Timeout(BaseException):
    pass


def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def diversity_select(candidates, width):
    """Greedy max-min: seed with the eval-best, then repeatedly take
    the candidate (from the top 4*width eval pool — diversity must
    not rescue hopeless states) with the largest min-distance to the
    selected set. Feature vectors are the NNUE 20."""
    pool = candidates[:4 * width]
    if len(pool) <= width:
        return pool
    feats = {id(s): featurize(s.expr) for s in pool}
    chosen = [pool[0]]
    rest = pool[1:]
    while len(chosen) < width and rest:
        best_i, best_d = 0, -1.0
        for i, s in enumerate(rest):
            d = min(_dist(feats[id(s)], feats[id(c)]) for c in chosen)
            if d > best_d:
                best_i, best_d = i, d
        chosen.append(rest.pop(best_i))
    return chosen


def main(n: int, level: int, budget: int) -> None:
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    prior = MarkovPrior.load()
    x = sp.Symbol("x")
    arms = {"plain": None, "diverse": diversity_select}
    solved = {a: 0 for a in arms}
    nodes_t = {a: 0 for a in arms}
    tos = {a: 0 for a in arms}
    print(f"# entropy-beam race — int L{level} n={n} budget={budget} w=4")
    for i in range(n):
        p = make_integrate(level, 800_000 + i)  # disjoint stream
        root = sp.Integral(p._expr, x)
        row = {}
        for arm, sel in arms.items():
            signal.alarm(120)
            try:
                res = beam_search(
                    root, width=4, max_plies=24, max_nodes=budget,
                    proposer=prior.proposer(), propose_k=3,
                    use_macros=True, verify_p=0.1,
                    state_filter=lambda s: not is_dead(s),
                    select_fn=sel)
                row[arm] = (res.solved, res.nodes)
            except _Timeout:
                row[arm] = (False, budget)
                tos[arm] += 1
            except Exception:
                row[arm] = (False, budget)
            finally:
                signal.alarm(0)
            solved[arm] += row[arm][0]
            nodes_t[arm] += row[arm][1]
        print(f"{i:>3} plain={row['plain']} diverse={row['diverse']}",
              flush=True)
    print("TOTALS")
    for arm in arms:
        print(f"  {arm:>7}: solved {solved[arm]}/{n}  "
              f"nodes {nodes_t[arm]}  timeouts {tos[arm]}")
    print("bar: diverse solves strictly more (ties -> fewer nodes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--budget", type=int, default=300)
    a = ap.parse_args()
    main(a.n, a.level, a.budget)
