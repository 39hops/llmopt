"""Dispatcher-net labels: which brain wins each problem (2026-07-10,
chasing the router's oracle ceiling — 127/130 vs threshold's 124).

The threshold router dispatches on HOW HARD (estimator cost head);
the autopsy showed the real question is WHICH BRAIN — hardness and
brain-affinity correlate but diverge exactly on the problems that
matter. Run BOTH arms per problem, label = the dominating arm by
(solved, then wall) — the FA Law as a label definition. Every
problem yields a label, not just the ~6% disagreements.

Features: root featurize + root rule-fire syndromes (same recipe as
the policy — the syndrome bits are what see i_transcend_div fires
that the cost head reads as mere hardness).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import signal
import time
from pathlib import Path

import sympy as sp

from llmopt.mathgen.problems import make_integrate
from llmopt.search.derivation import beam_search
from llmopt.search.engine import MarkovPrior, SyndromePolicy
from llmopt.search.features import featurize
from llmopt.search.magic import is_dead
from llmopt.search.rules import INT_RULES

X = sp.Symbol("x")
WALL = 130


def _syndromes(expr: sp.Expr) -> list[float]:
    ints = list(expr.atoms(sp.Integral))
    if not ints:
        return [0.0] * len(INT_RULES)
    node = max(ints, key=sp.count_ops)
    out = []
    for _, rule in INT_RULES:
        try:
            out.append(1.0 if rule(node) else 0.0)
        except Exception:
            out.append(0.0)
    return out


def _worker(job, q):
    lv, sd = job
    root = sp.Integral(make_integrate(lv, sd)._expr, X)
    mk = MarkovPrior.load().proposer()
    pol = SyndromePolicy.load().proposer()
    row = {"level": lv, "seed": sd,
           "features": featurize(root) + _syndromes(root)}
    for arm, prop in (("markov", mk), ("policy", pol)):
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
            KeyboardInterrupt()))
        signal.alarm(120)
        t0 = time.time()
        ok = False
        try:
            ok = beam_search(root, width=3, max_plies=24, max_nodes=200,
                             proposer=prop, propose_k=3, use_macros=True,
                             verify_p=0.1,
                             state_filter=lambda s: not is_dead(s)).solved
        except BaseException:
            ok = False
        finally:
            signal.alarm(0)
        row[arm] = bool(ok)
        row[arm + "_t"] = round(time.time() - t0, 1)
    # dominating arm by (solved, then wall) — the FA Law as a label
    m = (row["markov"], -row["markov_t"])
    p = (row["policy"], -row["policy_t"])
    row["winner"] = "policy" if p >= m else "markov"
    q.put(row)


def main(n_per: int, seed_base: int, workers: int, out: Path) -> None:
    jobs = [(lv, seed_base + 1000 * lv + i) for lv in (3, 4, 5)
            for i in range(n_per)]
    ctx = mp.get_context("fork")
    pending = list(reversed(jobs))
    running, n = {}, 0
    with out.open("w") as f:
        while pending or running:
            while pending and len(running) < workers:
                j = pending.pop()
                q = ctx.Queue()
                pr = ctx.Process(target=_worker, args=(j, q))
                pr.start()
                running[pr] = (q, time.monotonic() + WALL * 2.5)
            time.sleep(0.2)
            for pr in list(running):
                q, dl = running[pr]
                if pr.is_alive() and time.monotonic() < dl:
                    continue
                if pr.is_alive():
                    pr.kill()
                    pr.join()
                else:
                    try:
                        f.write(json.dumps(q.get(timeout=10)) + "\n")
                        f.flush()
                        n += 1
                        if n % 50 == 0:
                            print(f"[{n} rows]", flush=True)
                    except Exception:
                        pass
                del running[pr]
    print(f"wrote {n} dispatch rows -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", type=Path,
                    default=Path("data/dispatch_labels.jsonl"))
    a = ap.parse_args()
    main(a.n_per, a.seed_base, a.workers, a.out)
