"""Dispatcher v2 labels: disagreement-oversampled farming.

v1's null mechanism (RESULTS): dominance labels are 96/4
policy-skewed, the net degenerates to pure policy. Farming uniformly
costs ~16x per useful (markov-win) label. Fix the economics with a
two-stage farm: markov-wins only exist where the policy FAILS or is
SLOW, so run the policy arm on everything and spend markov runs only
there — fast policy solves are emitted (subsampled 10%, we have
1,141 already) WITHOUT the markov run.

Emitted rows match gen_dispatch_labels.py (features+syndromes,
per-arm solved/wall where measured, winner) plus "stage":
"policy-fast" (markov not run, winner=policy by construction) or
"dual" (true dominance label).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
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
FAST = 6.0     # policy solve under this wall = boring tie, subsample
KEEP_FAST = 0.1
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


def _run(root, prop):
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
    return bool(ok), round(time.time() - t0, 1)


def _worker(job, q):
    lv, sd = job
    root = sp.Integral(make_integrate(lv, sd)._expr, X)
    pol = SyndromePolicy.load().proposer()
    row = {"level": lv, "seed": sd,
           "features": featurize(root) + _syndromes(root)}
    p_ok, p_t = _run(root, pol)
    row["policy"], row["policy_t"] = p_ok, p_t
    rng = random.Random(f"dispatch-v2-{sd}")
    if p_ok and p_t < FAST and rng.random() > KEEP_FAST:
        q.put(None)  # boring tie, dropped (economics, not bias: the
        return       # kept 10% represent the class in training)
    if p_ok and p_t < FAST:
        row["stage"] = "policy-fast"
        row["winner"] = "policy"
        q.put(row)
        return
    mk = MarkovPrior.load().proposer()
    m_ok, m_t = _run(root, mk)
    row["markov"], row["markov_t"] = m_ok, m_t
    row["stage"] = "dual"
    row["winner"] = ("policy"
                     if (p_ok, -p_t) >= (m_ok, -m_t) else "markov")
    q.put(row)


def main(n_per: int, seed_base: int, workers: int, out: Path,
         levels: list[int] | None = None) -> None:
    jobs = [(lv, seed_base + 10_000 * lv + i)
            for lv in (levels or [3, 4, 5])
            for i in range(n_per)]
    ctx = mp.get_context("fork")
    pending = list(reversed(jobs))
    running, n, mkwins = {}, 0, 0
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
                        row = q.get(timeout=10)
                        if row is not None:
                            f.write(json.dumps(row) + "\n")
                            f.flush()
                            n += 1
                            mkwins += row["winner"] == "markov"
                            if n % 50 == 0:
                                print(f"[{n} rows, {mkwins} markov-wins]",
                                      flush=True)
                    except Exception:
                        pass
                del running[pr]
    print(f"wrote {n} rows ({mkwins} markov-wins) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=400)
    ap.add_argument("--seed-base", type=int, default=1_100_000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", type=Path,
                    default=Path("data/dispatch_labels_v2.jsonl"))
    ap.add_argument("--levels", type=int, nargs="+", default=None)
    a = ap.parse_args()
    main(a.n_per, a.seed_base, a.workers, a.out, a.levels)
