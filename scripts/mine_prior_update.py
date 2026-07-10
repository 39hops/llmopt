"""Re-mine the markov prior after adding rules (2026-07-10).

Why: MarkovPrior's median-smoothing gives unseen rules 0.01*median
unigram mass, but at nodes with a bigram table the seen rules score
full bigram counts — propose_k=3 guillotines any new rule anyway
(measured twice: i_unprod in the docstring, i_log_power/i_transcend_div
in the frontier-gap re-run: full-enum solves in 4 plies, solve() never
proposes the closed form). Fix: harvest winning histories with a
FULL-ENUM beam (proposer=None, every kid kept) on fresh train-band
seeds, merge the counts into checkpoints/markov_prior.json.

Draws go through fork isolation (make_integrate on L4/L5 — sympy
pathology #7, see CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import signal
from collections import Counter, defaultdict
from pathlib import Path

import sympy as sp

from llmopt.mathgen.problems import make_integrate
from llmopt.search.derivation import beam_search
from llmopt.search.magic import is_dead

X = sp.Symbol("x")
PRIOR = Path("checkpoints/markov_prior.json")


class _Timeout(BaseException):
    pass


def main(n_per: int, seed_base: int, wall: int) -> None:
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    ctx = mp.get_context("fork")

    def _draw(chunk, q):
        q.put([sp.sstr(make_integrate(lv, sd)._expr) for lv, sd in chunk])

    jobs = [(lv, seed_base + i) for lv in (3, 4, 5) for i in range(n_per)]
    probs: list[sp.Expr] = []
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
            probs += [sp.sympify(s) for s in q.get(timeout=10)]
        except Exception:
            continue

    unigram: Counter = Counter()
    bigram: dict[str, Counter] = defaultdict(Counter)
    wins = 0
    for i, g in enumerate(probs):
        signal.alarm(wall)
        try:
            r = beam_search(sp.Integral(g, X), width=4, max_plies=24,
                            max_nodes=300, verify_p=0.1,
                            state_filter=lambda s: not is_dead(s))
        except (_Timeout, Exception):
            continue
        finally:
            signal.alarm(0)
        if not r.solved:
            continue
        wins += 1
        prev = None
        for lab in r.state.history:
            rule = lab.split("@")[0]
            unigram[rule] += 1
            if prev is not None:
                bigram[prev][rule] += 1
            prev = rule
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(probs)}] wins={wins}", flush=True)

    print(f"harvested {wins}/{len(probs)} wins; "
          f"new-rule fires: i_log_power={unigram['i_log_power']} "
          f"i_transcend_div={unigram['i_transcend_div']}")
    d = json.loads(PRIOR.read_text())
    for r_, c in unigram.items():
        d["unigram"][r_] = d["unigram"].get(r_, 0) + c
    for p_, tab in bigram.items():
        dst = d["bigram"].setdefault(p_, {})
        for r_, c in tab.items():
            dst[r_] = dst.get(r_, 0) + c
    PRIOR.write_text(json.dumps(d))
    print(f"merged into {PRIOR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=60)
    ap.add_argument("--seed-base", type=int, default=940_000)
    ap.add_argument("--wall", type=int, default=90)
    a = ap.parse_args()
    main(a.n_per, a.seed_base, a.wall)
