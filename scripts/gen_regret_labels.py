"""Regret/corrective labels (DAgger-style, Artin's 'make it regret
the wrong node' — hindsight credit assignment made mechanical).

Imitation's measured limit (RESULTS): the policy can't out-order the
teacher it imitates. The fix: collect states from the POLICY'S OWN
search trajectories (its visit distribution, including the places it
goes wrong), and label each with what the proven markov engine
concludes FROM THAT STATE (first rule of a fresh winning derivation
rooted there). Failures of the fresh solve yield no label (honest
skip). Corrections concentrate exactly where the policy errs.

Output rows match policy_labels.jsonl (features + prev + rule), tag
"source": "regret" — trainer mixes them with imitation pairs.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import sympy as sp

sys.path.insert(0, "scripts")

from llmopt.mathgen.problems import make_integrate
from llmopt.search.derivation import beam_search
from llmopt.search.engine import MarkovPrior, solve
from llmopt.search.features import featurize
from llmopt.search.magic import is_dead
from llmopt.search.rules import INT_RULES

X = sp.Symbol("x")
WALL = 240


def _syndromes(expr):
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
    """Policy-guided search over one problem; every VISITED state gets
    an expert relabel: solve() fresh from that state, take the first
    rule of the winning derivation."""
    from bench_syndrome_policy import load_policy, make_policy_proposer
    lv, sd = job
    p = make_integrate(lv, sd)
    root = sp.Integral(p._expr, X)
    net, payload = load_policy()
    prop = make_policy_proposer(net, payload)
    trace = []
    try:
        beam_search(root, width=3, max_plies=24, max_nodes=60,
                    proposer=prop, propose_k=3, use_macros=True,
                    verify_p=0.1, trace=trace,
                    state_filter=lambda s: not is_dead(s))
    except Exception:
        pass
    pairs = []
    seen = set()
    for st in trace[:12]:  # cap expert calls per problem
        k = st.key()
        if k in seen or not st.expr.has(sp.Integral):
            continue
        seen.add(k)
        try:
            r = solve(st.expr, budget=100)
        except Exception:
            continue
        if not (r.solved and r.state.history):
            continue
        prev = (st.history[-1].split("@")[0] if st.history
                else "<start>")
        pairs.append({"seed": sd, "level": lv, "source": "regret",
                      "features": featurize(st.expr)
                      + _syndromes(st.expr),
                      "prev": prev,
                      "rule": r.state.history[0].split("@")[0]})
    q.put(pairs)


def main(n_per: int, workers: int, out: Path) -> None:
    jobs = [(lv, 980_000 + i) for lv in (3, 4, 5) for i in range(n_per)]
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
                running[pr] = (q, time.monotonic() + WALL)
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
                        for pair in q.get(timeout=10):
                            f.write(json.dumps(pair) + "\n")
                            n += 1
                        f.flush()
                        if n and n % 500 < 12:
                            print(f"[{n} pairs]", flush=True)
                    except Exception:
                        pass
                del running[pr]
    print(f"wrote {n} regret pairs -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path,
                    default=Path("data/regret_labels.jsonl"))
    a = ap.parse_args()
    main(a.n_per, a.workers, a.out)
