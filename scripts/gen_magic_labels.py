"""Magic-estimator labels: (root features, ground-truth hardness).

The continuous version of the magic detector (RESULTS: Risch = the
domain's Gottesman-Knill, binary + exact). Here we label ROOT
problems with measured solve-cost so a tiny net can learn "how far
beyond the easy subspace is this state" — a difficulty oracle for
search ordering and frontier generation (expert iteration needs
problems at the model's edge; buckets are coarse, this is continuous).

Labels per problem (JSONL):
  features   : featurize(root)  (20 floats, the NNUE feature set)
  solved     : engine.solve verdict at fixed budget
  nodes      : nodes expanded (the honest cost)
  plies      : winning-derivation length (0 if unsolved)
  risch_dead : Risch-certified non-elementary root (exact magic bit)
  level/seed : provenance (string-seeded, stream disjoint from
               value_labels.jsonl which used the train_value_head
               seed streams)

Budget is deliberately modest (200): the label is "hardness at
standard budget", and unsolved-at-200 is itself signal.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import sympy as sp

from llmopt.mathgen.problems import make_integrate
from llmopt.search.derivation import State
from llmopt.search.engine import solve
from llmopt.search.features import featurize
from llmopt.search.magic import is_dead

# sympy pathology #7 (measured, first label sweep): one L4 problem
# hung 90 min in a loop that never delivered SIGALRM — signal-based
# walls have a hole no Python-level fix closes. Hard isolation:
# fork a worker per problem (sympy already imported, fork is cheap),
# join with a deadline, SIGKILL on overrun. The kill also discards
# any sympy cache corruption a pathological problem left behind.
WORKER_WALL = 150  # seconds; > the engine's own budget comfortably


def _worker(level: int, seed: int, budget: int, q: "mp.Queue") -> None:
    p = make_integrate(level, seed)
    root = sp.Integral(p._expr, sp.Symbol("x"))
    res = solve(root, budget=budget)
    q.put({
        "level": level, "seed": seed,
        "integrand": sp.sstr(p._expr),
        "features": featurize(root),
        "solved": bool(res.solved),
        "nodes": res.nodes,
        "plies": len(res.state.history) if res.solved else 0,
        "risch_dead": bool(is_dead(State(root))),
    })


def solve_isolated(level: int, seed: int, budget: int) -> "dict | None":
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(level, seed, budget, q))
    proc.start()
    proc.join(WORKER_WALL)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return None  # hung: no label, honest skip
    try:
        # NOT get_nowait: join() returning does not mean the child's
        # put() has crossed the pipe into the parent's queue thread
        # (measured: seeds that solved fine were mislabeled SKIP)
        return q.get(timeout=10)
    except Exception:
        return None  # worker crashed before reporting


def main(per_level: int, budget: int, out: Path, levels) -> None:
    rows = 0
    with out.open("w") as f:
        for level in levels:
            for seed in range(per_level):
                row = solve_isolated(level, 700_000 + seed, budget)
                if row is None:
                    print(f"L{level} seed {seed}: SKIP (hung/crashed)",
                          flush=True)
                    continue
                f.write(json.dumps(row) + "\n")
                f.flush()
                rows += 1
                if seed % 25 == 0:
                    print(f"L{level} seed {seed}: solved={row['solved']} "
                          f"nodes={row['nodes']}", flush=True)
    print(f"wrote {rows} rows -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-level", type=int, default=250)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--out", type=Path,
                    default=Path("data/magic_labels.jsonl"))
    ap.add_argument("--levels", type=int, nargs="+",
                    default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    main(a.per_level, a.budget, a.out, a.levels)
