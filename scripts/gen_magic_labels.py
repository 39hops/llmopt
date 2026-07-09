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
WORKER_WALL = 300  # seconds. NOT 150: an honest UNSOLVED search at
# budget 200 can take >150s, and a tight wall silently eats exactly
# the negatives the solved-head needs (measured: L4 seed 700001,
# solved=False, walled at 150s but completes under 300)


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


def _estimator_order(jobs: list) -> "tuple[list, dict]":
    """Artin's active-labeling move (2026-07-09): the estimator
    improves its OWN label generation. Sort jobs by predicted cost
    ascending (cheap labels land first — the sweep becomes anytime:
    kill it whenever, you keep the most labels per hour) and give
    each job a predicted wall instead of the flat 300s (a problem
    predicted at 4 nodes doesn't deserve a 300s hang allowance).
    Walls only affect SKIPs, never label values — a mispredicted
    tight wall loses a label, it cannot corrupt one."""
    import torch
    import sys
    sys.path.insert(0, "scripts")
    from train_magic_estimator import Estimator
    payload = torch.load("checkpoints/magic_estimator.pt",
                         weights_only=False)
    m = Estimator(d_in=len(payload["mu"]))
    m.load_state_dict(payload["state_dict"])
    m.eval()
    feats = []
    for level, seed in jobs:
        p = make_integrate(level, seed)
        feats.append(featurize(sp.Integral(p._expr, sp.Symbol("x"))))
    xs = (torch.tensor(feats, dtype=torch.float32)
          - payload["mu"]) / payload["sd"]
    with torch.no_grad():
        _, cost = m(xs)
    cost = cost.tolist()
    order = sorted(range(len(jobs)), key=lambda i: cost[i])
    walls = {jobs[i]: int(min(max(30 * 2 ** max(cost[i], 0), 60), 300))
             for i in range(len(jobs))}
    return [jobs[i] for i in order], walls


def main(per_level: int, budget: int, out: Path, levels,
         seed_base: int = 700_000, guided: bool = False) -> None:
    global WORKER_WALL
    jobs = [(level, seed_base + s)
            for level in levels for s in range(per_level)]
    walls = {}
    if guided:
        jobs, walls = _estimator_order(jobs)
        print(f"guided: walls [{min(walls.values())}, "
              f"{max(walls.values())}]s, cheapest-first", flush=True)
    rows = 0
    with out.open("w") as f:
        for level, seed_abs in jobs:
                seed = seed_abs - seed_base
                WORKER_WALL = walls.get((level, seed_abs), WORKER_WALL)
                row = solve_isolated(level, seed_abs, budget)
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
    ap.add_argument("--seed-base", type=int, default=700_000)
    ap.add_argument("--guided", action="store_true")
    a = ap.parse_args()
    main(a.per_level, a.budget, a.out, a.levels, a.seed_base, a.guided)
