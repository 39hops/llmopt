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


def solve_isolated(level: int, seed: int, budget: int,
                   wall: "int | None" = None) -> "dict | None":
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(level, seed, budget, q))
    proc.start()
    proc.join(wall or WORKER_WALL)
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


def _run_parallel(jobs, walls, budget, f, workers: int) -> int:
    """N isolated workers at once — labeling is embarrassingly
    parallel (fork-isolated, no shared state), and the sequential
    sweep was leaving ~8 cores idle. Each running job keeps its own
    deadline; overruns get the same SIGKILL discipline."""
    import time
    ctx = mp.get_context("fork")
    pending = list(reversed(jobs))
    running = {}  # proc -> (level, seed_abs, queue, deadline)
    rows = 0
    while pending or running:
        while pending and len(running) < workers:
            level, seed_abs = pending.pop()
            q = ctx.Queue()
            p = ctx.Process(target=_worker,
                            args=(level, seed_abs, budget, q))
            p.start()
            running[p] = (level, seed_abs, q,
                          time.monotonic()
                          + walls.get((level, seed_abs), WORKER_WALL))
        time.sleep(0.3)
        for p in list(running):
            level, seed_abs, q, deadline = running[p]
            if p.is_alive() and time.monotonic() < deadline:
                continue
            if p.is_alive():
                p.kill()
                p.join()
                print(f"L{level} seed {seed_abs}: SKIP (wall)",
                      flush=True)
            else:
                try:
                    row = q.get(timeout=10)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    rows += 1
                    if rows % 25 == 0:
                        print(f"[{rows}] L{level} seed {seed_abs}: "
                              f"solved={row['solved']}", flush=True)
                except Exception:
                    print(f"L{level} seed {seed_abs}: SKIP (crash)",
                          flush=True)
            del running[p]
    return rows


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
    # make_integrate itself can hang in simplify (pathology #7) and
    # SIGALRM does not reach it — feature the jobs in fork-isolated
    # chunks; a hung chunk is killed and its jobs get default
    # order/wall (lost speed, never lost soundness)
    ctx = mp.get_context("fork")

    def _chunk_feats(chunk, q):
        out = []
        for level, seed in chunk:
            p = make_integrate(level, seed)
            out.append(featurize(sp.Integral(p._expr, sp.Symbol("x"))))
        q.put(out)

    feats: dict = {}
    CH = 25
    for i in range(0, len(jobs), CH):
        chunk = jobs[i:i + CH]
        q = ctx.Queue()
        pr = ctx.Process(target=_chunk_feats, args=(chunk, q))
        pr.start()
        pr.join(60)
        if pr.is_alive():
            pr.kill()
            pr.join()
            continue  # chunk hung: its jobs keep default wall/order
        try:
            for job, fv in zip(chunk, q.get(timeout=10)):
                feats[job] = fv
        except Exception:
            continue
    known = [j for j in jobs if j in feats]
    unknown = [j for j in jobs if j not in feats]
    if not known:
        return jobs, {}
    xs = (torch.tensor([feats[j] for j in known], dtype=torch.float32)
          - payload["mu"]) / payload["sd"]
    with torch.no_grad():
        _, cost = m(xs)
    cost = cost.tolist()
    order = sorted(range(len(known)), key=lambda i: cost[i])
    walls = {known[i]: int(min(max(30 * 2 ** max(cost[i], 0), 60), 300))
             for i in range(len(known))}
    # unfeaturizable jobs go LAST with the default wall (they already
    # smell pathological)
    return [known[i] for i in order] + unknown, walls


def main(per_level: int, budget: int, out: Path, levels,
         seed_base: int = 700_000, guided: bool = False,
         workers: int = 1) -> None:
    jobs = [(level, seed_base + s)
            for level in levels for s in range(per_level)]
    walls = {}
    if guided:
        jobs, walls = _estimator_order(jobs)
        print(f"guided: walls [{min(walls.values())}, "
              f"{max(walls.values())}]s, cheapest-first", flush=True)
    with out.open("w") as f:
        if workers > 1:
            rows = _run_parallel(jobs, walls, budget, f, workers)
        else:
            rows = 0
            for level, seed_abs in jobs:
                row = solve_isolated(level, seed_abs, budget,
                                     walls.get((level, seed_abs)))
                if row is None:
                    print(f"L{level} seed {seed_abs}: SKIP", flush=True)
                    continue
                f.write(json.dumps(row) + "\n")
                f.flush()
                rows += 1
                if rows % 25 == 0:
                    print(f"[{rows}] L{level} seed {seed_abs}: "
                          f"solved={row['solved']}", flush=True)
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
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    main(a.per_level, a.budget, a.out, a.levels, a.seed_base, a.guided,
         a.workers)
