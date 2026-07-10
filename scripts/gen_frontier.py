"""Magic-maximizing generation (frontier mining): draw a large
candidate pool, score each with the estimator (microseconds), keep
the ones predicted HARD-BUT-SOLVABLE, and measure whether selection
actually concentrated difficulty.

Pre-registered claims vs a uniform control of the same size:
  1. selected batch's MEASURED mean log2-nodes > uniform's
  2. selected batch's solve rate lands nearer the frontier (~50%)
     than uniform's (which the L1-5 ladder predicts at ~80%+)
Also emits rule-gap candidates: selected problems that FAILED at
budget (predicted-solvable + measured-unsolved = tomorrow's autopsy).

Selection: p_solve in [0.25, 0.9] (frontier band), then top-N by
predicted cost. Screening uses the 20-feature estimator (us per
candidate; the rf tier costs ms x 14 rule calls and isn't needed to
rank a pool).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import sympy as sp
import torch

from llmopt.mathgen.problems import make_integrate
from llmopt.search.engine import solve
from llmopt.search.features import featurize

X = sp.Symbol("x")
WALL = 300


def _draw(chunk, q):
    out = []
    for lv, sd in chunk:
        p = make_integrate(lv, sd)
        e = p._expr
        out.append((lv, sd, sp.sstr(e),
                    featurize(sp.Integral(e, X))))
    q.put(out)


def _solve_worker(item, q):
    lv, sd, s = item
    root = sp.Integral(sp.sympify(s), X)
    r = solve(root, budget=200)
    import math
    q.put({"level": lv, "seed": sd, "integrand": s,
           "solved": bool(r.solved),
           "lognodes": math.log2(1 + r.nodes)})


def _solve_batch(items, workers):
    ctx = mp.get_context("fork")
    pending = list(reversed(items))
    running, out = {}, []
    while pending or running:
        while pending and len(running) < workers:
            it = pending.pop()
            q = ctx.Queue()
            p = ctx.Process(target=_solve_worker, args=(it, q))
            p.start()
            running[p] = (q, time.monotonic() + WALL)
        time.sleep(0.2)
        for p in list(running):
            q, dl = running[p]
            if p.is_alive() and time.monotonic() < dl:
                continue
            if p.is_alive():
                p.kill()
                p.join()
            else:
                try:
                    out.append(q.get(timeout=10))
                except Exception:
                    pass
            del running[p]
    return out


def main(pool: int, keep: int, workers: int, out: Path,
         seed_base: int = 970_000) -> None:
    payload = torch.load("checkpoints/magic_estimator.pt",
                         weights_only=False)
    import sys
    sys.path.insert(0, "scripts")
    from train_magic_estimator import Estimator
    net = Estimator(d_in=len(payload["mu"]))
    net.load_state_dict(payload["state_dict"])
    net.eval()

    ctx = mp.get_context("fork")
    jobs = [(lv, seed_base + i) for lv in (3, 4, 5)
            for i in range(pool // 3)]
    cands = []
    for i in range(0, len(jobs), 25):
        q = ctx.Queue()
        pr = ctx.Process(target=_draw, args=(jobs[i:i + 25], q))
        pr.start()
        pr.join(90)
        if pr.is_alive():
            pr.kill()
            pr.join()
            continue
        try:
            cands += q.get(timeout=10)
        except Exception:
            continue
    print(f"pool: {len(cands)} candidates drawn", flush=True)
    xs = (torch.tensor([c[3] for c in cands], dtype=torch.float32)
          - payload["mu"]) / payload["sd"]
    with torch.no_grad():
        ls, lc = net(xs)
    ps = torch.sigmoid(ls).tolist()
    cost = lc.tolist()
    band = [i for i in range(len(cands)) if 0.25 <= ps[i] <= 0.9]
    band.sort(key=lambda i: -cost[i])
    sel = band[:keep]
    import random
    rng = random.Random("frontier-0")
    uni = rng.sample(range(len(cands)), keep)
    print(f"band {len(band)}; selected {len(sel)}; measuring both "
          f"batches at {workers} workers", flush=True)

    res = {}
    for name, idxs in (("selected", sel), ("uniform", uni)):
        items = [(cands[i][0], cands[i][1], cands[i][2]) for i in idxs]
        rows = _solve_batch(items, workers)
        mean_ln = sum(r["lognodes"] for r in rows) / len(rows)
        rate = sum(r["solved"] for r in rows) / len(rows)
        res[name] = rows
        print(f"{name}: n={len(rows)} solve-rate {rate:.2f} "
              f"mean log2-nodes {mean_ln:.2f}", flush=True)
    gaps = [r for r in res["selected"] if not r["solved"]]
    out.write_text("\n".join(json.dumps(r) for r in gaps))
    print(f"{len(gaps)} rule-gap candidates -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=1800)
    ap.add_argument("--keep", type=int, default=90)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path,
                    default=Path("data/frontier_gaps.jsonl"))
    ap.add_argument("--seed-base", type=int, default=970_000)
    a = ap.parse_args()
    main(a.pool, a.keep, a.workers, a.out, a.seed_base)
