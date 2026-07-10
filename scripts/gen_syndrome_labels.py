"""Syndrome-decoder labels (Artin's qLDPC riff, 2026-07-09): the
rule-fire bits are syndrome extraction (cheap local checks that
localize how a state deviates from the solvable subspace); a CODE
also decodes — syndrome pattern -> which correction to apply. Here:
re-solve known-solved problems recording the FIRST RULE of the
winning derivation, so a tiny net can learn syndrome -> opening move.

Parallel, fork-isolated (CLAUDE.md pathology-7 rule). Output rows:
existing label fields + "first_rule" + "opening" (first 3 labels).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import sympy as sp

from llmopt.search.engine import solve

WALL = 120  # solved problems are cheap; generous anyway


def _worker(row: dict, q: "mp.Queue") -> None:
    root = sp.Integral(sp.sympify(row["integrand"]), sp.Symbol("x"))
    res = solve(root, budget=200)
    if res.solved and res.state.history:
        out = dict(row)
        out["first_rule"] = res.state.history[0].split("@")[0]
        out["opening"] = [h.split("@")[0] for h in res.state.history[:3]]
        q.put(out)
    else:
        q.put(None)  # no longer solves (shouldn't happen) — drop


def main(labels: Path, out: Path, workers: int) -> None:
    rows = [json.loads(l) for l in labels.read_text().splitlines()
            if json.loads(l)["solved"]]
    print(f"{len(rows)} solved rows to decode", flush=True)
    ctx = mp.get_context("fork")
    pending = list(reversed(rows))
    running = {}
    n = 0
    with out.open("w") as f:
        while pending or running:
            while pending and len(running) < workers:
                r = pending.pop()
                q = ctx.Queue()
                p = ctx.Process(target=_worker, args=(r, q))
                p.start()
                running[p] = (q, time.monotonic() + WALL)
            time.sleep(0.2)
            for p in list(running):
                q, deadline = running[p]
                if p.is_alive() and time.monotonic() < deadline:
                    continue
                if p.is_alive():
                    p.kill()
                    p.join()
                else:
                    try:
                        row = q.get(timeout=10)
                        if row is not None:
                            f.write(json.dumps(row) + "\n")
                            f.flush()
                            n += 1
                            if n % 200 == 0:
                                print(f"[{n}]", flush=True)
                    except Exception:
                        pass
                del running[p]
    print(f"wrote {n} rows -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path,
                    default=Path("data/magic_labels_all_rf.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/syndrome_labels.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(a.labels, a.out, a.workers)
