"""Per-state syndrome-policy labels (the qLDPC decoder, generalized
from the root to EVERY node of the winning derivation).

For each solved problem: re-solve, then replay the winning history —
at each ply, record (features of the state, rule-fire syndromes of
its first unsolved Integral, previous rule) -> the rule actually
applied. One solved problem yields plies-many training pairs.

Replay follows the recorded labels through successors() (same-label
children: first match is fine for policy purposes — the RULE is the
label; replay_verify's backtracking rigor is for soundness, not
needed here). verify_p=0: we trust a history that already replayed
at solve time.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import sympy as sp

from llmopt.search.derivation import State, successors
from llmopt.search.engine import solve
from llmopt.search.features import featurize
from llmopt.search.rules import INT_RULES

WALL = 180
X = sp.Symbol("x")


def _syndromes(expr: sp.Expr) -> list[float]:
    ints = list(expr.atoms(sp.Integral))
    if not ints:
        return [0.0] * len(INT_RULES)
    node = max(ints, key=sp.count_ops)  # the main open subproblem
    out = []
    for _, rule in INT_RULES:
        try:
            out.append(1.0 if rule(node) else 0.0)
        except Exception:
            out.append(0.0)
    return out


def _worker(row: dict, q: "mp.Queue") -> None:
    root = sp.Integral(sp.sympify(row["integrand"]), X)
    res = solve(root, budget=200)
    pairs = []
    if res.solved:
        s = State(root)
        prev = "<start>"
        for lab in res.state.history:
            feats = featurize(s.expr) + _syndromes(s.expr)
            pairs.append({"seed": row["seed"], "level": row["level"],
                          "features": feats, "prev": prev,
                          "rule": lab.split("@")[0]})
            nxt = next((c for l, c in successors(s, use_macros=True,
                                                 verify_p=0.0)
                        if l == lab), None)
            if nxt is None:
                pairs = pairs[:-1]  # replay diverged: keep clean prefix
                break
            s = nxt
            prev = lab.split("@")[0]
    q.put(pairs)


def main(labels: Path, out: Path, workers: int,
         include_unsolved: bool = False) -> None:
    # include_unsolved: stored solved-flags are stale w.r.t. the
    # current engine (2026-07-10: three new rules + width 3) — the
    # newly-solvable families are exactly the ones the policy must
    # learn, and the re-solve inside _worker decides anyway.
    rows = [json.loads(l) for l in labels.read_text().splitlines()]
    if not include_unsolved:
        rows = [r for r in rows if r["solved"]]
    print(f"{len(rows)} problems to replay", flush=True)
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
                        for pair in q.get(timeout=10):
                            f.write(json.dumps(pair) + "\n")
                            n += 1
                        f.flush()
                        if n and n % 1000 < 8:
                            print(f"[{n} pairs]", flush=True)
                    except Exception:
                        pass
                del running[p]
    print(f"wrote {n} pairs -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path,
                    default=Path("data/magic_labels_all.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/policy_labels.jsonl"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--include-unsolved", action="store_true")
    a = ap.parse_args()
    main(a.labels, a.out, a.workers, a.include_unsolved)
