"""Winning-path (state, legal moves, chosen move) triples for proposer
SFT. Every row is verifier-approved: it comes from a SOLVED search, so
the chosen move provably leads to a solution. Spec:
2026-07-07-move-proposer-design.md.

  python scripts/gen_proposer_data.py --per-cell 60 --split train
  python scripts/gen_proposer_data.py --per-cell 15 --split eval \
      --exclude-roots data/proposer_train_roots.json
"""

from __future__ import annotations

import argparse
import json
import random
import signal
from pathlib import Path

import sympy as sp

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import State, beam_search, successors

X = sp.Symbol("x")
WALL = 60


class _Timeout(BaseException):
    pass


def _root(rng, level, kind):
    if kind == "diff":
        return sp.Derivative(_expression(rng, level), X)
    while True:
        g = sp.simplify(sp.diff(_expression(rng, level), X))
        if g != 0:
            return sp.Integral(g, X)


def path_rows(root: sp.Expr) -> list[dict]:
    """Replay the winning history move-by-move, recording the legal
    alternatives at each ply. Skip plies whose label isn't found
    (shouldn't happen; belt and braces)."""
    signal.signal(signal.SIGALRM,
                  lambda s, f: (_ for _ in ()).throw(_Timeout()))
    signal.alarm(WALL)
    try:
        r = beam_search(root, max_plies=20, max_nodes=400)
    except _Timeout:
        return []
    finally:
        signal.alarm(0)
    if not r.solved:
        return []
    rows, cur = [], State(root)
    for chosen in r.state.history:
        kids = list(successors(cur))
        labels = [name for name, _ in kids]
        if chosen not in labels:
            return rows  # keep what we have
        idx = labels.index(chosen)
        if len(labels) > 1:  # single-choice plies teach nothing
            rows.append({"state": sp.sstr(cur.expr), "moves": labels,
                         "answer": idx})
        cur = kids[idx][1]
    return rows


def main(per_cell: int, split: str, exclude_file: str | None) -> None:
    exclude: set[str] = set()
    if exclude_file and Path(exclude_file).exists():
        exclude = set(json.loads(Path(exclude_file).read_text()))
    roots_seen: list[str] = []
    out_path = Path(f"data/proposer_{split}.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    n_rows = 0
    with out_path.open("w") as f:
        for kind in ("diff", "int"):
            for level in (1, 2, 3):
                rng = random.Random(f"proposer-{split}-{kind}-{level}-0")
                for _ in range(per_cell):
                    root = _root(rng, level, kind)
                    rk = sp.srepr(root)
                    if rk in exclude:
                        continue
                    roots_seen.append(rk)
                    for row in path_rows(root):
                        f.write(json.dumps(row) + "\n")
                        n_rows += 1
                print(f"{kind} L{level}: {n_rows} rows so far", flush=True)
    Path(f"data/proposer_{split}_roots.json").write_text(
        json.dumps(roots_seen))
    print(f"wrote {n_rows} rows to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=60)
    ap.add_argument("--split", default="train")
    ap.add_argument("--exclude-roots", default=None,
                    help="roots json from another split (contamination guard)")
    a = ap.parse_args()
    main(a.per_cell, a.split, a.exclude_roots)
