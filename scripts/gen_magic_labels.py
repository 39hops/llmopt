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
import signal
from pathlib import Path

from llmopt.mathgen.problems import make_integrate
from llmopt.search.engine import solve
from llmopt.search.features import featurize
from llmopt.search.magic import is_dead
from llmopt.search.derivation import State


class _Timeout(BaseException):
    pass


def main(per_level: int, budget: int, out: Path) -> None:
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    rows = 0
    with out.open("w") as f:
        for level in (1, 2, 3, 4):
            for seed in range(per_level):
                p = make_integrate(level, 700_000 + seed)  # disjoint stream
                import sympy as sp
                root = sp.Integral(p._expr, sp.Symbol("x"))
                signal.alarm(120)
                try:
                    res = solve(root, budget=budget)
                    dead = is_dead(State(root))
                except _Timeout:
                    continue
                except Exception:
                    continue
                finally:
                    signal.alarm(0)
                row = {
                    "level": level, "seed": 700_000 + seed,
                    "integrand": sp.sstr(p._expr),
                    "features": featurize(root),
                    "solved": bool(res.solved),
                    "nodes": res.nodes,
                    "plies": len(res.state.history) if res.solved else 0,
                    "risch_dead": bool(dead),
                }
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
    a = ap.parse_args()
    main(a.per_level, a.budget, a.out)
