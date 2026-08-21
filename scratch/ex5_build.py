"""EX5-LAYERMATCH subject builder (the missing control cell after
EX4-UNIF: fixed layer profile, rank-unrestricted identity).

ex3_build.py and ex4_build.py are results-cited/frozen — imported,
never edited. Two fresh 3-mask families, both per-layer-count
matched to the named 80 (ex3_inv_pooled) and both EXCLUDING the
named carrier slots (the mask census caught 1 named slot inside
ex4_del_unif1 and 2 inside ex4_del_top80 — enforced here):

  checkpoints/ex5_del_rank{0,1,2}.json   fresh rank-window-matched
      draws (the frozen ex3 rule: per named carrier, one
      non-carrier expert from its +-8 in-layer demand-rank window,
      draw order = sorted carrier list), string seeds
      "ex5-rank-{j}" — the family that ALSO matches rank class.
  checkpoints/ex5_del_layer{0,1,2}.json  layer-only draws: per
      layer, the named per-layer deletion count drawn uniformly
      from that layer's non-carrier experts, string seeds
      "ex5-layer-{j}" — holds layer placement, breaks rank.

Emission reuses ex3_build.emit: byte-identity assert against any
existing artifact, VERIFY_ONLY=1 refuses to write. Composition
facts print for the census crosswalk.

Usage: .venv/bin/python scratch/ex5_build.py        (build + verify)
       VERIFY_ONLY=1 ... (assert byte-identity, write nothing)
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ex3_build import emit, pooled  # noqa: E402  (frozen, import-only)
from ex4_build import del_set, report  # noqa: E402  (frozen, import-only)

N_LAYERS, N_EXPERTS = 48, 128

named = {tuple(x) for x in json.loads(
    Path("checkpoints/ex3_inv_pooled.json").read_text())}


def rank_family(j):
    rng = random.Random(f"ex5-rank-{j}")
    deleted = set()
    for (l, e) in sorted(named):
        order = sorted(range(N_EXPERTS),
                       key=lambda x: (-pooled[l][x], x))
        rank = order.index(e)
        window = [x for x in order[max(0, rank - 8):rank + 9]
                  if (l, x) not in named
                  and (l, x) not in deleted and x != e]
        deleted.add((l, rng.choice(window)))
    return deleted


def layer_family(j):
    rng = random.Random(f"ex5-layer-{j}")
    per_layer = {}
    for (l, _) in named:
        per_layer[l] = per_layer.get(l, 0) + 1
    deleted = set()
    for l in sorted(per_layer):
        pool = [e for e in range(N_EXPERTS) if (l, e) not in named]
        deleted |= {(l, e) for e in rng.sample(pool, per_layer[l])}
    return deleted


def main():
    for j in range(3):
        d = rank_family(j)
        assert len(d) == len(named) and not (d & named)
        report(f"ex5_del_rank{j}", d)
        emit(f"ex5_del_rank{j}", del_set(d))
    for j in range(3):
        d = layer_family(j)
        assert len(d) == len(named) and not (d & named)
        report(f"ex5_del_layer{j}", d)
        emit(f"ex5_del_layer{j}", del_set(d))


if __name__ == "__main__":
    main()
