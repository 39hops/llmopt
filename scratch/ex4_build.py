"""EX4-UNIF subject builder (PRE-REG EX4-UNIF, RESULTS L23592;
ex3_build.py is results-cited/frozen — imported, never edited).

Builds, in order:
  checkpoints/ex4_del_unif0.json  full-128 minus 80 (layer, expert)
                                  slots drawn uniformly bank-wide
                                  (48x128), string seed "ex4-unif-0"
  checkpoints/ex4_del_unif1.json  independent second draw,
                                  string seed "ex4-unif-1"
  checkpoints/ex4_del_top80.json  full-128 minus the top-80 slots
                                  bank-wide by the pooled arm0
                                  demand counts (ties: lower layer,
                                  lower expert id first) — the same
                                  demand instrument that ordered the
                                  carriers' rank windows

Emission reuses ex3_build.emit: byte-identity assert against any
existing artifact, VERIFY_ONLY=1 refuses to write. Per-layer
deletion distributions printed for descriptive booking.

Usage: .venv/bin/python scratch/ex4_build.py        (build + verify)
       VERIFY_ONLY=1 ... (assert byte-identity, write nothing)
"""

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ex3_build import emit, pooled  # noqa: E402  (frozen, import-only)

N_DEL = 80
N_LAYERS, N_EXPERTS = 48, 128


def del_set(deleted):
    return {str(l): sorted(set(range(N_EXPERTS))
                           - {e for (ll, e) in deleted if ll == l})
            for l in range(N_LAYERS)}


def report(name, deleted):
    per_layer = Counter(l for (l, _) in deleted)
    n_layers = len(per_layer)
    mx = max(per_layer.values())
    print(f"{name}: {len(deleted)} deleted across {n_layers} layers "
          f"(max {mx} in one layer); per-layer "
          f"{dict(sorted(per_layer.items()))}")


def main():
    all_slots = [(l, e) for l in range(N_LAYERS)
                 for e in range(N_EXPERTS)]
    for j in (0, 1):
        rng = random.Random(f"ex4-unif-{j}")
        deleted = set(rng.sample(all_slots, N_DEL))
        assert len(deleted) == N_DEL
        report(f"ex4_del_unif{j}", deleted)
        emit(f"ex4_del_unif{j}", del_set(deleted))
    ranked = sorted(all_slots,
                    key=lambda le: (-pooled[le[0]][le[1]], le[0], le[1]))
    top = set(ranked[:N_DEL])
    report("ex4_del_top80", top)
    print(f"ex4_del_top80 demand range: "
          f"{pooled[ranked[0][0]][ranked[0][1]]} down to "
          f"{pooled[ranked[N_DEL - 1][0]][ranked[N_DEL - 1][1]]}")
    emit("ex4_del_top80", del_set(top))


if __name__ == "__main__":
    main()
