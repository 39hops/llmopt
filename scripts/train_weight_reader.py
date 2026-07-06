"""Three-arm weight-space reader experiment (see the 2026-07-06 spec).

Generates oracle-labeled subject MLPs (process pool — each subject is
an independent ~0.5 s CPU fit), then trains the same reader on the same
data under three treatments of hidden-neuron permutation symmetry:

  raw        weights as trained
  canonical  neurons sorted by incoming-weight L2 norm (symmetry
             collapsed to one representative per class)
  augmented  random permutations as train-time augmentation

Chance floor is 1/6 = 16.7%. Prediction on record in the spec:
canonical >> augmented > raw.
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmopt.weightspace.reader import evaluate_reader, train_reader
from llmopt.weightspace.subjects import FAMILIES, make_subject

N_TRAIN, N_EVAL = 4000, 500
EPOCHS = 30
OUT = Path("checkpoints/weight_reader_results.json")


def _make(args):
    family, i, seed = args
    s = make_subject(family, i, seed)
    return s.family, s.coeffs, [w.tolist() for w in s.weights], s.fit_mse


def build(n, seed, exclude=frozenset()):
    import torch

    from llmopt.weightspace.subjects import Subject

    jobs = [(FAMILIES[j % len(FAMILIES)], j // len(FAMILIES), seed) for j in range(n)]
    out = []
    with ProcessPoolExecutor() as pool:
        for family, coeffs, weights, mse in pool.map(_make, jobs, chunksize=8):
            if tuple(coeffs) in exclude:
                continue
            out.append(Subject(family, tuple(coeffs),
                               [torch.tensor(w) for w in weights], mse))
    return out


def main() -> None:
    t0 = time.perf_counter()
    train = build(N_TRAIN, seed=0)
    banned = frozenset(s.coeffs for s in train)
    evals = build(N_EVAL, seed=1, exclude=banned)
    labels_t = [FAMILIES.index(s.family) for s in train]
    labels_e = [FAMILIES.index(s.family) for s in evals]
    print(f"subjects built in {time.perf_counter() - t0:.0f}s "
          f"(train {len(train)}, eval {len(evals)}, chance {1 / len(FAMILIES):.1%})")

    results = {}
    for arm, kw in (
        ("raw", dict(canonical=False, augment=False)),
        ("canonical", dict(canonical=True, augment=False)),
        ("augmented", dict(canonical=False, augment=True)),
    ):
        t0 = time.perf_counter()
        model = train_reader(train, labels_t, epochs=EPOCHS, seed=0, **kw)
        acc = evaluate_reader(model, evals, labels_e, canonical=kw["canonical"])
        results[arm] = acc
        print(f"  {arm:10s} eval acc {acc:6.1%}  "
              f"({time.perf_counter() - t0:.0f}s)")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
