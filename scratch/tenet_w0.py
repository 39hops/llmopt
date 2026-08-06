"""TENET W0: is reverse structure visible in weights? (pre-reg
TENET-W0, 2026-08-05 — the battery's cheapest rung.)

Train the certified WeightReader recipe (permutation-AUGMENTED, the
88.4% arm) on FORWARD subjects only, then evaluate on INVERSE TWINS:
subject MLPs fit to the same generated function with the data axes
SWAPPED (y -> x), i.e. the numerical inverse where one exists. Twins
inherit their SOURCE family label; the question is whether forward-
trained weight features survive interface reversal.

Same rejection rule as forward subjects (normalized MSE < FIT_TOL,
resample up to 8): non-invertible draws mostly fail to fit and drop
out — per-family eligible counts are REPORTED (a family with < 30
eligible twins books as ineligible, not as evidence).

FENCE (registered): "reverse" here is function-INVERSE — an analogy
probe for the reversed-token LM, not the same object. Null: inverse
accuracy at the 16.7% chance floor.

Usage: .venv/bin/python scratch/tenet_w0.py   [N_TRAIN=4000 N_EVAL=500
       N_INV=100 (per family) SEED=0 env overrides]
"""

import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from llmopt.weightspace.subjects import (FAMILIES, FIT_STEPS, FIT_TOL,
                                         HIDDEN, Subject, _draw_coeffs,
                                         _target, forward, make_dataset)
from llmopt.weightspace.reader import evaluate_reader, train_reader

N_TRAIN = int(os.environ.get("N_TRAIN", 4000))
N_EVAL = int(os.environ.get("N_EVAL", 500))
N_INV = int(os.environ.get("N_INV", 100))
SEED = int(os.environ.get("SEED", 0))


def _fit_xy(x_in, y_out, torch_seed):
    """The subjects._fit loop on arbitrary (input, output) data —
    identical hyperparameters, so twin fits are comparable."""
    gen = torch.Generator().manual_seed(torch_seed)
    dims = [(HIDDEN, 1), (HIDDEN,), (HIDDEN, HIDDEN), (HIDDEN,),
            (1, HIDDEN), (1,)]
    weights = [
        (torch.randn(d, generator=gen) * (0.5 / math.sqrt(d[-1])))
        .requires_grad_() for d in dims
    ]
    scale = y_out.abs().mean().clamp(min=0.5)
    opt = torch.optim.Adam(weights, lr=0.02)
    for _ in range(FIT_STEPS):
        loss = ((forward(weights, x_in) - y_out) / scale).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return [w.detach().clone() for w in weights], float(loss.detach())


def make_inverse_twin(family, i, seed):
    """Fit an MLP to the axis-swapped data of a fresh draw from
    `family` (eval-namespace string seed, disjoint from training)."""
    for attempt in range(8):
        rng = random.Random(f"inv-{family}-{i}-{seed}-{attempt}")
        coeffs = _draw_coeffs(rng)
        x = torch.linspace(-2, 2, 128)[:, None]
        y = _target(family, coeffs, x)
        # swapped axes: input = y (sorted for sanity), output = x
        order = torch.argsort(y[:, 0])
        weights, mse = _fit_xy(y[order], x[order],
                               torch_seed=rng.randrange(2**31))
        if mse < FIT_TOL:
            return Subject(family, coeffs, weights, mse)
    return None


def main():
    t0 = time.time()
    train = make_dataset(N_TRAIN, seed=SEED)
    excl = frozenset(s.coeffs for s in train)
    ev = make_dataset(N_EVAL, seed=SEED + 1, exclude=excl)
    print(f"[w0] subjects: {len(train)} train / {len(ev)} eval "
          f"({time.time() - t0:.0f}s)", flush=True)

    fam_idx = {f: i for i, f in enumerate(FAMILIES)}
    model = train_reader(train, [fam_idx[s.family] for s in train],
                         augment=True, seed=SEED)
    acc_fwd = evaluate_reader(model, ev, [fam_idx[s.family] for s in ev])
    print(f"[w0] forward held-out acc {acc_fwd:.4f} "
          f"(certified-recipe sanity; 2026-07-06 arm was 0.884)",
          flush=True)

    per_family = {}
    all_twins, all_labels = [], []
    for f in FAMILIES:
        twins = [t for i in range(N_INV)
                 if (t := make_inverse_twin(f, i, SEED)) is not None]
        per_family[f] = len(twins)
        all_twins += twins
        all_labels += [fam_idx[f]] * len(twins)
    print(f"[w0] inverse twins eligible per family: {per_family}",
          flush=True)
    eligible = {f: n for f, n in per_family.items() if n >= 30}
    keep = [(t, l) for t, l in zip(all_twins, all_labels)
            if per_family[t.family] >= 30]
    if keep:
        tw, lb = zip(*keep)
        acc_inv = evaluate_reader(model, list(tw), list(lb))
        print(f"[w0] INVERSE-TWIN acc {acc_inv:.4f} over "
              f"{len(tw)} twins in {len(eligible)} eligible families "
              f"(chance 0.167) | {time.time() - t0:.0f}s total",
              flush=True)
        for f in eligible:
            sub = [(t, l) for t, l in keep if t.family == f]
            a = evaluate_reader(model, [t for t, _ in sub],
                                [l for _, l in sub])
            print(f"[w0]   {f}: {a:.4f} (n={len(sub)})", flush=True)
    else:
        print("[w0] NO eligible families — books as ineligible-corpus",
              flush=True)


if __name__ == "__main__":
    main()
