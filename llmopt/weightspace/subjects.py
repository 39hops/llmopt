"""Subject MLPs: tiny nets trained on generated functions, whose
weights become the *dataset* for the weight-space reader.

Each subject is a 1 -> 16 -> 16 -> 1 tanh MLP fit to one function drawn
from a family with string-seeded coefficients (CLAUDE.md convention).
The generating function is the oracle: the family label is known with
certainty. Subjects that fail to fit (MSE > 0.01) are rejected and
resampled — a diverged net is label noise, not a hard example.

Transforms for the permutation-symmetry ablation:
- permute_hidden: reorder hidden neurons (function-preserving — the
  symmetry the reader must cope with).
- canonicalize: sort hidden neurons by incoming-weight L2 norm, so
  every member of a permutation class maps to one representative.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

FAMILIES = ("sin", "poly2", "poly3", "gauss", "abslin", "tanh")

HIDDEN = 16
FIT_STEPS = 400
FIT_TOL = 0.01
_MAX_RESAMPLE = 8


def _target(family: str, coeffs: tuple, x: torch.Tensor) -> torch.Tensor:
    a, b, c = coeffs
    if family == "sin":
        return a * torch.sin(b * x + c)
    if family == "poly2":
        return a * x**2 + b * x + c
    if family == "poly3":
        return a * x**3 + b * x**2 + c * x
    if family == "gauss":
        return a * torch.exp(-b * x**2) + c
    if family == "abslin":
        return a * torch.abs(x) + b * x + c
    if family == "tanh":
        return a * torch.tanh(b * x) + c
    raise ValueError(family)


def _draw_coeffs(rng: random.Random) -> tuple:
    # magnitudes bounded away from zero so families stay distinguishable
    def draw():
        return round(rng.uniform(0.5, 2.0) * rng.choice((-1, 1)), 4)

    return (draw(), draw(), round(rng.uniform(-1.0, 1.0), 4))


def forward(weights: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    w1, b1, w2, b2, w3, b3 = weights
    h = torch.tanh(x @ w1.T + b1)
    h = torch.tanh(h @ w2.T + b2)
    return h @ w3.T + b3


@dataclass
class Subject:
    family: str
    coeffs: tuple
    weights: list[torch.Tensor]  # [W1 (16,1), b1, W2 (16,16), b2, W3 (1,16), b3]
    fit_mse: float


def _fit(family: str, coeffs: tuple, torch_seed: int) -> tuple[list[torch.Tensor], float]:
    gen = torch.Generator().manual_seed(torch_seed)
    dims = [(HIDDEN, 1), (HIDDEN,), (HIDDEN, HIDDEN), (HIDDEN,), (1, HIDDEN), (1,)]
    weights = [
        (torch.randn(d, generator=gen) * (0.5 / math.sqrt(d[-1]))).requires_grad_()
        for d in dims
    ]
    x = torch.linspace(-2, 2, 128)[:, None]
    y = _target(family, coeffs, x)
    scale = y.abs().mean().clamp(min=0.5)  # normalize so FIT_TOL is comparable
    opt = torch.optim.Adam(weights, lr=0.02)
    for _ in range(FIT_STEPS):
        loss = ((forward(weights, x) - y) / scale).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return [w.detach().clone() for w in weights], float(loss.detach())


def make_subject(family: str, i: int, seed: int) -> Subject:
    """Deterministic in (family, i, seed); resamples coefficients (and
    the init) until the fit converges."""
    for attempt in range(_MAX_RESAMPLE):
        rng = random.Random(f"{family}-{i}-{seed}-{attempt}")
        coeffs = _draw_coeffs(rng)
        weights, mse = _fit(family, coeffs, torch_seed=rng.randrange(2**31))
        if mse < FIT_TOL:
            return Subject(family, coeffs, weights, mse)
    raise RuntimeError(f"no converged fit for {family}-{i}-{seed}")


def make_dataset(
    n: int, seed: int, exclude: frozenset = frozenset()
) -> list[Subject]:
    """Round-robin over families. `exclude` is a frozenset of coeff
    tuples (from the other split) that must not reappear."""
    out: list[Subject] = []
    i = 0
    while len(out) < n:
        family = FAMILIES[len(out) % len(FAMILIES)]
        s = make_subject(family, i, seed)
        i += 1
        if s.coeffs in exclude:
            continue
        out.append(s)
    return out


def permute_hidden(
    weights: list[torch.Tensor], perm1: torch.Tensor, perm2: torch.Tensor
) -> list[torch.Tensor]:
    """Reorder hidden-layer neurons; the computed function is unchanged
    because each layer's outgoing columns are permuted to match."""
    w1, b1, w2, b2, w3, b3 = weights
    return [
        w1[perm1], b1[perm1],
        w2[perm2][:, perm1], b2[perm2],
        w3[:, perm2], b3,
    ]


def canonicalize(weights: list[torch.Tensor]) -> list[torch.Tensor]:
    """Sort each hidden layer's neurons by incoming-weight L2 norm
    (ties broken by bias). Maps every permutation-equivalent net to the
    same representative: canonicalize(permute(net)) == canonicalize(net).
    """
    w1, b1, w2, b2, w3, b3 = weights

    def order(w_in: torch.Tensor, b_in: torch.Tensor) -> torch.Tensor:
        key = torch.stack([w_in.pow(2).sum(1).sqrt(), b_in], dim=1)
        return torch.tensor(
            sorted(range(len(b_in)), key=lambda i: (key[i, 0].item(), key[i, 1].item()))
        )

    p1 = order(w1, b1)
    # layer-2 incoming weights depend on layer-1 order; apply p1 first
    w2_r = w2[:, p1]
    p2 = order(w2_r, b2)
    return [w1[p1], b1[p1], w2_r[p2], b2[p2], w3[:, p2], b3]
