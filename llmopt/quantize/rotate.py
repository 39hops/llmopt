"""Rotate-then-quantize: the same weights, arranged for better numbers.

An orthogonal Q changes every value of W while W' = QW computes the
same function (apply Q^T at runtime, or absorb into adjacent layers at
deploy). QuaRot-style incoherence processing exploits this: rotation
smears outlier columns across all coordinates, so the per-row min-max
range that round-to-nearest must cover shrinks and the quantization
grid spends its levels where the mass is.

rotation_error measures the round-trip in the rotated basis:
||W - Q^T rtn(QW)||_F / ||W||_F, against the rotation=None baseline.
"""

from __future__ import annotations

import torch

from llmopt.quantize.methods import rtn


def hadamard(n: int) -> torch.Tensor:
    """Normalized Hadamard matrix (Sylvester), n a power of 2."""
    assert n > 0 and (n & (n - 1)) == 0, f"n={n} not a power of 2"
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return h / n**0.5


def random_orthogonal(n: int, seed: int = 0) -> torch.Tensor:
    """Q factor of a seeded Gaussian (Haar-ish random rotation)."""
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(n, n, generator=g))
    # fix signs so the distribution is uniform over rotations
    return q * torch.sign(torch.diagonal(r))


def rotation_error(
    w: torch.Tensor, bits: int, rotation: torch.Tensor | None = None
) -> float:
    """Relative Frobenius error of rtn quantization in the rotated
    basis (rotation applied to the row space: W' = W @ Q^T, so the
    per-row quantization ranges see the mixed coordinates)."""
    w = w.float()
    if rotation is None:
        back = rtn(w, bits)
    else:
        q = rotation.to(w.dtype)
        back = rtn(w @ q.T, bits) @ q
    return float((w - back).norm() / w.norm())
