"""Low-rank factorization: W [out, in] ~= A @ B with rank r.

Truncated SVD gives the Frobenius-optimal rank-r approximation
(Eckart-Young). Memory/compute drop from out*in to r*(out+in) — a win
when r << min(out, in), and the same structural move LoRA makes for
the *update* instead of the weight.
"""

from __future__ import annotations


def svd_factorize(w, rank: int):
    """Returns (a [out, r], b [r, in]); a @ b is the best rank-r
    approximation of w in Frobenius norm."""
    import torch

    u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
    root = s[:rank].sqrt()
    return u[:, :rank] * root, root[:, None] * vh[:rank]


def rank_error_curve(w, ranks) -> list[float]:
    """Relative Frobenius error per rank — how compressible is W?"""
    import torch

    total = float(torch.linalg.norm(w.float()))
    out = []
    for r in ranks:
        a, b = svd_factorize(w, r)
        out.append(float(torch.linalg.norm(w.float() - a @ b)) / total)
    return out
