"""Linear CKA: how similar are two sets of representations?

CKA(X, Y) = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F), columns
centered. Invariant to orthogonal transforms and isotropic scaling —
the right tool for "which layers compute the same thing" questions:
adjacent-layer CKA near 1 marks redundant depth (prune / early-exit
candidates); a sharp drop marks where representations reorganize.
"""

from __future__ import annotations


def linear_cka(x, y) -> float:
    """x: [n, d1], y: [n, d2] — same n examples, any widths."""
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    xty = (y.T @ x).pow(2).sum()
    xtx = (x.T @ x).pow(2).sum().sqrt()
    yty = (y.T @ y).pow(2).sum().sqrt()
    return float(xty / (xtx * yty + 1e-12))


def layer_cka_matrix(model, ids):
    """[L+1, L+1] pairwise linear CKA over hidden states (positions are
    the examples)."""
    import torch

    with torch.inference_mode():
        hs = model(
            input_ids=torch.tensor([list(ids)], device=model.device),
            output_hidden_states=True,
        ).hidden_states
        reps = [h[0].float() for h in hs]  # [seq, hidden] each
        n = len(reps)
        m = torch.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                m[i, j] = m[j, i] = linear_cka(reps[i], reps[j])
    return m
