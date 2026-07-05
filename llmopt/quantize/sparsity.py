"""Pruning: unstructured magnitude and 2:4 semi-structured sparsity.

2:4 ("two of every four") keeps exactly the 2 largest-magnitude weights
in each contiguous group of 4 — the pattern NVIDIA sparse tensor cores
execute at ~2x dense throughput, unlike unstructured masks which save
memory but not time.
"""

from __future__ import annotations


def magnitude_prune(w, sparsity: float):
    """Zero the smallest-|w| fraction globally. Returns (w_pruned, mask)."""
    import torch

    k = int(w.numel() * sparsity)
    if k == 0:
        return w.clone(), torch.ones_like(w, dtype=torch.bool)
    thresh = w.abs().flatten().kthvalue(k).values
    mask = w.abs() > thresh
    return w * mask, mask


def prune_24(w):
    """2:4 semi-structured: keep top-2 |w| per group of 4 along the
    input dim. Returns (w_pruned, mask)."""
    import torch

    out_f, in_f = w.shape
    assert in_f % 4 == 0, "input dim must be divisible by 4"
    groups = w.reshape(out_f, in_f // 4, 4)
    idx = groups.abs().topk(2, dim=-1).indices
    mask = torch.zeros_like(groups, dtype=torch.bool).scatter_(-1, idx, True)
    mask = mask.reshape(out_f, in_f)
    return w * mask, mask
