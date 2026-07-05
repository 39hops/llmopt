"""Sequence packing: fill fixed-length training rows with multiple
documents instead of padding each to max length.

Greedy first-fit-decreasing bin packing, then per-row position_ids that
reset at document boundaries and a block-diagonal 4D attention mask so
packed documents cannot attend each other — packing must be invisible
to the loss.
"""

from __future__ import annotations

from typing import Sequence


def pack_greedy(lengths: Sequence[int], capacity: int) -> list[list[int]]:
    """First-fit-decreasing: returns bins of sequence indices."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    bins: list[tuple[int, list[int]]] = []  # (used, indices)
    for i in order:
        n = lengths[i]
        assert n <= capacity, f"sequence {i} longer than capacity"
        for j, (used, idx) in enumerate(bins):
            if used + n <= capacity:
                bins[j] = (used + n, idx + [i])
                break
        else:
            bins.append((n, [i]))
    return [idx for _, idx in bins]


def pack_batch(seqs: Sequence[Sequence[int]], capacity: int, pad_id: int = 0):
    """Pack token sequences into rows. Returns dict of tensors:
    input_ids [B, capacity], position_ids (reset per document),
    attention_mask [B, 1, capacity, capacity] block-diagonal causal
    (additive, 0 = attend), and doc_spans per row for loss masking.
    """
    import torch

    bins = pack_greedy([len(s) for s in seqs], capacity)
    b = len(bins)
    ids = torch.full((b, capacity), pad_id, dtype=torch.long)
    pos = torch.zeros((b, capacity), dtype=torch.long)
    neg = torch.finfo(torch.float32).min
    mask = torch.full((b, 1, capacity, capacity), neg)
    spans: list[list[tuple[int, int]]] = []

    for row, idx in enumerate(bins):
        cur = 0
        row_spans = []
        for i in idx:
            s = list(seqs[i])
            n = len(s)
            ids[row, cur : cur + n] = torch.tensor(s)
            pos[row, cur : cur + n] = torch.arange(n)
            tri = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            mask[row, 0, cur : cur + n, cur : cur + n] = torch.where(
                tri, neg, 0.0
            )
            row_spans.append((cur, cur + n))
            cur += n
        spans.append(row_spans)

    return {
        "input_ids": ids,
        "position_ids": pos,
        "attention_mask": mask,
        "doc_spans": spans,
    }
