"""Routing-masked expert pruning for MLX MoE models.

Given router utilization stats (moe/router_stats.py, saved by
scripts/moe_router_stats.py), derive per-layer keep-sets and *mask* the
router: dropped experts get -inf gate logits, so the softmax renormalizes
over survivors and the router can never select a pruned expert.

Masking rather than physically deleting weights keeps the experiment
honest and reversible — the model runs exactly as a pruned model would
route, while one flag flip restores the full model. Memory savings are
reported as the keep fraction (physical slicing of the stacked
quantized expert tensors is a later step, only worth it if quality
holds).
"""

from __future__ import annotations

import json
from pathlib import Path

from llmopt.moe.router_stats import RouterStats


def stats_from_json(path: str | Path) -> tuple[RouterStats, RouterStats, int]:
    """Load (math_stats, general_stats, n_experts) saved by
    scripts/moe_router_stats.py. JSON keys arrive as strings."""
    raw = json.loads(Path(path).read_text())
    n = raw["n_experts"]

    def build(section):
        s = RouterStats(n_experts=n)
        s.counts = {int(k): v for k, v in section["counts"].items()}
        s.mass = {int(k): v for k, v in section["mass"].items()}
        return s

    return build(raw["math"]), build(raw["general"]), n


def keep_sets(
    stats: RouterStats, criterion: str, threshold: float = 0.99
) -> dict[int, set[int]]:
    return {
        layer: stats.keep_set(layer, criterion, threshold)
        for layer in sorted(stats.counts)
    }


def keep_fraction(keep: dict[int, set[int]], n_experts: int) -> float:
    """Mean fraction of experts kept across layers — the (idealized)
    expert-memory saving is 1 minus this."""
    if not keep:
        return 1.0
    return sum(len(s) for s in keep.values()) / (len(keep) * n_experts)


def mask_router(model, keep: dict[int, set[int]], n_experts: int):
    """Patch every sparse-MoE block's class __call__ so pruned experts
    get -inf gate logits. Returns an `unmask()` restoring the original.

    Class-level patch (obj(x) dispatches via type(obj).__call__) with a
    per-instance mask registry, same pattern as the stats capture.
    """
    import mlx.core as mx

    moe_layers = [
        (i, layer.mlp)
        for i, layer in enumerate(model.model.layers)
        if hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "top_k")
    ]
    masks = {}
    for li, block in moe_layers:
        kept = keep.get(li, set(range(n_experts)))
        assert len(kept) >= block.top_k, (
            f"layer {li}: keeping {len(kept)} experts < top_k={block.top_k}; "
            "the router would be forced to select -inf experts (NaN)"
        )
        bias = [0.0 if e in kept else float("-inf") for e in range(n_experts)]
        masks[id(block)] = mx.array(bias)

    cls = type(moe_layers[0][1])
    original = cls.__call__

    def wrapped(self, x):
        logits = self.gate(x) + masks[id(self)]
        gates = mx.softmax(logits, axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)

    cls.__call__ = wrapped

    def unmask():
        cls.__call__ = original

    return unmask
