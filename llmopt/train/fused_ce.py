"""Fused (chunked) cross-entropy for MLX — the Liger-style trick.

The biggest single object in our training runs is not weights or
activations, it is the logits: (tokens, 151936) at fp32 is ~600KB per
token — a 16k-token batch materializes ~9.3GB just to take a softmax.
This module never builds that tensor. Tokens are processed in chunks:
each chunk's logits (chunk, V) are formed, reduced to a loss
contribution, and the backward recomputes the chunk's softmax to emit
grads — so peak memory is O(chunk * V) instead of O(N * V).

`mx.custom_function` carries the custom VJP: the backward trades a
second pass of chunk matmuls for the memory. Loss is mean over
non-ignored targets (ignore_index=-100, the answer-tokens-only
convention from train/lora.py).

Two scars from v1 (measured 2026-07-13, 8k tokens, c=1024):
- NO mx.eval INSIDE a custom_function under a grad transform. The
  body runs at graph-construction time, so each internal eval forces
  the half-built outer graph — 41s and 52GB peak (worse than naive on
  both axes, OOM-killed at 16k). Removing them: 3.1s. Laziness is the
  memory bound now (a few chunks in flight, constant in N), not one
  chunk exactly — accepted.
- No dense onehot in the vjp. (chunk, V) fp32 onehot rebuilt the very
  tensor the module exists to avoid, three times over per chunk;
  scatter the -1 into the softmax with put_along_axis instead, and do
  the two grad matmuls in fp16 (12.4GB -> 8.5GB, 3.1s -> 2.5s).

Honest accounting: the fused path does ~2x the matmul FLOPs of naive
(recompute in vjp). The bar in scripts/bench_fused_ce.py reports both
peak MB and tok/s — if the tok/s loss outweighs the memory win at our
shapes, that verdict gets recorded too.
"""
from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx

IGNORE_INDEX = -100


def naive_ce(hidden: mx.array, weight: mx.array,
             targets: mx.array) -> mx.array:
    """Reference: full logits, mean CE over non-ignored targets."""
    logits = (hidden @ weight.T).astype(mx.float32)
    lse = mx.logsumexp(logits, axis=-1)
    picked = mx.take_along_axis(
        logits, mx.maximum(targets, 0)[:, None], axis=-1)[:, 0]
    keep = (targets != IGNORE_INDEX).astype(mx.float32)
    n = mx.maximum(keep.sum(), 1.0)
    return ((lse - picked) * keep).sum() / n


def _make_fused(c: int):
    @mx.custom_function
    def fused(hidden, weight, targets):
        total = mx.array(0.0, dtype=mx.float32)
        count = mx.array(0.0, dtype=mx.float32)
        for i in range(0, hidden.shape[0], c):
            h_c, t_c = hidden[i:i + c], targets[i:i + c]
            logits = (h_c @ weight.T).astype(mx.float32)
            lse = mx.logsumexp(logits, axis=-1)
            picked = mx.take_along_axis(
                logits, mx.maximum(t_c, 0)[:, None], axis=-1)[:, 0]
            keep = (t_c != IGNORE_INDEX).astype(mx.float32)
            total = total + ((lse - picked) * keep).sum()
            count = count + keep.sum()
        return total / mx.maximum(count, 1.0)

    @fused.vjp
    def fused_vjp(primals, cotangent, _outputs):
        hidden, weight, targets = primals
        n = mx.maximum(
            (targets != IGNORE_INDEX).sum(), 1).astype(mx.float32)
        scale = cotangent / n
        ghs = []
        g_w = mx.zeros(weight.shape, dtype=mx.float32)
        for i in range(0, hidden.shape[0], c):
            h_c, t_c = hidden[i:i + c], targets[i:i + c]
            logits = (h_c @ weight.T).astype(mx.float32)
            keep = (t_c != IGNORE_INDEX).astype(mx.float32)[:, None]
            d = mx.softmax(logits, axis=-1)
            idx = mx.maximum(t_c, 0)[:, None]
            d = mx.put_along_axis(
                d, idx, mx.take_along_axis(d, idx, axis=-1) - 1.0,
                axis=-1)
            d = (d * (keep * scale)).astype(hidden.dtype)
            ghs.append(d @ weight)
            g_w = g_w + (d.T @ h_c).astype(mx.float32)
        return (mx.concatenate(ghs), g_w.astype(weight.dtype),
                mx.zeros_like(targets))

    return fused


_CACHE: dict[int, Callable[..., Any]] = {}


def fused_ce(hidden: mx.array, weight: mx.array, targets: mx.array,
             chunk: int = 1024) -> mx.array:
    """Chunked CE: same value/grads as naive_ce, O(chunk*V)-class peak."""
    if chunk not in _CACHE:
        _CACHE[chunk] = _make_fused(chunk)
    return _CACHE[chunk](hidden, weight, targets)
