"""Glue between RadixCache and HF KV caches: prefix reuse for prefill.

The radix tree stores opaque payloads; here the payload is a per-layer
list of (k, v) tensors [1, H, t, D] covering the edge's tokens. Three
operations make the tree usable as a prompt-prefix cache:

- slice_payload: cut [start, end) out of a finished prefill cache (the
  ``payload_fn`` for RadixCache.insert).
- split_payload: cut a payload in two (enables mid-edge splits when a
  new prompt diverges inside a cached edge).
- payloads_to_cache: concatenate matched payloads back into a
  DynamicCache the model can continue prefilling from.

Wired into BatchEngine via ``prefix_cache=RadixCache(split_payload=
split_payload)``: prefill skips the matched prefix entirely, and every
finished prefill is inserted back into the tree.
"""

from __future__ import annotations

from typing import Sequence

Payload = "list[tuple[torch.Tensor, torch.Tensor]]"  # per layer, [1, H, t, D]


def slice_payload(legacy_kv, start: int, end: int):
    """Clone [start, end) of every layer (clone: don't pin the source
    cache's full tensors alive through the tree)."""
    return [
        (k[:, :, start:end].clone(), v[:, :, start:end].clone())
        for k, v in legacy_kv
    ]


def split_payload(payload, n: int):
    """(head, tail) at token n. Views, not copies — both halves keep the
    same storage; the tree just tracks them separately."""
    head = [(k[:, :, :n], v[:, :, :n]) for k, v in payload]
    tail = [(k[:, :, n:], v[:, :, n:]) for k, v in payload]
    return head, tail


def payloads_to_cache(payloads: Sequence, upto: int | None = None):
    """Concatenate matched payloads (root-to-leaf order) into a
    DynamicCache; optionally truncate to the first ``upto`` tokens."""
    import torch
    from transformers import DynamicCache

    cache = DynamicCache()
    for layer in range(len(payloads[0])):
        k = torch.cat([p[layer][0] for p in payloads], dim=2)
        v = torch.cat([p[layer][1] for p in payloads], dim=2)
        if upto is not None:
            k, v = k[:, :, :upto], v[:, :, :upto]
        cache.update(k.clone(), v.clone(), layer)
    return cache
