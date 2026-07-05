"""Radix-tree prefix KV cache (SGLang-style, pure Python structure).

The tree stores token-id sequences on edges; each node owns the KV payload
for its edge's tokens (opaque to the tree -- tensors, tuples, whatever the
caller slices out of an HF `past_key_values`). Lookup returns the longest
cached prefix so prefill can skip it. LRU eviction by leaf access time.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

_ids = itertools.count()


@dataclass
class _Node:
    edge: list[int] = field(default_factory=list)  # tokens on edge into this node
    payload: Any = None  # KV for edge tokens, len(payload) == len(edge) conceptually
    children: dict[int, "_Node"] = field(default_factory=dict)  # first-token -> child
    parent: "_Node | None" = None
    last_access: float = 0.0
    node_id: int = field(default_factory=lambda: next(_ids))


class RadixCache:
    """Longest-prefix KV reuse with LRU eviction.

    match(tokens)  -> (matched_len, payloads)  payloads in root-to-leaf order
    insert(tokens, payload_fn) -> stores payload for the uncached suffix;
        payload_fn(start, end) must return the KV payload covering
        tokens[start:end] (caller slices its past_key_values).
    evict(n_tokens) -> frees at least n_tokens of cached tokens, LRU first.
    """

    def __init__(self, max_tokens: int = 1_000_000, split_payload=None):
        """``split_payload(payload, n) -> (head, tail)`` enables edge
        splitting: a prompt diverging mid-edge reuses the common head
        (SGLang behavior). Without it, partial edges are treated as a
        miss and diverging prompts branch at the last full node."""
        self.root = _Node()
        self.max_tokens = max_tokens
        self.cached_tokens = 0
        self.split_payload = split_payload

    def match(self, tokens: Sequence[int]) -> tuple[int, list[Any]]:
        node = self.root
        pos = 0
        payloads: list[Any] = []
        tokens = list(tokens)
        now = time.monotonic()
        while pos < len(tokens):
            child = node.children.get(tokens[pos])
            if child is None:
                break
            n_common = _common_len(child.edge, tokens[pos:])
            if n_common < len(child.edge):
                if self.split_payload is None:
                    # partial edge unusable without payload slicing
                    break
                child = self._split(child, n_common)
            payloads.append(child.payload)
            child.last_access = now
            node = child
            pos += len(child.edge)
        return pos, payloads

    def _split(self, child: _Node, n: int) -> _Node:
        """Split child's edge at n: new middle node owns edge[:n] (and the
        head payload); child keeps edge[n:]. Returns the middle node."""
        head, tail = self.split_payload(child.payload, n)
        mid = _Node(
            edge=child.edge[:n], payload=head, parent=child.parent,
            last_access=child.last_access,
        )
        assert child.parent is not None
        child.parent.children[mid.edge[0]] = mid
        child.edge = child.edge[n:]
        child.payload = tail
        child.parent = mid
        mid.children[child.edge[0]] = child
        return mid

    def insert(self, tokens: Sequence[int], payload_fn) -> int:
        """Cache the uncached suffix of tokens. Returns tokens newly cached."""
        tokens = list(tokens)
        matched, _ = self.match(tokens)
        if matched == len(tokens):
            return 0
        node = self._node_at(tokens[:matched])
        new_edge = tokens[matched:]
        child = _Node(
            edge=new_edge,
            payload=payload_fn(matched, len(tokens)),
            parent=node,
            last_access=time.monotonic(),
        )
        node.children[new_edge[0]] = child
        self.cached_tokens += len(new_edge)
        while self.cached_tokens > self.max_tokens:
            if not self._evict_one():
                break
        return len(new_edge)

    def evict(self, n_tokens: int) -> int:
        freed = 0
        while freed < n_tokens:
            got = self._evict_one()
            if not got:
                break
            freed += got
        return freed

    def _node_at(self, tokens: list[int]) -> _Node:
        node = self.root
        pos = 0
        while pos < len(tokens):
            child = node.children[tokens[pos]]
            pos += len(child.edge)
            node = child
        return node

    def _evict_one(self) -> int:
        leaves = [n for n in self._walk() if not n.children and n is not self.root]
        if not leaves:
            return 0
        victim = min(leaves, key=lambda n: n.last_access)
        assert victim.parent is not None
        del victim.parent.children[victim.edge[0]]
        self.cached_tokens -= len(victim.edge)
        return len(victim.edge)

    def _walk(self):
        stack = [self.root]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children.values())


def _common_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n
