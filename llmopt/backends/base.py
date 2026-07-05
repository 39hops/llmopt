"""Backend protocol for greedy block-verify decoding.

The decode loop (llmopt.decoding.lookup_generic) is framework-agnostic:
only plain Python ints cross this boundary. A backend owns the model,
its KV cache, and any compilation strategy (CUDA graphs, mx.compile).

Contract mirrors the StaticCache block-verify scheme:

- ``begin`` allocates a cache big enough for ``max_len`` positions and
  prefills the prompt, returning the greedy next token.
- ``step_argmax`` feeds ``fed`` (fixed block: last token + draft + pads)
  at absolute positions ``start_pos..start_pos+len(fed)`` and returns
  the greedy next-token prediction for the first ``n_real`` positions.
  Pads sit at the highest positions, so causal masking keeps them
  invisible to real queries.
- ``rewind`` resets the cache write pointer to ``length`` so the next
  block overwrites stale (rejected/pad) slots.
"""

from __future__ import annotations

from typing import Protocol, Sequence


class DecodeBackend(Protocol):
    def begin(self, prompt_ids: Sequence[int], max_len: int) -> int:
        """Allocate cache, prefill prompt, return greedy next token."""
        ...

    def step_argmax(
        self, fed: Sequence[int], start_pos: int, n_real: int
    ) -> list[int]:
        """Run one fixed-shape block; return argmax preds for first n_real."""
        ...

    def rewind(self, length: int) -> None:
        """Reset cache write pointer to `length` positions."""
        ...
