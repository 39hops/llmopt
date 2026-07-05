"""The full stack in one engine: radix prefix reuse + prompt-lookup
decoding + StaticCache/CUDA-graph steps.

Each optimization attacks a different phase and they compose cleanly:

- radix prefix KV reuse (cache/radix.py + cache/prefix_reuse.py) kills
  repeated prefill — warm requests pay only their divergent suffix
  (TTFT);
- prompt-lookup decoding (decoding/lookup_generic.py) turns
  input-grounded decode steps into multi-token verify blocks (TPOT);
- StaticCache + torch.compile reduce-overhead replays one CUDA graph
  per verify block (launch overhead).

Output stays token-identical to eager greedy: reuse only changes where
prefill KV comes from, lookup only accepts exact-match drafts.
"""

from __future__ import annotations

from typing import Sequence

from llmopt.backends.torch_static import TorchStaticBackend
from llmopt.cache.prefix_reuse import split_payload
from llmopt.cache.radix import RadixCache
from llmopt.decoding.lookup_generic import generate_lookup


class StackedEngine:
    """Sequential-request engine composing all three optimizations."""

    def __init__(
        self, model, *, compiled_step=None, num_draft: int = 16,
        max_ngram: int = 4, cache_max_tokens: int = 100_000,
        datastore=None,
    ):
        """``datastore``: a decoding.datastore.SuffixDatastore — REST-style
        retrieval drafting; every finished generation is added, so accept
        rates climb as the engine sees more traffic."""
        self.model = model
        self.compiled_step = compiled_step
        self.num_draft = num_draft
        self.max_ngram = max_ngram
        self.datastore = datastore
        self.prefix_cache = RadixCache(
            max_tokens=cache_max_tokens, split_payload=split_payload
        )

    def generate(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int = 128,
        eos_token_id: int | None = None,
    ) -> tuple[list[int], dict]:
        import torch

        prompt_ids = list(prompt_ids)
        matched, payloads = self.prefix_cache.match(prompt_ids)
        matched = min(matched, len(prompt_ids) - 1)
        prefix_kv = None
        if matched:
            prefix_kv = [
                (
                    torch.cat([p[i][0] for p in payloads], dim=2)[:, :, :matched],
                    torch.cat([p[i][1] for p in payloads], dim=2)[:, :, :matched],
                )
                for i in range(len(payloads[0]))
            ]

        backend = TorchStaticBackend(
            self.model, compiled_step=self.compiled_step, prefix_kv=prefix_kv
        )
        draft_fn = None
        if self.datastore is not None:
            from llmopt.decoding.datastore import make_draft_fn

            draft_fn = make_draft_fn(self.datastore, max_ngram=self.max_ngram)
        tokens, stats = generate_lookup(
            backend, prompt_ids, max_new_tokens=max_new_tokens,
            num_draft=self.num_draft, max_ngram=self.max_ngram,
            eos_token_id=eos_token_id, draft_fn=draft_fn,
        )
        stats["prefix_hit_tokens"] = matched
        if self.datastore is not None:
            self.datastore.add(tokens)

        # store this prompt's KV for future requests (uncached suffix only)
        layers = backend.cache.layers
        n = len(prompt_ids)
        self.prefix_cache.insert(
            prompt_ids,
            lambda s, e: [
                (l.keys[:, :, s:e].clone(), l.values[:, :, s:e].clone())
                for l in layers
            ],
        )
        return tokens, stats
