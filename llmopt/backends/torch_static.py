"""Torch/HF backend: StaticCache + optional torch.compile CUDA-graph step.

Implements the DecodeBackend protocol (see base.py). Same mechanics as
llmopt.decoding.lookup_static, factored so the decode loop can also run
on non-torch backends (e.g. MLX).
"""

from __future__ import annotations

from typing import Sequence


class TorchStaticBackend:
    """DecodeBackend over a HF causal LM with a StaticCache.

    compiled_step: optionally torch.compile(model, mode="reduce-overhead");
    prefill always runs the eager model (dynamic length, happens once).
    """

    def __init__(self, model, *, compiled_step=None, prefix_kv=None):
        """``prefix_kv``: optional per-layer [(k, v), ...] with shapes
        [1, H, t, D] (cache.prefix_reuse payload format). begin() seeds
        the StaticCache with it and prefills only the remaining suffix —
        radix prefix reuse for the fast decode path."""
        self.model = model
        self.step = model if compiled_step is None else compiled_step
        self.cache = None
        self.prefix_kv = prefix_kv

    def begin(self, prompt_ids: Sequence[int], max_len: int) -> int:
        import torch
        from transformers import StaticCache

        m = self.model
        if self.step is not m:
            # compiled step: bucket the cache length so requests of nearby
            # sizes share one graph instead of re-capturing per length
            max_len = -(-max_len // 512) * 512
        self.cache = StaticCache(
            config=m.config, max_batch_size=1, max_cache_len=max_len,
            device=m.device, dtype=m.dtype,
        )
        skip = 0
        with torch.inference_mode():
            if self.prefix_kv is not None:
                # seed through cache.update: the sanctioned write path
                # (allocates lazily, advances cumulative_length)
                skip = self.prefix_kv[0][0].shape[2]
                assert skip < len(prompt_ids), "must prefill >= 1 real token"
                seed_pos = torch.arange(skip, device=m.device)
                for i, (k, v) in enumerate(self.prefix_kv):
                    self.cache.update(
                        k.to(m.device, m.dtype), v.to(m.device, m.dtype), i,
                        {"cache_position": seed_pos},
                    )
            pos = torch.arange(skip, len(prompt_ids), device=m.device)
            out = m(
                input_ids=torch.tensor([list(prompt_ids[skip:])], device=m.device),
                past_key_values=self.cache, cache_position=pos, use_cache=True,
            )
            return int(out.logits[0, -1].argmax())

    def step_argmax(
        self, fed: Sequence[int], start_pos: int, n_real: int
    ) -> list[int]:
        import torch

        device = self.model.device
        with torch.inference_mode():
            cp = torch.arange(start_pos, start_pos + len(fed), device=device)
            out = self.step(
                input_ids=torch.tensor([list(fed)], device=device),
                past_key_values=self.cache, cache_position=cp, use_cache=True,
            )
            return out.logits[0, :n_real].argmax(-1).tolist()

    def rewind(self, length: int) -> None:
        import torch

        # cumulative_length is created lazily inside inference_mode, so the
        # in-place fill must happen inside inference_mode too
        with torch.inference_mode():
            for layer in self.cache.layers:
                cl = getattr(layer, "cumulative_length", None)
                if cl is None:
                    continue
                if hasattr(cl, "fill_"):
                    cl.fill_(length)
                else:  # plain int in some transformers versions
                    layer.cumulative_length = length
