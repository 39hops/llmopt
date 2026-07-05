"""MLX backend: mlx-lm model + trimmable KV cache (Apple silicon).

Implements the DecodeBackend protocol (see base.py) so the generic
prompt-lookup loop (llmopt.decoding.lookup_generic) runs on MLX.

Unlike the torch StaticCache path, mlx-lm caches track their write
position implicitly (``cache[0].offset``); there is no per-call
``cache_position``. The loop's rewind discipline keeps that offset in
sync with ``start_pos``, and ``begin``/``step_argmax`` assert it.
"""

from __future__ import annotations

from typing import Sequence


class MLXBackend:
    """DecodeBackend over an mlx-lm causal LM.

    ``model`` is the nn.Module returned by ``mlx_lm.load``. ``max_len``
    passed to ``begin`` is unused: mlx-lm's default KVCache grows on
    demand and supports trimming, which is all rewind needs.
    """

    def __init__(self, model):
        self.model = model
        self.cache = None

    def begin(self, prompt_ids: Sequence[int], max_len: int) -> int:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        self.cache = make_prompt_cache(self.model)
        logits = self.model(mx.array([list(prompt_ids)]), cache=self.cache)
        return int(mx.argmax(logits[0, -1]))

    def step_argmax(
        self, fed: Sequence[int], start_pos: int, n_real: int
    ) -> list[int]:
        import mlx.core as mx

        assert self.cache[0].offset == start_pos, (
            f"cache offset {self.cache[0].offset} != start_pos {start_pos}"
        )
        logits = self.model(mx.array([list(fed)]), cache=self.cache)
        return [int(t) for t in mx.argmax(logits[0, :n_real], axis=-1).tolist()]

    def rewind(self, length: int) -> None:
        from mlx_lm.models.cache import trim_prompt_cache

        excess = self.cache[0].offset - length
        assert excess >= 0, (
            f"rewind length {length} > cache offset {self.cache[0].offset}"
        )
        trim_prompt_cache(self.cache, excess)
