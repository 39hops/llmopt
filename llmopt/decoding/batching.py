"""Chunked prefill + continuous batching over a padded batched KV cache.

A toy single-process engine with the real scheduling structure:

- Prefill is chunked: long prompts are processed ``chunk_size`` tokens
  per engine step on a private batch=1 cache, so one long prompt cannot
  stall decoding for requests already running.
- Batching is continuous: a finished request frees its slot immediately
  and a waiting request joins mid-flight; running requests never wait
  for stragglers.
- The batch shares one left-padded KV cache [B, H, T, D] + a 0/1 pad
  mask. Keys carry RoPE from their real (unpadded) positions, so rows of
  different true lengths coexist; per-row position_ids keep decoding
  correct. Output per request is token-identical to unbatched greedy.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Request:
    rid: int
    prompt: list[int]
    max_new_tokens: int
    eos_token_id: int | None = None
    priority: int = 0  # higher = more urgent (used by scheduler.py)
    # engine state
    generated: list[int] = field(default_factory=list)
    prefilled: int = 0  # prompt tokens processed so far
    cache: object = None  # private cache during chunked prefill
    done: bool = False


class BatchEngine:
    """Greedy continuous-batching engine. Call ``run`` or step manually."""

    def __init__(
        self, model, *, max_batch: int = 4, chunk_size: int = 16,
        prefix_cache=None,
    ):
        """``prefix_cache``: a cache.radix.RadixCache (built with
        cache.prefix_reuse.split_payload) enables cross-request prompt
        prefix reuse — prefill skips the longest cached prefix."""
        self.model = model
        self.max_batch = max_batch
        self.chunk_size = chunk_size
        self.prefix_cache = prefix_cache
        self.device = next(model.parameters()).device
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.kv = None  # list[(k, v)] per layer, [B, H, T, D]
        self.mask = None  # [B, T] 0/1
        self.stats = {
            "steps": 0, "decode_tokens": 0, "batch_occupancy": [],
            "prefill_tokens": 0, "prefix_hit_tokens": 0,
        }

    def submit(
        self, prompt, max_new_tokens: int, eos_token_id=None, priority: int = 0
    ) -> int:
        rid = self._next_rid = getattr(self, "_next_rid", -1) + 1
        self.waiting.append(
            Request(rid, list(prompt), max_new_tokens, eos_token_id, priority)
        )
        return rid

    # --- one engine step: a prefill chunk or a batched decode ------------

    def step(self) -> None:
        self.stats["steps"] += 1
        if self.waiting and len(self.running) < self.max_batch:
            self._prefill_chunk(self.waiting[0])
        elif self.running:
            self._decode()

    def run(self) -> dict[int, list[int]]:
        results: dict[int, list[int]] = {}
        while self.waiting or self.running:
            self.step()
            for r in [r for r in self.running if r.done]:
                results[r.rid] = r.prompt + r.generated
                self._release(r)
        return results

    # --- prefill ----------------------------------------------------------

    def _prefill_chunk(self, req: Request) -> None:
        import torch

        if req.cache is None and req.prefilled == 0 and self.prefix_cache is not None:
            self._reuse_prefix(req)
        chunk = req.prompt[req.prefilled : req.prefilled + self.chunk_size]
        self.stats["prefill_tokens"] += len(chunk)
        pos = torch.arange(
            req.prefilled, req.prefilled + len(chunk), device=self.device
        )
        with torch.inference_mode():
            out = self.model(
                input_ids=torch.tensor([chunk], device=self.device),
                past_key_values=req.cache, position_ids=pos[None],
                cache_position=pos, use_cache=True,
            )
        req.cache = out.past_key_values
        req.prefilled += len(chunk)
        if req.prefilled == len(req.prompt):
            if self.prefix_cache is not None:
                self._store_prefix(req)
            req.generated.append(int(out.logits[0, -1].argmax()))
            self.waiting.remove(req)
            self._join(req)
            self._check_done(req)

    def _reuse_prefix(self, req: Request) -> None:
        """Skip prefill of the longest radix-cached prefix. Cap at
        len(prompt) - 1: the final prompt token must run through the model
        to produce first-token logits."""
        from llmopt.cache.prefix_reuse import payloads_to_cache

        matched, payloads = self.prefix_cache.match(req.prompt)
        matched = min(matched, len(req.prompt) - 1)
        if matched:
            req.cache = payloads_to_cache(payloads, upto=matched)
            req.prefilled = matched
            self.stats["prefix_hit_tokens"] += matched

    def _store_prefix(self, req: Request) -> None:
        """Insert this prompt's KV into the radix tree (uncached suffix
        only; the tree slices what it needs)."""
        from llmopt.cache.prefix_reuse import slice_payload

        legacy = self._legacy(req.cache)
        self.prefix_cache.insert(
            req.prompt, lambda s, e: slice_payload(legacy, s, e)
        )

    # --- batched cache plumbing --------------------------------------------

    @staticmethod
    def _legacy(past) -> list:
        if hasattr(past, "to_legacy_cache"):
            return [list(kv) for kv in past.to_legacy_cache()]
        if hasattr(past, "layers"):
            return [[l.keys, l.values] for l in past.layers]
        return [list(kv) for kv in past]

    def _to_cache(self):
        """Rebuild a DynamicCache from self.kv (transformers 5.x dropped
        from_legacy_cache; update() per layer works on every version)."""
        from transformers import DynamicCache

        cache = DynamicCache()
        for i, (k, v) in enumerate(self.kv):
            cache.update(k, v, i)
        return cache

    def _join(self, req: Request) -> None:
        """Left-pad the newcomer's KV (or the batch) to equal T and stack."""
        import torch

        new = self._legacy(req.cache)
        req.cache = None
        new_len = new[0][0].shape[2]
        new_mask = torch.ones((1, new_len), device=self.device, dtype=torch.long)

        if self.kv is None:
            self.kv, self.mask = new, new_mask
        else:
            cur_len = self.kv[0][0].shape[2]
            pad = abs(cur_len - new_len)
            if pad:
                def lpad(t):
                    z = torch.zeros_like(t[:, :, :1]).expand(-1, -1, pad, -1)
                    return torch.cat([z, t], dim=2)

                if new_len < cur_len:
                    new = [[lpad(k), lpad(v)] for k, v in new]
                    new_mask = torch.nn.functional.pad(new_mask, (pad, 0))
                else:
                    self.kv = [[lpad(k), lpad(v)] for k, v in self.kv]
                    self.mask = torch.nn.functional.pad(self.mask, (pad, 0))
            self.kv = [
                [torch.cat([k, nk]), torch.cat([v, nv])]
                for (k, v), (nk, nv) in zip(self.kv, new)
            ]
            self.mask = torch.cat([self.mask, new_mask])
        self.running.append(req)

    def _release(self, req: Request) -> None:
        import torch

        i = self.running.index(req)
        self.running.remove(req)
        if not self.running:
            self.kv, self.mask = None, None
            return
        keep = [j for j in range(self.mask.shape[0]) if j != i]
        idx = torch.tensor(keep, device=self.device)
        self.kv = [
            [k.index_select(0, idx), v.index_select(0, idx)] for k, v in self.kv
        ]
        self.mask = self.mask.index_select(0, idx)
        # trim columns that are now pad for every remaining row
        lead = int((self.mask.cumsum(1) == 0).all(0).sum())
        if lead:
            self.kv = [[k[:, :, lead:], v[:, :, lead:]] for k, v in self.kv]
            self.mask = self.mask[:, lead:]

    # --- decode -------------------------------------------------------------

    def _decode(self) -> None:
        import torch

        self.stats["batch_occupancy"].append(len(self.running))
        last = [r.generated[-1] for r in self.running]
        # real position of the fed token per row
        pos = torch.tensor(
            [[len(r.prompt) + len(r.generated) - 1] for r in self.running],
            device=self.device,
        )
        t = self.kv[0][0].shape[2]
        mask = torch.cat(
            [self.mask, torch.ones_like(self.mask[:, :1])], dim=1
        )
        with torch.inference_mode():
            out = self.model(
                input_ids=torch.tensor(last, device=self.device)[:, None],
                past_key_values=self._to_cache(),
                attention_mask=mask, position_ids=pos,
                cache_position=torch.arange(t, t + 1, device=self.device),
                use_cache=True,
            )
        self.kv = self._legacy(out.past_key_values)
        self.mask = mask
        for row, req in enumerate(self.running):
            req.generated.append(int(out.logits[row, -1].argmax()))
            self.stats["decode_tokens"] += 1
            self._check_done(req)

    @staticmethod
    def _check_done(req: Request) -> None:
        if req.eos_token_id is not None and req.generated[-1] == req.eos_token_id:
            req.done = True
        elif len(req.generated) >= req.max_new_tokens:
            req.done = True
