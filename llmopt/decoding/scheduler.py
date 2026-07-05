"""Readable scheduler over the continuous-batching engine: priority
admission, preemption, and prefill/decode disaggregation.

Policies, in the order they matter:

- Admission is by priority (ties: arrival order), not FIFO.
- Preemption: if a strictly higher-priority request is waiting and the
  batch is full, the lowest-priority running request is evicted — its
  KV is dropped and it re-queues with prompt + generated-so-far as its
  new prompt (recompute-on-resume; output stays greedy-exact because
  greedy decoding is deterministic in the token prefix).
- Disaggregation: at most `prefill_ratio` of engine steps do prefill
  work while decodes are pending, bounding how much TTFT-optimizing
  prefill can inflate running requests' TPOT (the interference that
  motivates separate prefill/decode pools in production systems).
"""

from __future__ import annotations

from llmopt.decoding.batching import BatchEngine, Request


class Scheduler(BatchEngine):
    def __init__(self, model, *, max_batch=4, chunk_size=16,
                 prefill_ratio: float = 0.5):
        super().__init__(model, max_batch=max_batch, chunk_size=chunk_size)
        self.prefill_ratio = prefill_ratio
        self.stats["preemptions"] = 0
        self._prefill_steps = 0

    # --- policy hooks --------------------------------------------------------

    def _pick_waiting(self) -> Request:
        return max(self.waiting, key=lambda r: (r.priority, -r.rid))

    def _should_prefill(self) -> bool:
        if not self.waiting:
            return False
        if not self.running:
            return True
        # disaggregation budget: prefill only while under ratio
        return self._prefill_steps < self.prefill_ratio * self.stats["steps"]

    def _maybe_preempt(self) -> None:
        if not self.waiting or len(self.running) < self.max_batch:
            return
        top = self._pick_waiting()
        victim = min(self.running, key=lambda r: (r.priority, -r.rid))
        if top.priority > victim.priority:
            self.stats["preemptions"] += 1
            self._release(victim)
            # resume later by re-prefilling prompt + progress so far
            victim.prompt = victim.prompt + victim.generated
            victim.max_new_tokens -= len(victim.generated)
            victim.generated, victim.prefilled, victim.cache = [], 0, None
            self.waiting.append(victim)

    # --- engine step with policies -------------------------------------------

    def step(self) -> None:
        self.stats["steps"] += 1
        self._maybe_preempt()
        if len(self.running) < self.max_batch and self._should_prefill():
            self._prefill_steps += 1
            self._prefill_chunk(self._pick_waiting())
        elif self.running:
            self._decode()
        elif self.waiting:  # nothing running: prefill regardless of ratio
            self._prefill_steps += 1
            self._prefill_chunk(self._pick_waiting())
