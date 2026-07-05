"""Serving latency metrics: TTFT and TPOT.

TTFT (time to first token) is dominated by prefill — the user's
"did it hear me" number. TPOT (time per output token) is the decode
steady state — the "how fast does it talk" number. Total latency =
TTFT + TPOT * (tokens - 1); reporting only aggregate tok/s hides which
phase is slow (see eval/roofline: they're bound by different limits).
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class LatencyReport:
    ttft_s: float
    tpot_s: float
    new_tokens: int

    @property
    def total_s(self) -> float:
        return self.ttft_s + self.tpot_s * max(self.new_tokens - 1, 0)

    @property
    def decode_tok_s(self) -> float:
        return 1.0 / self.tpot_s if self.tpot_s > 0 else float("inf")


def measure_latency(model, prompt_ids, *, max_new_tokens: int = 32) -> LatencyReport:
    """Greedy decode with a KV cache, timing prefill and decode
    separately."""
    import torch

    device = next(model.parameters()).device
    ids = list(prompt_ids)
    with torch.inference_mode():
        t0 = time.perf_counter()
        out = model(
            input_ids=torch.tensor([ids], device=device), use_cache=True
        )
        nxt = int(out.logits[0, -1].argmax())
        ttft = time.perf_counter() - t0

        past = out.past_key_values
        t1 = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            out = model(
                input_ids=torch.tensor([[nxt]], device=device),
                past_key_values=past, use_cache=True,
            )
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
        tpot = (time.perf_counter() - t1) / max(max_new_tokens - 1, 1)
    return LatencyReport(ttft, tpot, max_new_tokens)
