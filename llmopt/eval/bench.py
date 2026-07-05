"""Benchmark harness: perplexity and tokens/sec, shared by all experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class GenBenchResult:
    tokens_per_sec: float
    new_tokens: int
    wall_sec: float
    stats: dict


def perplexity(model, token_ids, *, batch_size: int = 8, device: str | None = None) -> float:
    """Perplexity of model over list of token-id sequences (teacher-forced)."""
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    total_nll, total_toks = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(token_ids), batch_size):
            for seq in token_ids[start : start + batch_size]:
                ids = torch.tensor([list(seq)], device=device)
                logits = model(input_ids=ids).logits[0, :-1].float()
                lp = torch.log_softmax(logits, dim=-1)
                tgt = ids[0, 1:]
                total_nll += float(-lp.gather(-1, tgt.unsqueeze(-1)).sum())
                total_toks += tgt.numel()
    import math

    return math.exp(total_nll / total_toks)


def bench_generate(generate_fn, *, warmup: int = 1, repeats: int = 3) -> GenBenchResult:
    """Time a zero-arg callable returning (tokens, stats) with known prompt len
    embedded in stats['prompt_len'] (or pass full tokens; new = len - prompt).

    generate_fn must be deterministic-ish; median of repeats reported.
    """
    for _ in range(warmup):
        generate_fn()
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        tokens, stats = generate_fn()
        dt = time.perf_counter() - t0
        new = len(tokens) - stats.get("prompt_len", 0)
        runs.append((dt, new, stats))
    runs.sort(key=lambda r: r[0])
    dt, new, stats = runs[len(runs) // 2]
    return GenBenchResult(
        tokens_per_sec=new / dt if dt > 0 else float("inf"),
        new_tokens=new,
        wall_sec=dt,
        stats=stats,
    )
