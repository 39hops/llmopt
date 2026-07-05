"""Composable sampling pipeline: logits processors + terminal sampler.

A processor maps ``(logits, ctx) -> logits`` for one next-token
distribution (1-D tensor over vocab); ``ctx`` is the full token list so
far (prompt + generated). Filtering processors mask excluded tokens to
-inf; the pipeline ends with a softmax + multinomial draw (or argmax).

Implemented: temperature, top_k, top_p (nucleus), min_p, DRY repetition
penalty, and mirostat v2 (stateful, replaces the terminal sampler).
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

Processor = Callable[[torch.Tensor, Sequence[int]], torch.Tensor]


def temperature(t: float) -> Processor:
    def proc(logits, ctx):
        return logits / t

    return proc


def top_k(k: int) -> Processor:
    """Keep the k highest-probability tokens."""

    def proc(logits, ctx):
        kth = torch.topk(logits, k).values[-1]
        return logits.masked_fill(logits < kth, float("-inf"))

    return proc


def top_p(p: float) -> Processor:
    """Nucleus: keep the smallest prefix of tokens with cumulative
    probability >= p (the crossing token is kept)."""

    def proc(logits, ctx):
        sorted_logits, idx = torch.sort(logits, descending=True)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(-1)
        drop_sorted = cum >= p
        drop_sorted[1:] = drop_sorted[:-1].clone()  # keep crossing token
        drop_sorted[0] = False  # always keep the top token
        drop = torch.zeros_like(drop_sorted)
        drop.scatter_(0, idx, drop_sorted)
        return logits.masked_fill(drop, float("-inf"))

    return proc


def min_p(p: float) -> Processor:
    """Keep tokens with prob >= p * max_prob (scale-adaptive floor)."""

    def proc(logits, ctx):
        probs = torch.softmax(logits, dim=-1)
        return logits.masked_fill(probs < p * probs.max(), float("-inf"))

    return proc


def dry(
    multiplier: float = 0.8,
    base: float = 1.75,
    allowed_len: int = 2,
    window: int = 256,
) -> Processor:
    """DRY repetition penalty (p-e-w): penalize tokens that would extend
    a verbatim repeat of the context suffix. A token continuing a match
    of length L > allowed_len gets ``multiplier * base**(L - allowed_len)``
    subtracted from its logit.
    """

    def proc(logits, ctx):
        ctx = list(ctx[-window:])
        n = len(ctx)
        if n < 2:
            return logits
        penalty: dict[int, float] = {}
        for i in range(n - 1):  # match ending at ctx[i], continuation ctx[i+1]
            if ctx[i] != ctx[-1]:
                continue
            length = 1
            while (
                length < i + 1
                and length < n
                and ctx[i - length] == ctx[n - 1 - length]
            ):
                length += 1
            if length > allowed_len:
                tok = ctx[i + 1]
                pen = multiplier * base ** (length - allowed_len)
                penalty[tok] = max(penalty.get(tok, 0.0), pen)
        if penalty:
            logits = logits.clone()
            for tok, pen in penalty.items():
                logits[tok] -= pen
        return logits

    return proc


class Mirostat:
    """Mirostat v2 (Basu et al. 2021): keep observed surprise near tau
    bits by truncating to tokens with surprise <= mu and adapting mu.

    Stateful terminal sampler: use as ``SamplerPipeline(..., sampler=m)``.
    """

    def __init__(self, tau: float = 3.0, eta: float = 0.1):
        self.tau = tau
        self.eta = eta
        self.mu = 2.0 * tau

    def __call__(self, logits: torch.Tensor, gen: torch.Generator) -> int:
        probs = torch.softmax(logits, dim=-1)
        surprise = -torch.log2(probs)
        cut = surprise > self.mu
        if cut.all():  # keep at least the top token
            cut[int(probs.argmax())] = False
        probs = probs.masked_fill(cut, 0.0)
        probs = probs / probs.sum()
        tok = int(torch.multinomial(probs, 1, generator=gen))
        self.mu -= self.eta * (float(surprise[tok]) - self.tau)
        return tok


def _default_sampler(logits: torch.Tensor, gen: torch.Generator) -> int:
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=gen))


class SamplerPipeline:
    """Apply processors in order, then draw a token.

    ``sampler`` is the terminal draw; pass a ``Mirostat`` instance to
    replace plain multinomial sampling. ``seed`` fixes the generator.
    """

    def __init__(
        self,
        *processors: Processor,
        sampler: Callable[[torch.Tensor, torch.Generator], int] | None = None,
        seed: int | None = None,
    ):
        self.processors = processors
        self.sampler = sampler or _default_sampler
        self.gen = torch.Generator()
        if seed is not None:
            self.gen.manual_seed(seed)

    def __call__(self, logits: torch.Tensor, ctx: Sequence[int] = ()) -> int:
        for proc in self.processors:
            logits = proc(logits, ctx)
        return self.sampler(logits, self.gen)
