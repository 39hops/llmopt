"""Population LoRA for MLX: K adapters, ONE frozen base, one forward.

The memory-rich play (36GB Mac): the frozen base's weights are the
expensive object, the adapters are ~MB each. Fold the population into
the batch dimension — inputs are (K*B, T), the unchanged mlx-lm model
treats that as an ordinary batch, and only the LoRA-wrapped linears
are population-aware: they reshape activations to (K, B*T, D) and
apply per-adapter A_k/B_k with one batched einsum. Every adapter
trains simultaneously on its own data stream for one base forward's
worth of weight traffic.

Correctness contract (tests/test_population.py): a population step is
EXACTLY K independent single-adapter steps — per-adapter loss slices
are normalized separately (fused CE per slice), so grads for adapter k
never see adapter j's token counts. Same-function-not-same-weights
does not apply here: we verify by grads matching the sequential run,
which is running the weights, not comparing them.

Pairs with train/fused_ce.py: K x logits would be the unaffordable
object (K=4 at 8k tokens = 74GB of fp32 logits); chunked CE never
builds it.

VERDICT (measured 2026-07-13, scripts/bench_population.py): NULL at
our shapes. MLX 0.5B training runs ~1250 tok/s FLAT from ~256
tokens/step upward — the base forward is compute-saturated by one
adapter's batch, so there is no weight-traffic slack to amortize:
corpus shape (B=8, T=160) 1.04x at K=4, 1.03x at K=8; big shapes
LOSE (0.62x at K=4 B=4 T=512 — K x activations = memory pressure +
~33% einsum overhead). Only the launch-bound regime pays (1.22x at
B=1 T=256). Machinery stays: grads are exactly K sequential runs
(tests), so it banks for tiny-net populations (weightspace threads)
where per-step work IS launch-bound.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from llmopt.train.fused_ce import fused_ce


class PopLoRALinear(nn.Module):
    """K low-rank adapters over one frozen linear.

    Input x arrives as (K*B, ..., D) — population folded into batch.
    Base output is one shared matmul; the delta reshapes to expose K
    and einsums with the stacked (K, r, in) / (K, out, r) adapters.
    B zero-init: training starts at the base model exactly.
    """

    def __init__(self, base: nn.Linear, k: int, r: int = 16,
                 alpha: float = 32.0):
        super().__init__()
        self.base = base
        self.base.freeze()
        out_d, in_d = base.weight.shape
        dt = base.weight.dtype
        self.k = k
        self.scaling = alpha / r
        self.a = mx.random.normal((k, r, in_d)).astype(dt) / r**0.5
        self.b = mx.zeros((k, out_d, r), dtype=dt)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        xr = x.reshape(self.k, -1, x.shape[-1])
        za = mx.einsum("knd,krd->knr", xr, self.a)
        zb = mx.einsum("knr,kor->kno", za, self.b)
        return y + self.scaling * zb.reshape(y.shape)


TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")


def apply_population_lora(model, k: int, *, r: int = 16,
                          alpha: float = 32.0,
                          targets=TARGETS) -> int:
    """Freeze the model, wrap every matching nn.Linear. Returns count."""
    model.freeze()
    wrapped = 0
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if isinstance(module, nn.Linear) and leaf in targets:
            parent = model
            *path, last = name.split(".")
            for p in path:
                parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
            wrapped_layer = PopLoRALinear(module, k, r=r, alpha=alpha)
            if last.isdigit():
                parent[int(last)] = wrapped_layer
            else:
                setattr(parent, last, wrapped_layer)
            wrapped += 1
    return wrapped


def population_loss(hidden: mx.array, head_weight: mx.array,
                    targets: mx.array, k: int,
                    chunk: int = 1024) -> mx.array:
    """Sum of per-adapter mean CEs. hidden: (K*B, T, D) or (K*N, D);
    targets aligned. Per-slice normalization keeps adapter k's grads
    identical to a lone run regardless of adapter j's ignore counts."""
    d = hidden.shape[-1]
    h = hidden.reshape(k, -1, d)
    t = targets.reshape(k, -1)
    total = mx.array(0.0, dtype=mx.float32)
    for i in range(k):
        total = total + fused_ce(h[i], head_weight, t[i], chunk=chunk)
    return total


def adapter_state(model, i: int) -> dict:
    """Extract adapter i's {name.a, name.b} for saving/merging —
    the {**a, **b} adapter-dict convention from the torch side."""
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, PopLoRALinear):
            out[f"{name}.a"] = module.a[i]
            out[f"{name}.b"] = module.b[i]
    return out
