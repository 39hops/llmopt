"""Bit-width allocator: pick per-layer bits to hit a quality (delta-KL)
target at minimum memory. Pure Python, no torch.

Model: total delta-KL assumed additive across layers (verify final config
end-to-end -- the assumption is approximate). Greedy Lagrangian relaxation:
start everything at the lowest bit width, repeatedly upgrade the layer with
the best (KL reduction / extra bits) ratio until the KL budget is met.
Greedy is near-optimal here because the frontier is convex-ish per layer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Assignment:
    bits_by_layer: dict[str, int]
    est_delta_kl: float
    total_bits: int  # sum(bits * n_params), i.e. memory in bits
    total_params: int

    @property
    def avg_bits(self) -> float:
        return self.total_bits / self.total_params


def allocate_bits(
    sensitivities,
    *,
    kl_budget: float,
    fallback_bits: int = 16,
) -> Assignment:
    """sensitivities: list of LayerSensitivity-like (name, bits, delta_kl,
    n_params). Layers may offer any set of bit widths; fp16 (delta_kl=0,
    fallback_bits) is always available as the top rung.

    Returns min-memory assignment with sum(delta_kl) <= kl_budget.
    """
    # group: name -> sorted [(bits, delta_kl)], plus n_params
    table: dict[str, dict[int, float]] = {}
    params: dict[str, int] = {}
    for s in sensitivities:
        table.setdefault(s.name, {})[s.bits] = s.delta_kl
        params[s.name] = s.n_params
    for name in table:
        table[name][fallback_bits] = 0.0

    # start at cheapest rung everywhere
    current = {name: min(opts) for name, opts in table.items()}
    total_kl = sum(table[n][b] for n, b in current.items())

    def upgrades(name):
        opts = sorted(table[name])
        i = opts.index(current[name])
        return opts[i + 1 :]

    while total_kl > kl_budget:
        best = None  # (ratio, name, new_bits)
        for name in table:
            for nb in upgrades(name):
                kl_gain = table[name][current[name]] - table[name][nb]
                if kl_gain <= 0:
                    continue
                cost = (nb - current[name]) * params[name]
                ratio = kl_gain / cost
                if best is None or ratio > best[0]:
                    best = (ratio, name, nb)
        if best is None:
            break  # everything already at top rung
        _, name, nb = best
        total_kl -= table[name][current[name]] - table[name][nb]
        current[name] = nb

    total_bits = sum(current[n] * params[n] for n in current)
    return Assignment(
        bits_by_layer=dict(current),
        est_delta_kl=total_kl,
        total_bits=total_bits,
        total_params=sum(params.values()),
    )


def pareto_front(sensitivities, *, kl_budgets, fallback_bits: int = 16):
    """Sweep budgets -> list of (kl_budget, Assignment) for quality-vs-size
    Pareto curve. Plot est_delta_kl (or end-to-end perplexity re-measured
    per config) against total_bits / 8e9 GB."""
    return [
        (b, allocate_bits(sensitivities, kl_budget=b, fallback_bits=fallback_bits))
        for b in kl_budgets
    ]
