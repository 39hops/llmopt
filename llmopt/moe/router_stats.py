"""Router utilization stats for MoE expert pruning.

The router is the trained mechanism that decides which experts see a
token. Run a domain corpus (mathgen prompts) and a general corpus
through the model, record which experts each layer's router picks, and
the difference is a map of "where the domain lives" — the measurable
version of loading only the math/coding weights of a big model.

Framework-agnostic core: callers feed per-token top-k expert indices
(and optionally router weights) per layer; this module aggregates and
derives keep-sets under three criteria worth testing side by side:

- "ever":  keep experts selected at least once. Cheapest possible rule;
  the hypothesis is that for right-or-wrong domains (math, code) an
  expert either carries applicable circuits or it doesn't.
- "mass":  keep the smallest set covering >= `threshold` of cumulative
  router probability mass. Drops experts that were only ever marginal
  picks, even if technically "ever selected".
- "topq":  keep the top `threshold` fraction of experts by selection
  count. A blunt utilization quantile, the standard pruning baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RouterStats:
    """Accumulates per-layer expert selection counts and weight mass."""

    n_experts: int
    counts: dict[int, list[int]] = field(default_factory=dict)
    mass: dict[int, list[float]] = field(default_factory=dict)
    tokens: int = 0

    def _layer(self, layer: int) -> tuple[list[int], list[float]]:
        if layer not in self.counts:
            self.counts[layer] = [0] * self.n_experts
            self.mass[layer] = [0.0] * self.n_experts
        return self.counts[layer], self.mass[layer]

    def update(self, layer: int, topk_indices, topk_weights=None) -> None:
        """topk_indices: iterable over tokens, each an iterable of expert
        ids picked for that token. topk_weights mirrors it with router
        probabilities (defaults to 1.0 per pick)."""
        counts, mass = self._layer(layer)
        for t, picks in enumerate(topk_indices):
            picks = list(picks)
            ws = (
                list(topk_weights[t])
                if topk_weights is not None
                else [1.0] * len(picks)
            )
            for e, w in zip(picks, ws):
                counts[int(e)] += 1
                mass[int(e)] += float(w)
        if self.counts and layer == min(self.counts):
            self.tokens += len(list(topk_indices))

    def keep_set(self, layer: int, criterion: str, threshold: float = 0.99) -> set[int]:
        counts, mass = self._layer(layer)
        if criterion == "ever":
            return {e for e, c in enumerate(counts) if c > 0}
        if criterion == "mass":
            order = sorted(range(self.n_experts), key=lambda e: -mass[e])
            total = sum(mass) or 1.0
            kept, acc = set(), 0.0
            for e in order:
                if acc >= threshold * total:
                    break
                if mass[e] > 0:
                    kept.add(e)
                    acc += mass[e]
            return kept
        if criterion == "topq":
            n_keep = max(1, round(threshold * self.n_experts))
            order = sorted(range(self.n_experts), key=lambda e: -counts[e])
            return {e for e in order[:n_keep] if counts[e] > 0}
        raise ValueError(f"unknown criterion: {criterion}")

    def utilization(self, layer: int) -> list[float]:
        """Fraction of routing decisions each expert received."""
        counts, _ = self._layer(layer)
        total = sum(counts) or 1
        return [c / total for c in counts]


def overlap(a: set[int], b: set[int]) -> float:
    """Jaccard overlap between two keep-sets — near 1.0 between the
    domain and general corpora means routing is NOT domain-biased and
    the pruning idea dies (report that honestly)."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def prune_summary(
    domain: RouterStats, general: RouterStats, criterion: str, threshold: float = 0.99
) -> dict[int, dict]:
    """Per-layer: domain keep-set size, general keep-set size, jaccard."""
    out = {}
    for layer in sorted(domain.counts):
        d = domain.keep_set(layer, criterion, threshold)
        g = general.keep_set(layer, criterion, threshold)
        out[layer] = {
            "domain_kept": len(d),
            "general_kept": len(g),
            "jaccard": overlap(d, g),
            "domain_set": sorted(d),
        }
    return out
