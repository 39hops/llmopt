"""Mixture-of-Experts layer: top-k routing, capacity limits, aux losses.

A router scores tokens over E experts; each token is processed by its
top-k experts and the outputs combined by the (renormalized) router
weights. Two failure modes drive the extra machinery:

- Collapse: the router sends everything to one expert. The Switch aux
  loss E * sum_i(f_i * P_i) (token fraction * mean prob per expert) is
  minimized when routing is uniform; z-loss keeps router logits small.
- Overflow: real deployments give each expert a fixed token budget
  (capacity = ceil(cf * n * k / E)). Tokens over budget are dropped at
  that expert — they keep the residual path, losing only the expert's
  contribution.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoELayer(nn.Module):
    """Top-k gated MoE over SwiGLU experts.

    forward(x: [n, hidden]) -> (out [n, hidden], aux) where aux carries
    router probs / expert loads for the balancing losses and drop stats.
    capacity_factor=None disables token dropping.
    """

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        num_experts: int,
        *,
        top_k: int = 2,
        capacity_factor: float | None = None,
    ):
        super().__init__()
        self.router = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLUExpert(hidden, intermediate) for _ in range(num_experts)
        )
        self.top_k = top_k
        self.capacity_factor = capacity_factor

    def forward(self, x):
        n, e = x.shape[0], len(self.experts)
        logits = self.router(x)  # [n, E]
        probs = F.softmax(logits, dim=-1)
        weight, idx = probs.topk(self.top_k, dim=-1)  # [n, k]
        weight = weight / weight.sum(-1, keepdim=True)

        capacity = (
            math.ceil(self.capacity_factor * n * self.top_k / e)
            if self.capacity_factor is not None
            else n * self.top_k
        )

        out = torch.zeros_like(x)
        dropped = 0
        for ei, expert in enumerate(self.experts):
            token, slot = (idx == ei).nonzero(as_tuple=True)  # [m] each
            if token.numel() > capacity:  # first-come priority
                dropped += token.numel() - capacity
                token, slot = token[:capacity], slot[:capacity]
            if token.numel():
                out[token] += weight[token, slot, None] * expert(x[token])

        aux = {
            "probs": probs,
            "expert_load": F.one_hot(idx, e).sum((0, 1)).float() / (n * self.top_k),
            "logits": logits,
            "dropped": dropped,
        }
        return out, aux


def load_balance_loss(aux) -> torch.Tensor:
    """Switch-style: E * sum_i f_i * P_i; equals 1 at perfect balance."""
    e = aux["probs"].shape[-1]
    return e * (aux["expert_load"] * aux["probs"].mean(0)).sum()


def router_z_loss(aux) -> torch.Tensor:
    """Penalize large router logits: mean logsumexp^2."""
    return aux["logits"].logsumexp(-1).pow(2).mean()
