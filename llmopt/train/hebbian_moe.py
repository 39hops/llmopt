"""Hebbian-coupled MoE birth with a merge-free dense endpoint.

The measured recipe ("birth as Hebbian MoE, ship as dense",
RESULTS 2026-07-31): train a top-1 switch MoE while periodically
relaxing each expert toward its co-routed peers, then average the
experts into ONE dense FFN at ship time.

Measured basis (scaffold program, 15+ cells):
- merge-free at n=4 (three Mac seeds + one cuda birth; merge
  deltas {+2, -2, +1, +2} on the 120 gate — never negative in
  expectation);
- lambda-sweep on cuda: gates FLAT across lam {0.1..1.0} while
  expert correlation rises monotonically 0.83 -> 0.97 — the pull
  dials anatomy, not capability;
- gate advantage over load-balanced MoE is device-scoped (+5-6 on
  Mac at n=3, ~0 on cuda n=1) — the recipe's robust claim is the
  FREE MERGE, not a capability lift;
- expert decorrelation is the init default preserved by training
  (UMOE-1/2): without the pull, experts share nothing and cannot
  be merged post-hoc (N3: top-pair merge +3.4 ppl on OLMoE).

Usage sketch (see scratch/umoe_conserve.py for the full trainer):

    coupler = HebbianCoupler(n_experts=4, lam=0.5, every=100)
    ...
    probs = router(h).softmax(-1)          # [B, T, E]
    coupler.observe(probs)                 # each forward
    coupler.maybe_relax(step, experts())   # each optimizer step
    ...
    merged = merge_experts(experts())      # ship time: E -> 1

`experts()` yields one list of same-shaped tensors per expert
(e.g. [gate.weight, up.weight, down.weight]). All relaxation is
in-place under no_grad; never call it mid-backward.
"""
from __future__ import annotations

import torch

__all__ = ["HebbianCoupler", "merge_experts"]


class HebbianCoupler:
    """Tracks co-routing overlap and relaxes experts toward peers.

    Overlap EMA: ema <- decay*ema + (1-decay)*mean_bt(p p^T), the
    router-probability outer product averaged over batch and time.
    Relaxation (every `every` steps): for each ordered pair (i, j),
    w_i += lam * ema[i, j] * (w_j - w_i), optionally restricted to
    `edges` (e.g. tree siblings — measured: edge restriction builds
    a phylogeny (within-pair corr 0.95) but pays -6 on the gate;
    default all-to-all is the recipe).
    """

    def __init__(self, n_experts: int, lam: float = 0.5,
                 every: int = 100, ema_decay: float = 0.99,
                 edges: set[tuple[int, int]] | None = None):
        if n_experts < 2:
            raise ValueError("need at least 2 experts")
        self.n = n_experts
        self.lam = lam
        self.every = every
        self.decay = ema_decay
        self.edges = edges
        self.ema: torch.Tensor | None = None

    def observe(self, probs: torch.Tensor) -> None:
        """probs: [B, T, E] router probabilities (post-softmax)."""
        if probs.shape[-1] != self.n:
            raise ValueError(
                f"probs last dim {probs.shape[-1]} != {self.n}")
        ov = torch.einsum("bti,btj->ij", probs, probs)
        ov = (ov / (probs.shape[0] * probs.shape[1])).detach().cpu()
        self.ema = ov if self.ema is None else \
            self.decay * self.ema + (1 - self.decay) * ov

    @torch.no_grad()
    def maybe_relax(self, step: int,
                    expert_params: list[list[torch.Tensor]]) -> bool:
        """Relax if due. Returns True when a relaxation ran."""
        if self.ema is None or step % self.every != 0:
            return False
        if len(expert_params) != self.n:
            raise ValueError(
                f"{len(expert_params)} expert lists != {self.n}")
        # snapshot so every pairwise pull reads pre-step weights
        ref = [[w.detach().clone() for w in ws]
               for ws in expert_params]
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                if self.edges is not None \
                        and (i, j) not in self.edges:
                    continue
                c = self.lam * float(self.ema[i, j])
                for wi, wj in zip(expert_params[i], ref[j]):
                    wi.add_(c * (wj - wi))
        return True


@torch.no_grad()
def merge_experts(
        expert_params: list[list[torch.Tensor]]
) -> list[torch.Tensor]:
    """Ship-time collapse: average E experts into one weight list.

    Note the merged model is dense-PLUS-SCALAR-GATE, not purely
    dense: a top-1 switch MoE scales the FFN output by the winning
    router probability, and that per-token scalar survives the
    merge. Exporters must keep the router (one [E, d] matrix) for
    the max-probability scalar, or fold an approximation.
    """
    if not expert_params:
        raise ValueError("no experts")
    return [torch.stack([ws[k] for ws in expert_params]).mean(0)
            for k in range(len(expert_params[0]))]
