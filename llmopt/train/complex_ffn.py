"""Complex-valued SwiGLU-style FFN (modReLU + genuine complex
multiply), promoted from scratch/complex_model.py.

Measured card (B6, 2026-07-31, Mod diet, paired v the real
crystal): UNSNAPPED complex FFN ties the real control overall
(0.530 v 0.526) and shows a suggestive advantage exactly at the
carry-requiring modulus (k=8: 0.77 v 0.63, ~2.2 sigma, single
seed — resolution-law fenced). The exact-phase zeta-N SNAP
variant measured neutral-to-NEGATIVE on the same diet (0.504,
worst at its own resonant modulus) and is deliberately NOT
promoted — architecture-provided exact rotation is not adopted
even where the diet is periodic (teach-don't-impose, strongest
form). If a diet is carry/composition-heavy, this block is the
candidate; do not expect rotation for free (clock-placement law:
representations follow the diet).

Weight layout matches the scratch lineage so checkpoints
transfer: gate/up rows pair as (re | im) halves along the ffn
dim; down columns pair the same way. `ffn` counts REAL rows, so
there are ffn/2 complex units.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ComplexFFN"]


class ComplexFFN(nn.Module):
    """y = W_d (modReLU(W_g h) * W_u h) over C^(ffn/2).

    modReLU (Arjovsky et al. 1511.06464): relu(|z| + b) * z/|z| —
    magnitude-gated, phase-preserving. The product is a genuine
    complex multiply (the rotation happens in activations; weight
    matrices stay real-parametrized as (re | im) halves).
    """

    def __init__(self, d: int, ffn: int):
        super().__init__()
        if ffn % 2:
            raise ValueError("ffn must be even (re|im pairs)")
        self.f2 = ffn // 2
        self.gate = nn.Linear(d, ffn, bias=False)
        self.up = nn.Linear(d, ffn, bias=False)
        self.down = nn.Linear(ffn, d, bias=False)
        self.mod_b = nn.Parameter(torch.zeros(self.f2))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        f2 = self.f2
        zg = self.gate(h)
        zu = self.up(h)
        gr, gi = zg[..., :f2], zg[..., f2:]
        ur, ui = zu[..., :f2], zu[..., f2:]
        mag = torch.sqrt(gr * gr + gi * gi + 1e-12)
        act = F.relu(mag + self.mod_b) / mag       # modReLU
        ar, ai = gr * act, gi * act
        pr = ar * ur - ai * ui                     # complex multiply
        pi = ar * ui + ai * ur
        return self.down(torch.cat([pr, pi], -1))
