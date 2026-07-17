"""Ternary-from-birth (BitNet-style QAT) — the wiring-thesis re-ask.

Post-hoc ternarization of the fp32 crystal kept 24/120 (the signed
graph walks); BitNet-class results train the constraint FROM BIRTH.
This wraps every block Linear in a straight-through ternary forward
(fp32 latent weights, absmean-scaled {-1,0,+1} function), reuses the
full gen-4 recipe via train_mathnative.main, and saves BOTH the
latent and the deployed (ternarized) snapshot. Gate the deployed one.

The lm head (out_features == vocab) stays full precision — 40-way
logits over ternary features is punishment enough.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_OUT = 40  # head excluded from ternarization


def ternary(w: torch.Tensor) -> torch.Tensor:
    s = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    return torch.where(w.abs() < 0.5 * s,
                       torch.zeros_like(w), torch.sign(w) * s)


class TernaryLinear(nn.Linear):
    def forward(self, x):
        if self.out_features == VOCAB_OUT:
            return F.linear(x, self.weight, self.bias)
        w = self.weight
        wq = w + (ternary(w) - w).detach()  # straight-through
        return F.linear(x, wq, self.bias)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/mathnative_45m_ternary.pt")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--ffn", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=8)
    a = ap.parse_args()

    nn.Linear = TernaryLinear  # build_model picks this up
    import train_mathnative as T
    latent = a.out.replace(".pt", "_latent.pt")
    T.main(v2=False, d=a.d, layers=a.layers, ffn=a.ffn,
           out=latent, heads=a.heads, v21=False, fast=False,
           v22=True, gen4=True)
    # deploy: ternarize the latents (head untouched)
    sd = torch.load(latent, map_location="cpu")
    dep = {}
    for k, W in sd.items():
        if W.dim() == 2 and "emb" not in k and W.shape[0] != VOCAB_OUT:
            dep[k] = ternary(W.float())
        else:
            dep[k] = W
    torch.save(dep, a.out)
    print(f"deployed ternary snapshot -> {a.out}", flush=True)
