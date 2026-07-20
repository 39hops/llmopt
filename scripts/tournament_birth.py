"""Alphabet tournament: parameterized discrete-weight birth.
Contestants (real-valued bracket): B {+-1}, T {0,+-1}, M4
{-1,0,1,2}, M5 {0,+-1,+-2}, P2 {0,+-.5,+-1,+-2,+-4}.
STE + fp32 latents (the proven recipe); absmean-family scaling.

    python scripts/tournament_birth.py --alpha M5 --epochs 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn

VOCAB_OUT = 40
LEVELS = {
    "B":  [-1.0, 1.0],
    "T":  [-1.0, 0.0, 1.0],
    "M4": [-1.0, 0.0, 1.0, 2.0],
    "M5": [-2.0, -1.0, 0.0, 1.0, 2.0],
    "P2": [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0],
}
_ALPHA = "T"


def quantize(w: torch.Tensor) -> torch.Tensor:
    lv = torch.tensor(LEVELS[_ALPHA], device=w.device)
    s = w.abs().mean()
    d = (w.unsqueeze(-1) - s * lv).abs()
    return (s * lv)[d.argmin(-1)]


class AlphaLinear(nn.Linear):
    def forward(self, x):
        wq = self.weight + (quantize(self.weight)
                            - self.weight).detach()
        return nn.functional.linear(x, wq, self.bias)


def main() -> None:
    global _ALPHA
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", required=True, choices=list(LEVELS))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=1536)
    ap.add_argument("--heads", type=int, default=6)
    a = ap.parse_args()
    _ALPHA = a.alpha
    out = f"checkpoints/tourn_{a.alpha}.pt"
    real_linear = nn.Linear

    class Patched(AlphaLinear):
        pass
    nn.Linear = Patched
    import train_mathnative as T
    latent = out.replace(".pt", "_latent.pt")
    T.main(v2=False, d=a.d, layers=a.layers, ffn=a.ffn, out=latent,
           heads=a.heads, v21=False, fast=False, v22=True,
           gen4=True, epochs=a.epochs)
    nn.Linear = real_linear
    sd = torch.load(latent, map_location="cpu")
    dep = {}
    for k, W in sd.items():
        if (W.dim() == 2 and "emb" not in k
                and W.shape[0] != VOCAB_OUT):
            dep[k] = quantize(W.float())
        else:
            dep[k] = W
    torch.save(dep, out)
    print(f"deployed {a.alpha} -> {out}", flush=True)


if __name__ == "__main__":
    main()
