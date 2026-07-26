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

if torch.cuda.is_available():  # TF32 births: parity-passed 2026-07-18
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

VOCAB_OUT = 40
LEVELS = {
    "B":  [-1.0, 1.0],
    "T":  [-1.0, 0.0, 1.0],
    "M4": [-1.0, 0.0, 1.0, 2.0],
    "M5": [-2.0, -1.0, 0.0, 1.0, 2.0],
    "P2": [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0],
    # the deletion square's fourth corner (Artin 2026-07-25):
    # silence without sign — 1 bit, excitation-only
    "Z1": [0.0, 1.0],
    # symmetry-without-zero at matched 2 bits (the born-S4 v
    # born-M4 law-converting cell; same global-absmean recipe)
    "S4": [-1.0, -1.0 / 3, 1.0 / 3, 1.0],
    # Z1 + SIGNED per-channel scale (is sign sufficient at channel
    # granularity? — the named z1 follow-up cell)
    "Z1S": [0.0, 1.0],
}
_ALPHA = "T"


def quantize(w: torch.Tensor) -> torch.Tensor:
    if _ALPHA == "Z1":
        # per-row POSITIVE scale x {0,1} (pre-reg: zero-or-positive-
        # scale, per-channel)
        s = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        return s * (w.abs() >= 0.5 * s).float()
    if _ALPHA == "Z1S":
        # per-row SIGNED scale x {0,1}: a whole channel shares one
        # sign — sign at channel granularity
        sa = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        m = w.abs() >= 0.5 * sa
        s = (w * m).sum(dim=1, keepdim=True) / m.sum(
            dim=1, keepdim=True).clamp(min=1)
        return s * m.float()
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
    ap.add_argument("--lr", type=float, default=None,
                    help="override birth LR (hot-LR discrete arms)")
    ap.add_argument("--tag", default="",
                    help="checkpoint suffix for variant arms")
    a = ap.parse_args()
    _ALPHA = a.alpha
    out = f"checkpoints/tourn_{a.alpha}{a.tag}.pt"
    real_linear = nn.Linear

    class Patched(AlphaLinear):
        pass
    nn.Linear = Patched
    import train_mathnative as T
    latent = out.replace(".pt", "_latent.pt")
    kw = {"lr": a.lr} if a.lr else {}
    T.main(v2=False, d=a.d, layers=a.layers, ffn=a.ffn, out=latent,
           heads=a.heads, v21=False, fast=False, v22=True,
           gen4=True, epochs=a.epochs, **kw)
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
