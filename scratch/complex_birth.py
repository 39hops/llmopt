"""Complex-FFN birth driver (spec 2026-07-26-complex-zx-program).

Patches train_mathnative.build_model with the complex builder, then
runs the standard trainer path (gen-4, v22, tournament recipe).

    CPLX_ALPHA=none|G5 BIRTH_SEED=1 python scratch/complex_birth.py --epochs 3
Deploy step (G5 only): snap latents pairwise, save *_dep.pt.
"""
import argparse
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

import torch

import complex_model as C
import train_mathnative as T

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--d", type=int, default=384)
ap.add_argument("--layers", type=int, default=8)
ap.add_argument("--ffn", type=int, default=1536)
ap.add_argument("--heads", type=int, default=6)
ap.add_argument("--diet", default=None,
                help="explicit diet jsonl (replaces gen4 corpus)")
ap.add_argument("--tag", default="")
a = ap.parse_args()

alpha = os.environ.get("CPLX_ALPHA", "none")
C.set_alpha(alpha)
T.build_model = C.build_complex_model
out = f"checkpoints/cplx_{alpha}{a.tag}.pt"
kw = {"diet": a.diet} if a.diet else {}
T.main(v2=False, d=a.d, layers=a.layers, ffn=a.ffn, out=out,
       heads=a.heads, v21=False, fast=False, v22=not a.diet,
       gen4=not a.diet, epochs=a.epochs, **kw)

if alpha != "none":
    sd = torch.load(out, map_location="cpu")
    dep = {}
    for k, w in sd.items():
        if w.ndim == 2 and ("gate.weight" in k or "up.weight" in k):
            n = w.shape[0] // 2
            qr, qi = C.g5_quantize(w[:n].float(), w[n:].float())
            dep[k] = torch.cat([qr, qi], 0).to(w.dtype)
        elif w.ndim == 2 and "down.weight" in k:
            n = w.shape[1] // 2
            qr, qi = C.g5_quantize(w[:, :n].float(), w[:, n:].float())
            dep[k] = torch.cat([qr, qi], 1).to(w.dtype)
        else:
            dep[k] = w
    torch.save(dep, out.replace(".pt", "_dep.pt"))
    print(f"deployed {alpha} -> {out.replace('.pt', '_dep.pt')}",
          flush=True)
