"""Deploy a born-rational (RAT_Q) crystal: apply the SAME snap the STE
trained through (s * best p/q, q <= Q, s = per-tensor absmean) to every
2-D weight — the output IS the trained function, exactly on-lattice.
Usage: rat_deploy.py <ckpt_in> <Q> <ckpt_out>
"""
import sys

import torch

ckpt_in, Q, ckpt_out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sd = torch.load(ckpt_in, map_location="cpu")
for k, w in sd.items():
    if w.ndim != 2 or not w.is_floating_point():
        continue
    wf = w.float()
    s = wf.abs().mean().clamp(min=1e-8)
    v = wf / s
    best = torch.round(v)
    err = (v - best).abs()
    for q in range(2, Q + 1):
        c = torch.round(v * q) / q
        e = (v - c).abs()
        m = e < err
        best = torch.where(m, c, best)
        err = torch.where(m, e, err)
    sd[k] = (s * best).to(w.dtype)
torch.save(sd, ckpt_out)
print(f"deployed rational lattice q<={Q} -> {ckpt_out}", flush=True)
