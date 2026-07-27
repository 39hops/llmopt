"""Rational-snap distillation (RIFF 2026-07-27, Artin's infinite-precision
push, rung a): snap every 2-D weight of a gated crystal to the nearest
fraction p/q with denominator q <= Q, then gate the snap. Asks "do
trained weights want simple exact numbers?" as a COMPRESSION question
(precision doctrine stays closed; E3 is its sole reopening).

Snap is exact-best over the denominator range (vectorized sweep: for
each q, candidate round(w*q)/q; keep per-element argmin |w - cand|),
so 1/3 is reachable at Q>=3 — the point of the rung vs dyadic quant.

Usage: rational_snap.py <ckpt_in> <Q> <ckpt_out>
Paired arms gate on the SAME device (instrument fence: Mac gate
numbers never compare to cuda gate numbers).
"""
import sys

import torch

ckpt_in, Q, ckpt_out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sd = torch.load(ckpt_in, map_location="cpu")
tot = snapped = 0
changed = []
for k, w in sd.items():
    tot += w.numel()
    if w.ndim != 2 or not w.is_floating_point():
        continue
    wf = w.float()
    best = torch.round(wf)  # q = 1
    err = (wf - best).abs()
    for q in range(2, Q + 1):
        cand = torch.round(wf * q) / q
        e = (wf - cand).abs()
        m = e < err
        best = torch.where(m, cand, best)
        err = torch.where(m, e, err)
    sd[k] = best.to(w.dtype)
    snapped += w.numel()
    changed.append((k, (best != wf).float().mean().item()))
torch.save(sd, ckpt_out)
mean_moved = sum(c for _, c in changed) / max(len(changed), 1)
print(f"Q={Q}: snapped {snapped}/{tot} params over {len(changed)} "
      f"tensors, mean moved-frac {mean_moved:.4f} -> {ckpt_out}",
      flush=True)
