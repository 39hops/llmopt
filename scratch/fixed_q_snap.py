"""Fixed-denominator snap (spec addendum 2026-07-27, 'integer twin'):
every 2-D weight -> round(w*q)/q for ONE shared q. Unlike best-rational
(free denominators), this makes W = P/q with integer P — the forward
pass becomes an integer GEMM / q, the road to exact integer inference
(ozaki/FX-V1 substrate). Error bound 1/(2q), vs ~1/Q^2 for best-rational.
Usage: fixed_q_snap.py <ckpt_in> <q> <ckpt_out>
"""
import sys

import torch

ckpt_in, q, ckpt_out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sd = torch.load(ckpt_in, map_location="cpu")
pmax = 0
for k, w in sd.items():
    if w.ndim != 2 or not w.is_floating_point():
        continue
    p = torch.round(w.float() * q)
    pmax = max(pmax, int(p.abs().max()))
    sd[k] = (p / q).to(w.dtype)
torch.save(sd, ckpt_out)
print(f"q={q}: integer twin saved, max |p| = {pmax} "
      f"({'int8' if pmax < 128 else 'int16' if pmax < 32768 else 'int32'}-range) -> {ckpt_out}", flush=True)
