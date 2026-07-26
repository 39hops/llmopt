"""Max-asymmetric {0,1,2,3} gauge-commutation arm (pre-reg 2026-07-26).

Builds m4x and gflip_m4x from the 19M infix twin. Gate both with
gate_ckpt (MPS). Floor fence: both <= 5 solves => VOID BY FLOOR.
"""
import sys

import torch

sys.path.insert(0, "scratch")
from prologue_arms import gauge_flip  # noqa: E402

PATH = "checkpoints/mathnative_19m_infixtwin.pt"


def m4x_rows(w):
    """{0,1,2,3} x per-row amax/3 scale — maximally asymmetric."""
    s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 3.0
    return torch.round(w / s).clamp(0, 3) * s


base = torch.load(PATH, map_location="cpu")
sd = {k: (m4x_rows(v.float()).to(v.dtype) if v.ndim == 2 else v.clone())
      for k, v in base.items()}
torch.save(sd, PATH.replace(".pt", "_m4x.pt"))
print("m4x saved", flush=True)

flipped, n = gauge_flip(base)
sd = {k: (m4x_rows(v.float()).to(v.dtype) if v.ndim == 2 else v.clone())
      for k, v in flipped.items()}
torch.save(sd, PATH.replace(".pt", "_gflip_m4x.pt"))
print(f"gflip_m4x saved ({n} mats flipped pre-quant)", flush=True)
