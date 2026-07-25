"""Scalar 4-bit PTQ arms (the tournament's missing bracket point):
P4 powers-of-two ladder, LM-16-zero (k-means), NF4-style quantile
codebook — all per-output-channel, on the 19M infix twin. Rides the
Lloyd-Max race harness; gates run separately on MPS.

    .venv/bin/python scratch/ptq4_arms.py checkpoints/mathnative_19m_infixtwin.pt
"""
import sys

import torch

from lloydmax_race import kmeans_rows


def p4_rows(w: torch.Tensor) -> torch.Tensor:
    """{0, ±1/4, ±1/2, ±1, ±2, ±4, ±8} x per-channel unit (absmax/8):
    15 levels + spare code = the symmetric zero-ful 4-bit ladder."""
    unit = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 8.0
    rungs = torch.tensor([0., .25, .5, 1., 2., 4., 8.], dtype=w.dtype)
    x = (w / unit)
    d = (x.abs()[:, :, None] - rungs[None, None, :]).abs()
    r = rungs[d.argmin(dim=2)]
    return torch.sign(w) * r * unit


def nf4_rows(w: torch.Tensor, k: int = 16) -> torch.Tensor:
    """Equal-mass quantile codebook per channel (NF4-style, but on
    the actual channel distribution, zero NOT guaranteed)."""
    qs = torch.linspace(0, 1, 2 * k + 1, dtype=w.dtype)[1::2]
    c = torch.quantile(w.float(), qs.float(), dim=1).T.to(w.dtype)
    d = (w[:, :, None] - c[:, None, :]).abs()
    return torch.gather(c, 1, d.argmin(dim=2))


def main() -> None:
    path = sys.argv[1]
    base = torch.load(path, map_location="cpu")
    for name, fn in (("p4", p4_rows),
                     ("lm16z", lambda x: kmeans_rows(x, 16, True)),
                     ("nf4", nf4_rows)):
        sd = {k: (fn(v.float()).to(v.dtype) if v.ndim == 2 else v.clone())
              for k, v in base.items()}
        out = path.replace(".pt", f"_{name}.pt")
        torch.save(sd, out)
        print(f"{out}", flush=True)


if __name__ == "__main__":
    main()
