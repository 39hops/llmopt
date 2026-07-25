"""The Lloyd-Max codebook race (pre-reg RESULTS 2026-07-25):
per-output-channel exact 1-D k-means quantizers on the 19M infix
twin, free vs zero-pinned centroids, PTQ-only. Writes one _lm*.pt
checkpoint per arm; gates run separately (MPS, to match baselines).

    .venv/bin/python scratch/lloydmax_race.py checkpoints/mathnative_19m_infixtwin.pt
"""
import sys

import torch


def kmeans_rows(w: torch.Tensor, k: int, pin_zero: bool,
                iters: int = 25) -> torch.Tensor:
    """Exact-enough 1-D k-means per row. w: (rows, cols). Returns
    the quantized tensor. Centroids init at Gaussian quantiles of
    each row's sigma; optional centroid pinned at 0."""
    rows, _ = w.shape
    sig = w.std(dim=1, keepdim=True).clamp(min=1e-12)
    # Gaussian-quantile init, symmetric
    qs = torch.linspace(-1.6, 1.6, k, dtype=w.dtype)
    c = sig * qs[None, :]                       # (rows, k)
    if pin_zero:
        zi = int((qs.abs()).argmin())
        c[:, zi] = 0.0
    for _ in range(iters):
        d = (w[:, :, None] - c[:, None, :]).abs()
        a = d.argmin(dim=2)                     # (rows, cols)
        for j in range(k):
            m = a == j
            cnt = m.sum(dim=1)
            s = torch.where(m, w, torch.zeros_like(w)).sum(dim=1)
            upd = cnt > 0
            if pin_zero and j == zi:
                continue
            c[upd, j] = s[upd] / cnt[upd]
    d = (w[:, :, None] - c[:, None, :]).abs()
    a = d.argmin(dim=2)
    return torch.gather(c, 1, a)


ARMS = {
    "lm2": (4, False), "lm2z": (4, True),
    "lm3": (8, False), "lm3z": (8, True),
    "lmt": (3, True),          # ternary Lloyd-Max, zero-pinned
}


def uniform_rows(w: torch.Tensor, bits: int) -> torch.Tensor:
    lv = 2 ** (bits - 1) - 1   # symmetric int range
    s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / lv
    return torch.round(w / s).clamp(-lv - 1, lv) * s


def main() -> None:
    path = sys.argv[1]
    base = torch.load(path, map_location="cpu")
    for name, (k, pin) in ARMS.items():
        sd = {kk: (kmeans_rows(v.float(), k, pin).to(v.dtype)
                   if v.ndim == 2 else v.clone())
              for kk, v in base.items()}
        out = path.replace(".pt", f"_{name}.pt")
        torch.save(sd, out)
        print(f"{out}: k={k} pin_zero={pin}", flush=True)
    sd = {kk: (uniform_rows(v.float(), 2).to(v.dtype)
               if v.ndim == 2 else v.clone())
          for kk, v in base.items()}
    out = path.replace(".pt", "_int2.pt")
    torch.save(sd, out)
    print(f"{out}: uniform int2", flush=True)


if __name__ == "__main__":
    main()
