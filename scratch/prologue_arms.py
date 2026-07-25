"""Zero-birth prologue arms (Opus-5 reviewer, 2026-07-25):
S4 symmetry-without-zero PTQ, sparsity control at ternary's
zero-fraction, and the gauge-commutation checkpoint pair.

    .venv/bin/python scratch/prologue_arms.py checkpoints/mathnative_19m_infixtwin.pt
"""
import sys

import torch


def s4_rows(w):
    """{±1/3, ±1} x per-channel absmax: symmetric, 2 bits, NO zero."""
    s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    x = w / s
    mag = torch.where(x.abs() > 2 / 3, torch.ones_like(x),
                      torch.full_like(x, 1 / 3))
    return torch.sign(w) * mag * s


def ternary_rows(w):
    """Absmean ternary (reference zero-fraction source)."""
    s = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-12)
    q = torch.where(w.abs() > 0.7 * s, torch.sign(w), torch.zeros_like(w))
    return q * (w.abs() * q.abs()).sum(dim=1, keepdim=True) / \
        q.abs().sum(dim=1, keepdim=True).clamp(min=1)


def sparse_rows(w):
    """fp32 magnitudes, pruned to the SAME zero-fraction ternary
    would use per row — isolates 'zero = sparsity' from alphabet."""
    s = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-12)
    mask = (w.abs() > 0.7 * s).to(w.dtype)
    return w * mask


def gauge_flip(sd):
    """Sign-flip gauge on half the FFN hidden units: flip rows of
    `up` and matching columns of `down` — (-u)*act(g) through -d_col
    is function-identical in fp32. NEVER flip `gate` (SiLU is not
    odd)."""
    out = {k: v.clone() for k, v in sd.items()}
    ups = sorted(k for k in sd if k.endswith("up.weight"))
    downs = sorted(k for k in sd if k.endswith("down.weight"))
    for uk, dk in zip(ups, downs):
        h = out[uk].shape[0]
        out[uk][: h // 2] *= -1
        out[dk][:, : h // 2] *= -1
    return out, 2 * len(ups)


def main():
    path = sys.argv[1]
    base = torch.load(path, map_location="cpu")
    for name, fn in (("s4", s4_rows), ("sparse", sparse_rows),
                     ("tern", ternary_rows)):
        sd = {k: (fn(v.float()).to(v.dtype) if v.ndim == 2 else v.clone())
              for k, v in base.items()}
        torch.save(sd, path.replace(".pt", f"_{name}.pt"))
        print(f"{name} saved", flush=True)
    flipped, n = gauge_flip(base)
    # gauge-commutation pair: quantize AFTER the gauge transform
    for name, fn in (("gflip_tern", ternary_rows), ("gflip_s4", s4_rows)):
        q = fn
        sd = {k: (q(v.float()).to(v.dtype) if v.ndim == 2 else v.clone())
              for k, v in flipped.items()}
        torch.save(sd, path.replace(".pt", f"_{name}.pt"))
        print(f"{name} saved ({n} mats gauge-flipped pre-quant)", flush=True)


if __name__ == "__main__":
    main()


def m4_rows(w):
    """Asymmetric M4-style PTQ {-1,0,1,2} x per-channel scale — the
    gauge-commutation test's asymmetric arm."""
    s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 2.0
    return torch.round(w / s).clamp(-1, 2) * s
