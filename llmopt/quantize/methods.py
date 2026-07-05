"""Weight quantization methods beyond round-to-nearest: GPTQ, AWQ, HQQ.

All operate on one weight matrix W [out, in] and return a fake-quant
(dequantized fp) matrix, so quality is comparable directly against the
RTN baseline in sensitivity.py. What each adds over RTN:

- GPTQ (OBQ-style): quantize columns one at a time and *compensate* —
  each column's rounding error is folded into the not-yet-quantized
  columns using the inverse Hessian (H = X^T X from calibration data),
  so error that matters for X @ W^T gets cancelled downstream.
- AWQ: no per-weight compensation; instead find a per-in-channel scale
  s (grid search on activation magnitude^alpha) so that salient
  channels — the ones with big activations — get finer quantization.
  W' = quant(W * s) / s is mathematically fold-able into the previous
  layer at deploy time.
- HQQ: calibration-free. Optimizes the zero-point per group under a
  robust lp<1 error norm via half-quadratic splitting, absorbing
  outliers that wreck plain min-max ranges.
"""

from __future__ import annotations


def rtn(w, bits: int):
    """Symmetric per-row round-to-nearest baseline."""
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return (w / scale).round().clamp(-qmax, qmax) * scale


def gptq(w, hessian, bits: int, damp: float = 0.01):
    """GPTQ: column-serial quantization with inverse-Hessian error
    compensation. hessian: [in, in] = X^T X over calibration inputs."""
    import torch

    w = w.clone().float()
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax

    h = hessian.float().clone()
    h += damp * h.diagonal().mean() * torch.eye(h.shape[0])
    hinv = torch.linalg.inv(h)

    q = torch.zeros_like(w)
    for j in range(w.shape[1]):
        col = w[:, j]
        q[:, j] = (col / scale[:, 0]).round().clamp(-qmax, qmax) * scale[:, 0]
        err = (col - q[:, j]) / hinv[j, j]
        # push the error onto remaining columns (they'll absorb it when
        # their turn comes)
        w[:, j + 1 :] -= torch.outer(err, hinv[j, j + 1 :])
    return q


def awq(w, x, bits: int, grid: int = 20):
    """AWQ: per-in-channel scale search. x: [n, in] calibration
    activations. Returns (w_fake_quant, scales)."""
    import torch

    act = x.abs().mean(0).clamp(min=1e-8)  # per-in-channel magnitude
    best, best_loss, best_s = None, float("inf"), None
    ref = x @ w.T
    for i in range(grid + 1):
        alpha = i / grid
        s = act.pow(alpha)
        s = s / (s.max() * s.min()).sqrt()  # normalize the range
        wq = rtn(w * s, bits) / s
        loss = float((x @ wq.T - ref).pow(2).mean())
        if loss < best_loss:
            best, best_loss, best_s = wq, loss, s
    return best, best_s


def hqq(w, bits: int, group_size: int = 64, iters: int = 20, p: float = 0.7):
    """HQQ: per-group asymmetric quant, zero-point optimized under a
    robust |.|^p norm by half-quadratic splitting. Calibration-free."""
    import torch

    out_f, in_f = w.shape
    g = w.reshape(-1, group_size)  # [groups, group_size]
    qmax = 2**bits - 1
    lo, hi = g.min(1, keepdim=True).values, g.max(1, keepdim=True).values
    scale = ((hi - lo) / qmax).clamp(min=1e-8)
    zero = lo

    def dequant(z):
        q = ((g - z) / scale).round().clamp(0, qmax)
        return q * scale + z

    beta = 1.0
    for _ in range(iters):
        wr = dequant(zero)
        e = g - wr
        # lp-norm proximal (generalized soft threshold): shrinks small
        # residuals to zero, keeps outliers mostly intact
        shrink = torch.clamp(
            e.abs() - (e.abs().clamp(min=1e-8)).pow(p - 1) / beta, min=0
        )
        we = e.sign() * shrink
        q = ((g - zero) / scale).round().clamp(0, qmax)
        zero = (g - we - q * scale).mean(1, keepdim=True)
        beta *= 1.05
    return dequant(zero).reshape(out_f, in_f)
