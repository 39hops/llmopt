"""The capacity meter/dial: a zero-inference predicate for which
quantization regime a model's weights are in (promoted from
scratch/capacity_meter.py; verdicts 2026-07-29/30).

M = span_bits - code_stream_entropy at a per-row sigma/2 grid: the
fixed-width penalty the sigma grid pays to the worst outlier.
Measured dial (sigma-v-calibrated DeltaKL premium, monotone in M):
  born crystals   M 0.8-1.6  -> ~1x (parity; sigma-law domain)
  MoE experts     M 2.0-2.9  -> ~16x (monotone in PER-EXPERT size:
                     Qwen3 5M/exp 2.93, OLMoE 6M 2.85, V3 45M 2.33,
                     K2 40M 2.01)
  web-dense LLMs  M 3.6-3.9  -> ~34x (use max-anchored/calibrated)
Decision rule: sigma-law below M ~ 2 AND a knee-slack deployment
metric (outcome scoring); otherwise per-row max-anchored grids.
Fragility (k_c, flips/token) is an ORTHOGONAL axis this meter does
not see — pair with the calibration probe for the full card.
"""
from __future__ import annotations

import math


def meter(w):
    """fp tensor [out, in] -> (M bits, kurtosis) at per-row sigma/2."""
    import numpy as np
    import torch

    wf = w.float()
    s = wf.std(dim=1, keepdim=True).clamp(min=1e-8)
    codes = torch.round(wf * torch.ceil(2.0 / s))
    span = int(codes.max() - codes.min()) + 1
    span_bits = max(1, math.ceil(math.log2(span)))
    _, cnt = np.unique(codes.numpy().ravel(), return_counts=True)
    p = cnt / cnt.sum()
    ent = float(-(p * np.log2(p)).sum())
    k = float(((wf - wf.mean()) ** 4).mean() / wf.var() ** 2)
    return span_bits - ent, k


def meter_group(tensors):
    """Param-weighted (M, kurtosis, n_params) over an iterable of
    2-D fp tensors."""
    tot = wm = wk = 0
    for w in tensors:
        m, k = meter(w)
        n = w.numel()
        wm += m * n
        wk += k * n
        tot += n
    return wm / max(tot, 1), wk / max(tot, 1), tot
