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

Undefined statistics are explicit (2026-10-16): ``meter_detailed``
returns ``MeterResult(m_bits, kurtosis, flags)`` where a statistic the
input cannot define is NaN and ``flags`` names why; ``meter`` returns
the same two numbers as a tuple. Empty, non-2-D and non-finite inputs
raise ValueError. Ordinary inputs produce the exact values the meter
always produced (float32 arithmetic on CPU; tensors on other devices
are moved, which is the only path that ever worked since the code
entropy goes through numpy).

Flags:
  constant         every element equal (in float32): M and kurtosis NaN
  sigma_clamped    some row has sigma below 1e-8 (the grid clamp)
  single_column    one element per row: no row sigma, M NaN
  kurtosis_float64 float32 fourth moment over/underflowed; recomputed
                   in float64 (value differs from float32 only where
                   float32 had none)
"""
from __future__ import annotations

import math
from typing import NamedTuple


class MeterResult(NamedTuple):
    m_bits: float
    kurtosis: float
    flags: tuple[str, ...]


class GroupResult(NamedTuple):
    m_bits: float
    kurtosis: float
    n_params: int
    n_excluded: int
    excluded_flags: dict[str, int]


def _validate(w):
    import torch
    if w.dim() != 2:
        raise ValueError(f"meter expects a 2-D [out, in] tensor, got shape {tuple(w.shape)}")
    if w.numel() == 0:
        raise ValueError(f"meter: empty tensor, shape {tuple(w.shape)}")
    wf = w.detach().to("cpu").float()
    bad = int((~torch.isfinite(wf)).sum())
    if bad:
        raise ValueError(f"meter: {bad} non-finite value(s) in a tensor of shape "
                         f"{tuple(w.shape)}")
    return wf


def _kurtosis(wf) -> tuple[float, tuple[str, ...]]:
    """Kurtosis as the meter has always computed it: the biased fourth
    central moment over the UNBIASED variance squared (a factor
    ((N-1)/N)^2 below Pearson's; 3 for a Gaussian at large N), in
    float32; float64
    only where float32 has no finite answer; NaN only where no precision
    has one (zero variance)."""
    var = wf.var()
    if float(var) == 0.0:
        return math.nan, ("constant",)
    k = float(((wf - wf.mean()) ** 4).mean() / var ** 2)
    if math.isfinite(k):
        return k, ()
    wd = wf.double()
    vard = wd.var()
    if float(vard) == 0.0:
        return math.nan, ("constant",)
    k = float(((wd - wd.mean()) ** 4).mean() / vard ** 2)
    if not math.isfinite(k):
        return math.nan, ("kurtosis_float64",)
    return k, ("kurtosis_float64",)


def meter_detailed(w) -> MeterResult:
    """fp tensor [out, in] -> MeterResult at per-row sigma/2 (see module
    docstring for the flags)."""
    import numpy as np
    import torch

    wf = _validate(w)
    flags: list[str] = []
    kurt, kflags = _kurtosis(wf)
    flags += kflags
    if wf.shape[1] < 2:
        flags.append("single_column")
        return MeterResult(math.nan, kurt, tuple(flags))
    s_raw = wf.std(dim=1, keepdim=True)
    if bool((s_raw < 1e-8).any()):
        flags.append("sigma_clamped")
    if "constant" in flags:
        return MeterResult(math.nan, kurt, tuple(flags))
    s = s_raw.clamp(min=1e-8)
    codes = torch.round(wf * torch.ceil(2.0 / s))
    span = int(codes.max() - codes.min()) + 1
    span_bits = max(1, math.ceil(math.log2(span)))
    _, cnt = np.unique(codes.numpy().ravel(), return_counts=True)
    p = cnt / cnt.sum()
    ent = float(-(p * np.log2(p)).sum())
    return MeterResult(span_bits - ent, kurt, tuple(flags))


def meter(w):
    """fp tensor [out, in] -> (M bits, kurtosis) at per-row sigma/2.
    NaN in a slot means undefined; ``meter_detailed`` says why."""
    r = meter_detailed(w)
    return r.m_bits, r.kurtosis


def meter_group_detailed(tensors) -> GroupResult:
    """Param-weighted (M, kurtosis) over 2-D fp tensors. A tensor whose
    M or kurtosis is undefined is excluded from both means and counted
    in ``n_excluded`` with its flags; ``n_params`` counts the included
    parameters only. All excluded -> NaN means, n_params 0."""
    tot = 0
    wm = wk = 0.0
    n_excluded = 0
    excluded_flags: dict[str, int] = {}
    for w in tensors:
        r = meter_detailed(w)
        n = w.numel()
        if math.isnan(r.m_bits) or math.isnan(r.kurtosis):
            n_excluded += n
            for f in r.flags:
                excluded_flags[f] = excluded_flags.get(f, 0) + 1
            continue
        wm += r.m_bits * n
        wk += r.kurtosis * n
        tot += n
    if tot == 0:
        return GroupResult(math.nan, math.nan, 0, n_excluded, excluded_flags)
    return GroupResult(wm / tot, wk / tot, tot, n_excluded, excluded_flags)


def meter_group(tensors):
    """Param-weighted (M, kurtosis, n_params) over an iterable of 2-D fp
    tensors; undefined tensors excluded (see meter_group_detailed)."""
    r = meter_group_detailed(tensors)
    return r.m_bits, r.kurtosis, r.n_params
