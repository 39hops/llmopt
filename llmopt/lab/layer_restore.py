"""Pure-numpy helpers of a layer-restoration study on the Qwen3.5 arm C
tower (scratch/qwen_layer_restore.py, QWEN-LAYER-RESTORE-0).

  layer_tensor_map       the vendor index entries of one decoder layer
  shard_coverage         which layers each shard holds completely
  sanitize_vendor_tensor the mlx-lm load conventions for a vendor tensor
                         (conv1d [D, 1, K] -> [D, K, 1]; +1 on the five
                         zero-centred RMSNorm families)
  mlx_param_path         vendor tensor name -> mlx-lm parameter path
  boundary_stats         relative L2, per-row relative L2 quantiles,
                         max abs over row RMS, minimum cosine of two
                         activation matrices
  rel_frobenius          relative Frobenius error of two weight matrices
  causal_additive_mask   the float32 additive causal mask of eager
                         attention (0 on and below the diagonal, -inf
                         above)

Nothing here imports mlx or torch, so both halves of the one-layer
oracle (the mlx side in .venv, the torch side in .venv_teacher) share
these definitions.
"""
from __future__ import annotations

import re

import numpy as np

LAYER_PREFIX = "model.language_model.layers."
NORM_SHIFT_SUFFIXES = (".input_layernorm.weight", ".post_attention_layernorm.weight",
                       "model.norm.weight", ".q_norm.weight", ".k_norm.weight")
_LAYER_RE = re.compile(re.escape(LAYER_PREFIX) + r"(\d+)\.")


def _layer_of(name: str):
    m = _LAYER_RE.match(name)
    return int(m.group(1)) if m else None


def layer_tensor_map(weight_map: dict, layer: int) -> dict:
    """{tensor name: shard file} for exactly the given decoder layer.
    Raises KeyError when the index holds no tensor of that layer."""
    out = {k: v for k, v in weight_map.items() if _layer_of(k) == layer}
    if not out:
        raise KeyError(f"no tensors of layer {layer} in the index")
    return out


def shard_coverage(weight_map: dict) -> dict:
    """{shard: {"complete": [layers whose every tensor is in this shard],
    "partial": [layers with tensors here and elsewhere]}}, layers sorted.
    Shards holding no layer tensor are absent."""
    shards_of: dict = {}
    for k, v in weight_map.items():
        L = _layer_of(k)
        if L is not None:
            shards_of.setdefault(L, set()).add(v)
    out: dict = {}
    for L, shards in shards_of.items():
        for s in shards:
            d = out.setdefault(s, {"complete": [], "partial": []})
            d["complete" if len(shards) == 1 else "partial"].append(L)
    for d in out.values():
        d["complete"].sort()
        d["partial"].sort()
    return out


def sanitize_vendor_tensor(name: str, arr: np.ndarray) -> np.ndarray:
    """The conventions mlx-lm's Qwen3.5 sanitizer applies to vendor
    tensors (the same two the tower applies when loading arm C):
    depthwise conv1d weights move to [D, K, 1]; the five zero-centred
    RMSNorm families gain 1.0. Every other tensor passes unchanged."""
    if name.endswith("conv1d.weight") and arr.ndim == 3 and arr.shape[-1] != 1:
        return np.ascontiguousarray(np.moveaxis(arr, 2, 1))
    if arr.ndim == 1 and any(name.endswith(s) for s in NORM_SHIFT_SUFFIXES):
        return arr + 1.0
    return arr


def mlx_param_path(name: str) -> str:
    """Vendor tensor name -> the mlx-lm Model parameter path (the tower's
    mapping: model.language_model.X -> language_model.model.X, others
    under language_model)."""
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model."):]
    return "language_model." + name


def boundary_stats(a, b) -> dict:
    """Difference statistics of two [n, d] activation matrices (a is the
    reference). Per-row relative L2 is ||a_i - b_i|| / ||a_i||; the max
    abs is scaled by the RMS of the reference row it occurs in."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"shapes {a.shape} and {b.shape} must be equal 2-d")
    d = a - b
    an = np.sqrt((a * a).sum(1))
    dn = np.sqrt((d * d).sum(1))
    row_rel = dn / np.maximum(an, np.finfo(np.float64).tiny)
    rms = np.sqrt((a * a).mean(1))
    max_abs_rows = np.abs(d).max(1)
    cos = (a * b).sum(1) / np.maximum(an * np.sqrt((b * b).sum(1)), np.finfo(np.float64).tiny)
    q = np.quantile(row_rel, [0.5, 0.9, 0.99])
    return {"n_rows": int(a.shape[0]),
            "rel_l2": float(np.sqrt((d * d).sum()) / max(np.sqrt((a * a).sum()), np.finfo(np.float64).tiny)),
            "row_rel_l2_median": float(q[0]), "row_rel_l2_p90": float(q[1]), "row_rel_l2_p99": float(q[2]),
            "row_rel_l2_max": float(row_rel.max()),
            "max_abs": float(np.abs(d).max()),
            "max_abs_over_row_rms": float((max_abs_rows / np.maximum(rms, np.finfo(np.float64).tiny)).max()),
            "cosine_min": float(cos.min()), "ref_row_rms_median": float(np.median(rms))}


def rel_frobenius(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.finfo(np.float64).tiny))


def causal_additive_mask(n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.float32)
    m[np.triu(np.ones((n, n), dtype=bool), 1)] = -np.inf
    return m
