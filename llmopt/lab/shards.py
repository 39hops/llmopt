"""Streamed big-model weights -> instruments -> lake, as one call each.

The missing first-class path (2026-08-11): every piece existed —
byte-range expert fetch (scratch/k3_expert_demo.py, K3-D1 evidence),
the V4-Flash manifest/shard cache (scratch/v4flash_f1c/d.py, V4-F1*
evidence), the capacity meter, the anatomy dot views — but none was a
simple call from inside llmopt, so none got reused. This module is
the adoption. ``dequant`` is the CANONICAL body since 2026-08-12
(Phase 3 module 3): scratch/k3_expert_demo.py holds a
line-count-preserving re-export shim (its booked line citations
survive). The other scratch originals stay frozen as the record;
behavior is pinned by tests/test_shards.py (exact-value battery +
real-shard streaming test).

Typical use (offline anatomy of a 300B-class MoE on 36 GB — weights
stream one expert at a time, nothing accumulates but metrics):

    from llmopt.lab import shards, anatomy, lake
    rows, mats = [], []
    for label, W in shards.iter_v4flash_experts(sample=64, seed=1):
        rows.append(shards.weigh(W, source=label))
        mats.append(W)
    lake.append_weights(rows, lake_dir)
    anatomy.render_dot_views(torch.cat(mats), ...)

Decode speed never enters: this reads weights once and runs
instruments; it is not a chat path.
"""
from __future__ import annotations

import json
import os

import numpy as np

__all__ = ["dequant", "v4flash_manifest", "v4flash_expert",
           "iter_v4flash_experts", "list_v4flash_experts", "weigh"]

V4FLASH_CACHE = "checkpoints/v4flash_f1"

# --- MXFP4 exact dequant — VERBATIM from scratch/k3_expert_demo.py
# (K3-D1 evidence; fixes land in BOTH copies, same commit) ------------

LUT2X = np.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=np.int64)


def dequant(packed, scale):
    """MXFP4-pack -> (codes2x int64 [out,in], exps int64 [out,groups],
    fp32 exact dequant). Two 4-bit codes per byte, low nibble first;
    sign bit 0x8; scale is E8M0: w = e2m1 * 2^(scale-127)."""
    lo, hi = packed & 0x0F, packed >> 4
    nib = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,),
                   np.uint8)
    nib[..., 0::2], nib[..., 1::2] = lo, hi
    sign = np.where(nib & 0x8, -1, 1).astype(np.int64)
    codes2x = sign * LUT2X[nib & 0x7]
    exps = scale.astype(np.int64) - 127
    # exact fp32: codes2x * 2^(exp-1); |w| <= 6 * 2^exp, exact in fp32
    w = (codes2x.reshape(codes2x.shape[0], -1, 32)
         * np.exp2(exps - 1.0)[..., None].astype(np.float64))
    return codes2x, exps, w.reshape(codes2x.shape).astype(np.float32)


# --- V4-Flash shard cache (the 44 GB already on disk) ----------------


def v4flash_manifest(cache: str = V4FLASH_CACHE) -> dict:
    """The cached safetensors index: tensor name ->
    [file, header_end, (lo, hi), dtype, shape]. This reads the cache
    written by the frozen scratch drivers; it does not fetch."""
    p = os.path.join(cache, "manifest_all.json")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} missing — the V4-Flash cache is built by the frozen "
            "scratch/v4flash_f1* drivers, not by this module")
    return json.load(open(p))


def _cached_bin(cache: str, name: str) -> np.ndarray:
    p = os.path.join(cache, name.replace("/", "_") + ".bin")
    return np.frombuffer(open(p, "rb").read(), dtype=np.uint8)


def list_v4flash_experts(cache: str = V4FLASH_CACHE,
                         proj: str = "w1") -> list[tuple[int, int]]:
    """(layer, expert) pairs whose packed weight AND scale for `proj`
    are both in the local cache. Reads the filesystem, not the net."""
    have = set(os.listdir(cache))
    out = []
    for f in have:
        parts = f.split(".")
        if (len(parts) == 8 and parts[0] == "layers"
                and parts[5] == proj and parts[6] == "weight"):
            lay, eid = int(parts[1]), int(parts[4])
            if f.replace(".weight.", ".scale.") in have:
                out.append((lay, eid))
    return sorted(out)


def v4flash_expert(layer: int, expert: int, proj: str = "w1",
                   cache: str = V4FLASH_CACHE):
    """One routed expert's projection, exactly dequantized to a
    torch.float32 matrix [out, in]. Rows are neurons."""
    import torch
    man = v4flash_manifest(cache)
    base = f"layers.{layer}.ffn.experts.{expert}.{proj}"
    _, _, _, wdt, wshape = man[base + ".weight"]
    _, _, _, sdt, sshape = man[base + ".scale"]
    if wdt != "I8" or sdt != "F8_E8M0":
        raise ValueError(f"{base}: unexpected dtypes {wdt}/{sdt}")
    packed = _cached_bin(cache, base + ".weight").reshape(wshape)
    scale = _cached_bin(cache, base + ".scale").reshape(sshape)
    _, _, w = dequant(packed, scale)
    return torch.from_numpy(w)


def iter_v4flash_experts(sample: int | None = None, seed: int = 0,
                         proj: str = "w1",
                         cache: str = V4FLASH_CACHE):
    """Yield ("L<l>E<e>", W fp32) one expert at a time — stream,
    project, free. String-seeded sampling per doctrine."""
    import random
    pairs = list_v4flash_experts(cache, proj)
    if not pairs:
        raise FileNotFoundError(f"no cached {proj} experts in {cache}")
    if sample is not None and sample < len(pairs):
        pairs = random.Random(f"v4flash-{proj}-{seed}").sample(
            pairs, sample)
        pairs.sort()
    for lay, eid in pairs:
        yield f"L{lay}E{eid}", v4flash_expert(lay, eid, proj, cache)


# --- instruments -> one snake_case row -------------------------------


def weigh(W, source: str, model: str = "", proj: str = "") -> dict:
    """Run the desk instruments on one weight matrix. Returns a flat
    snake_case row ready for lake.append_weights: capacity meter M
    (span_bits - code entropy at sigma/2) + kurtosis from
    llmopt.quantize.meter, row-norm stats, shape."""
    from llmopt.quantize.meter import meter
    m_bits, kurt = meter(W)
    norms = W.float().norm(dim=1)
    return {
        "model": model,
        "source": source,
        "proj": proj,
        "n_rows": int(W.shape[0]),
        "n_cols": int(W.shape[1]),
        "meter_m_bits": float(m_bits),
        "kurtosis": float(kurt),
        "row_norm_mean": float(norms.mean()),
        "row_norm_std": float(norms.std()),
        "row_norm_max": float(norms.max()),
    }
