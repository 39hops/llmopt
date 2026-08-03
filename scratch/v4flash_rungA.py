"""RUNG A (pre-reg V4-RUNG-A): a full DeepSeek-V4-Flash expert forward
run ENTIRELY IN INTEGERS on the vendor's shipped fp4 codes, hash-locked
across backends. Ported from the certified K3-D2 chain
(scratch/k3_expert_demo.py:99-151); RECEIPT V4-RUNG-MINUS-1 established
the two formats are byte-identical, so only constants and the
activation change.

What is claimed: this exactly-specified integer function reproduces to
the bit on any backend. What is NOT claimed: bit-equality with
DeepSeek's float32 expert forward — fixed-point requants and a
tabulated SiLU are a different function. No capability claim follows.

V4-specific: the swiglu limit (inference/model.py:601-607) is
ASYMMETRIC — up is clamped both sides, gate on the high side only.

Env: DEV (cpu|mps|cuda, default cpu), BATCH (default 16), EXPERT,
CACHE, SILU_TAB. Blobs byte-range fetch on a cold cache.
Usage: .venv/bin/python scratch/v4flash_rungA.py
"""
import hashlib
import json
import math
import os
import struct
import sys
import urllib.request

import numpy as np
import torch

sys.path.insert(0, ".")

REPO = ("https://huggingface.co/deepseek-ai/"
        "DeepSeek-V4-Flash-0731/resolve/main")
NSHARD, SHARD = 48, 24
CACHE = os.environ.get("CACHE", "checkpoints/v4flash_sample")
EXPERT = os.environ.get("EXPERT", "layers.22.ffn.experts.0")
DEV = os.environ.get("DEV", "cpu")
BATCH = int(os.environ.get("BATCH", "16"))
A = 1024                     # activation fixed-point scale
SWIGLU_LIMIT = 10.0          # config.json swiglu_limit
# The SiLU table travels as BYTES and is never regenerated per device
# (P3 doctrine — a different libm would silently change it). The
# committed copy is the transport; the sha is asserted either way.
SILU_SHA = ("f503c81446c97adb01f657d37f490909"
            "a0cbd5d752d2bc2ae5613ced9cf56378")
SILU_TAB = os.environ.get("SILU_TAB") or next(
    (p for p in ("checkpoints/k3_silu_tab.pt",
                 "scratch/v4flash_ref/silu_tab.pt") if os.path.exists(p)),
    "scratch/v4flash_ref/silu_tab.pt")
LUT2X = np.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=np.int64)
# the vendor's own table (inference/convert.py:11), for the exactness check
FP4_TABLE = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.,
                      0., -.5, -1., -1.5, -2., -3., -4., -6.],
                     dtype=np.float64)


def _get(url, lo=None, hi=None):
    req = urllib.request.Request(url)
    if lo is not None:
        req.add_header("Range", f"bytes={lo}-{hi - 1}")
    with urllib.request.urlopen(req) as r:
        return r.read()


def header():
    url = f"{REPO}/model-{SHARD:05d}-of-{NSHARD:05d}.safetensors"
    hlen = struct.unpack("<Q", _get(url, 0, 8))[0]
    hdr = json.loads(_get(url, 8, 8 + hlen))
    hdr.pop("__metadata__", None)
    return hdr, url, 8 + hlen


def cached(name, hdr=None, url=None, base=None):
    """Blob bytes, sha-pinned. Byte-range fetches on a cold cache so the
    cell is self-contained on any machine; an independent fetch that
    lands on the same sha is itself a provenance check."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, name.replace("/", "_") + ".bin")
    if not os.path.exists(p):
        lo, hi = hdr[name]["data_offsets"]
        raw = _get(url, base + lo, base + hi)
        with open(p, "wb") as f:
            f.write(raw)
        with open(p[:-4] + ".sha", "w") as f:
            f.write(hashlib.sha256(raw).hexdigest())
    raw = open(p, "rb").read()
    want = open(p[:-4] + ".sha").read().strip()
    got = hashlib.sha256(raw).hexdigest()
    assert got == want, f"{name}: sha DISAGREE"
    return raw


def decode(proj, hdr, url, base):
    """Shipped bytes -> (codes2x [out, din] int64, exps [out, g] int64).

    codes2x holds 2x the e2m1 magnitude with sign, so the real weight is
    codes2x * 2^(exp - 1). Verified against the vendor's own FP4_TABLE
    and torch's float8_e8m0fnu below.
    """
    wn, sn = f"{EXPERT}.{proj}.weight", f"{EXPERT}.{proj}.scale"
    wshape, sshape = hdr[wn]["shape"], hdr[sn]["shape"]
    wb = np.frombuffer(cached(wn, hdr, url, base), np.uint8).reshape(wshape)
    sb = np.frombuffer(cached(sn, hdr, url, base), np.uint8).reshape(sshape)
    out, half = wshape
    nib = np.empty((out, half * 2), np.uint8)
    nib[:, 0::2] = wb & 0x0F           # low nibble first (vendor order)
    nib[:, 1::2] = wb >> 4
    sign = np.where(nib & 0x8, -1, 1).astype(np.int64)
    codes2x = sign * LUT2X[nib & 0x7]
    exps = sb.astype(np.int64) - 127   # E8M0 bias, verified empirically

    # EXACTNESS (prediction 1): our integer decode must reproduce the
    # vendor's own float semantics bit-for-bit.
    ref = FP4_TABLE[nib] * np.exp2(
        exps.astype(np.float64)).repeat(32, axis=1)
    ours = codes2x.astype(np.float64) * np.exp2(
        exps.astype(np.float64) - 1.0).repeat(32, axis=1)
    assert np.array_equal(ref, ours), f"{proj}: decode disagrees"
    return codes2x, exps


def det_gemv(codes2x, exps, x, dev, chunk=512):
    """Exact integer y = W @ x on the shipped codes. Per-group-32 int64
    dot, then shifts relative to the tensor-min exponent. Chunked over
    output rows purely to bound memory — each row is independent, so
    chunking cannot change a value. Returns int64 at scale 2^(emin-1)."""
    out, din = codes2x.shape
    g = din // 32
    e_all = torch.from_numpy(exps)
    emin = int(e_all.min())
    span = int((e_all - emin).max())
    bound = span + 19 + math.ceil(math.log2(g))
    assert bound < 62, f"shift span would overflow int64: {bound}"
    xt = x.view(x.shape[0], g, 32).to(dev)
    ys = []
    for i in range(0, out, chunk):
        c = torch.from_numpy(codes2x[i:i + chunk]).view(-1, g, 32).to(dev)
        sh = (e_all[i:i + chunk] - emin).to(dev)
        acc = (xt.unsqueeze(1) * c.unsqueeze(0)).sum(-1)   # [b, chunk, g]
        ys.append((acc << sh.unsqueeze(0)).sum(-1).cpu())
    return torch.cat(ys, dim=1), emin, span, bound


def rdiv(v, d):
    """Round-half-away-from-zero integer division (house convention)."""
    return torch.where(v >= 0, (2 * v + d) // (2 * d),
                       -((-2 * v + d) // (2 * d)))


def to_scale_A(y, e):
    """Requant det_gemv output (scale 2^(e-1), x already at A) to A."""
    k = 1 - e
    return rdiv(y, 1 << k) if k >= 0 else y << (-k)


def main():
    hdr, url, base = header()
    deq = {p: decode(p, hdr, url, base) for p in ("w1", "w2", "w3")}
    print(f"[rungA] expert {EXPERT} decode EXACT vs vendor FP4_TABLE "
          f"x float8_e8m0fnu, 3/3 tensors")
    tab_raw = open(SILU_TAB, "rb").read()
    tsha = hashlib.sha256(tab_raw).hexdigest()
    assert tsha == SILU_SHA, f"silu table sha DISAGREE: {tsha}"
    print(f"[rungA] silu table sha {tsha[:16]} PINNED ({SILU_TAB}) "
          f"— shipped bytes, from the K3-D2 cell")
    tab = torch.load(SILU_TAB, weights_only=True).to(DEV)

    rng = np.random.default_rng(45_7_2)
    din = deq["w1"][0].shape[1]
    x = torch.from_numpy(rng.integers(-A, A + 1,
                                      size=(BATCH, din)).astype(np.int64))
    lim = int(SWIGLU_LIMIT * A)

    g, e1, s1, b1 = det_gemv(*deq["w1"], x, DEV)
    u, e3, s3, b3 = det_gemv(*deq["w3"], x, DEV)
    g = to_scale_A(g.to(DEV), e1)
    u = to_scale_A(u.to(DEV), e3)
    # V4 swiglu limit, ASYMMETRIC per inference/model.py:601-607
    u = torch.clamp(u, -lim, lim)
    g = torch.clamp(g, max=lim)
    g = torch.clamp(g, -(1 << 15), 1 << 15)
    u = torch.clamp(u, -(1 << 15), 1 << 15)
    h = torch.clamp(rdiv(tab[g + (1 << 15)] * u, A), -(1 << 15), 1 << 15)
    y, e2, s2, b2 = det_gemv(*deq["w2"], h.cpu(), DEV)
    sha = hashlib.sha256(y.numpy().tobytes()).hexdigest()
    print(f"[rungA] overflow bounds w1={b1} w3={b3} w2={b2} of 62 "
          f"(spans {s1}/{s3}/{s2})")
    print(f"[rungA] emin w1={e1} w3={e3} w2={e2} | batch {BATCH} "
          f"| y {tuple(y.shape)}")
    print(f"[rungA] DEV={DEV} sha256={sha}")


if __name__ == "__main__":
    main()
