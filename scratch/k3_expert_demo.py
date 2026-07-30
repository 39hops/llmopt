"""K3-D1: the Kimi-K3 single-expert deterministic demo.

Pulls ONE routed expert (layer 45, expert 7; w1/w2/w3 = ~66M params)
out of moonshotai/Kimi-K3 (2.8T) by safetensors byte-range, then:
  (a) exact integer dequant of the MXFP4-pack format (e2m1 codes as
      2x-integers 0..12 signed, times E8M0 power-of-two scales);
  (b) llmopt.quantize.meter on the dequantized weights (per-expert-
      size law: predict M <= 2.0);
  (c) rANS size of the shipped 4-bit code stream v Shannon entropy;
  (d) deterministic integer GEMV hash on a fixed battery — run this
      same script on Mac and 3080, compare the sha256 lines.

Pre-reg: RESULTS.md K3-D1 2026-07-30. Usage:
  .venv/bin/python scratch/k3_expert_demo.py [--dev cpu|mps|cuda]
Expert bytes cache to checkpoints/k3_expert_l45_e7/ (untracked).
"""
import argparse
import hashlib
import json
import math
import os
import struct
import urllib.request

import numpy as np
import torch

REPO = "https://huggingface.co/moonshotai/Kimi-K3/resolve/main"
SHARD = "model-00046-of-000096.safetensors"
PREFIX = "language_model.model.layers.45.block_sparse_moe.experts.7."
CACHE = "checkpoints/k3_expert_l45_e7"
# e2m1 magnitudes {0,.5,1,1.5,2,3,4,6} stored as 2x-integers
LUT2X = np.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=np.int64)


def _get(url, lo=None, hi=None):
    req = urllib.request.Request(url)
    if lo is not None:
        req.add_header("Range", f"bytes={lo}-{hi - 1}")
    with urllib.request.urlopen(req) as r:
        return r.read()


def fetch_expert():
    os.makedirs(CACHE, exist_ok=True)
    meta_p = os.path.join(CACHE, "meta.json")
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p))
    else:
        url = f"{REPO}/{SHARD}"
        (hlen,) = struct.unpack("<Q", _get(url, 0, 8))
        hdr = json.loads(_get(url, 8, 8 + hlen))
        meta = {}
        for name, info in hdr.items():
            if not name.startswith(PREFIX):
                continue
            short = name.removeprefix(PREFIX)
            lo, hi = info["data_offsets"]
            blob = _get(url, 8 + hlen + lo, 8 + hlen + hi)
            with open(os.path.join(CACHE, short + ".bin"), "wb") as f:
                f.write(blob)
            meta[short] = {"dtype": info["dtype"],
                           "shape": info["shape"],
                           "nbytes": hi - lo,
                           "sha": hashlib.sha256(blob).hexdigest()}
        json.dump(meta, open(meta_p, "w"), indent=1)
    tensors, total = {}, 0
    for short, info in meta.items():
        raw = open(os.path.join(CACHE, short + ".bin"), "rb").read()
        assert hashlib.sha256(raw).hexdigest() == info["sha"], short
        assert info["dtype"] in ("U8", "uint8"), info["dtype"]
        tensors[short] = np.frombuffer(raw, np.uint8).reshape(
            info["shape"])
        total += info["nbytes"]
        print(f"  {short:22s} {str(info['shape']):14s} "
              f"sha {info['sha'][:12]}")
    print(f"[extract] {total / 1e6:.1f} MB fetched "
          f"({len(tensors)} tensors)")
    return tensors


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


def det_gemv(codes2x, exps, x, dev):
    """Exact integer y = W @ x on the shipped MXFP4 codes.
    Per-group int64 dot, then shifts relative to the tensor-min
    exponent. Returns int64 [batch, out] at scale 2^(emin-1)."""
    out, din = codes2x.shape
    g = din // 32
    c = torch.from_numpy(codes2x).view(out, g, 32).to(dev)
    e = torch.from_numpy(exps).to(dev)
    emin = int(e.min())
    shift = (e - emin)
    assert int(shift.max()) + 19 + math.ceil(math.log2(g)) < 62, \
        "shift span would overflow int64"
    xt = x.view(x.shape[0], g, 32).to(dev)
    acc = (xt.unsqueeze(1) * c.unsqueeze(0)).sum(-1)   # [b, out, g]
    y = (acc << shift.unsqueeze(0)).sum(-1)
    return y.cpu(), emin


A = 1024  # activation fixed-point scale for the D2 chain
SILU_TAB = "checkpoints/k3_silu_tab.pt"


def chain(deq, dev):
    """K3-D2: full deterministic expert forward y = w2 @ (silu(w1@x)
    * (w3@x)) in integers. SiLU via the shipped table (generated once
    on the Mac, sha-pinned); requants are power-of-two shifts."""
    if not os.path.exists(SILU_TAB):
        x = np.arange(-(1 << 15), (1 << 15) + 1, dtype=np.float64) / A
        tab = np.round(x / (1.0 + np.exp(-x)) * A).astype(np.int64)
        torch.save(torch.from_numpy(tab), SILU_TAB)
        print("[chain] silu table GENERATED (Mac master) — ship it")
    tab = torch.load(SILU_TAB, weights_only=True)
    raw = open(SILU_TAB, "rb").read()
    print(f"[chain] silu table sha {hashlib.sha256(raw).hexdigest()[:16]}")
    tab = tab.to(dev)
    rng = np.random.default_rng(45_7_2)
    x = torch.from_numpy(rng.integers(
        -A, A + 1, size=(64, deq["w1"][0].shape[1])).astype(np.int64))

    def rdiv(v, d):
        return torch.where(v >= 0, (2 * v + d) // (2 * d),
                           -((-2 * v + d) // (2 * d)))

    g, e1 = det_gemv(*deq["w1"], x, dev)
    u, e3 = det_gemv(*deq["w3"], x, dev)
    g = torch.clamp(rdiv(g.to(dev), 1 << (1 - e1)), -(1 << 15), 1 << 15)
    u = torch.clamp(rdiv(u.to(dev), 1 << (1 - e3)), -(1 << 15), 1 << 15)
    h = torch.clamp(rdiv(tab[g + (1 << 15)] * u, A),
                    -(1 << 15), 1 << 15)
    y, e2 = det_gemv(*deq["w2"], h.cpu(), dev)
    hh = hashlib.sha256(y.numpy().tobytes()).hexdigest()
    print(f"[chain] dev={dev} e1={e1} e3={e3} e2={e2} "
          f"sha256={hh}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--chain", action="store_true")
    args = ap.parse_args()
    from llmopt.quantize.meter import meter, meter_group
    from llmopt.quantize.pack import rans_size

    t = fetch_expert()
    names = ["w1", "w3", "w2"]  # gate, up: [3072,7168]; down: [7168,3072]
    deq, ws = {}, []
    for n in names:
        codes2x, exps, w = dequant(t[n + ".weight_packed"],
                                   t[n + ".weight_scale"])
        # exactness: int reconstruction roundtrips through fp32
        back = np.round(
            w.astype(np.float64).reshape(w.shape[0], -1, 32)
            / np.exp2(exps - 1.0)[..., None]).astype(np.int64)
        assert (back.reshape(codes2x.shape) == codes2x).all(), n
        deq[n] = (codes2x, exps)
        ws.append(torch.from_numpy(w))
        m, k = meter(ws[-1])
        print(f"[meter] {n}: M={m:.2f} kurt={k:.2f} "
              f"shape={list(w.shape)}")
    mg, kg, npar = meter_group(ws)
    print(f"[meter] expert group: M={mg:.2f} kurt={kg:.2f} "
          f"params={npar / 1e6:.1f}M")

    if args.chain:
        chain(deq, args.dev)
        return

    for n in names[:1]:
        codes = deq[n][0] // 1  # signed 2x-ints, |.| <= 12
        try:
            nbytes, ent = rans_size(codes.ravel().astype(np.float64),
                                    verify=True)
            print(f"[rans] {n}: {8 * nbytes / codes.size:.3f} "
                  f"bits/param (H={ent:.3f}) v 4.25 shipped "
                  f"(4b code + 8b/32 scale)")
        except ImportError:
            print("[rans] constriction not installed; skipped")

    rng = np.random.default_rng(45_7)
    for n in names:
        codes2x, exps = deq[n]
        x = torch.from_numpy(rng.integers(
            -1024, 1025, size=(64, codes2x.shape[1])).astype(np.int64))
        y, emin = det_gemv(codes2x, exps, x, args.dev)
        h = hashlib.sha256(y.numpy().tobytes()).hexdigest()
        print(f"[hash] {n} dev={args.dev} emin={emin} "
              f"sha256={h}")


if __name__ == "__main__":
    main()
