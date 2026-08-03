"""RUNG -1 (spec 2026-08-02-v4flash-lossless-recode): read a
DeepSeek-V4-Flash safetensors HEADER by HTTP byte-range and report the
tensor inventory — names, dtypes, shapes, and the implied fp4 scale
granularity. Costs well under 1 MB; downloads no weights.

The header settles what config.json cannot: whether the expert scale
layout is [128,128] blocks (~17 MB of scales, no size rung) or group-32
(~8.7 GB of scales, ~5% of the artifact). The pattern is the booked
K3-D1 extraction cell (scratch/k3_expert_demo.py:44).

Env: SHARD (1-indexed, default 24), NSHARD (default 48).
Usage: .venv/bin/python scratch/v4flash_header.py
"""
import collections
import hashlib
import json
import os
import re
import struct
import urllib.request

REPO = ("https://huggingface.co/deepseek-ai/"
        "DeepSeek-V4-Flash-0731/resolve/main")
NSHARD = int(os.environ.get("NSHARD", "48"))
SHARD = int(os.environ.get("SHARD", "24"))
# bytes per element, by safetensors dtype string
WIDTH = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
         "F8_E8M0": 1, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
         "BOOL": 1, "F4_E2M1": 0.5, "F4_E2M1_X2": 1}


def _get(url, lo=None, hi=None):
    req = urllib.request.Request(url)
    if lo is not None:
        req.add_header("Range", f"bytes={lo}-{hi - 1}")
    with urllib.request.urlopen(req) as r:
        return r.read()


def read_header(shard):
    name = f"model-{shard:05d}-of-{NSHARD:05d}.safetensors"
    url = f"{REPO}/{name}"
    hlen = struct.unpack("<Q", _get(url, 0, 8))[0]
    raw = _get(url, 8, 8 + hlen)
    return name, hlen, raw, json.loads(raw)


def group_of(n):
    """Coarse tensor class, from the name alone."""
    if ".experts." in n:
        return "expert.scale" if "scale" in n else "expert.weight"
    if "shared_expert" in n or "shared_experts" in n:
        return "shared_expert"
    if "gate" in n and "proj" not in n:
        return "router/gate"
    if "index" in n:
        return "index"
    if "hash" in n:
        return "hash"
    if "dspark" in n:
        return "dspark"
    if "embed" in n or "lm_head" in n:
        return "embed/head"
    if "norm" in n:
        return "norm"
    return "attention/other"


def main():
    name, hlen, raw, hdr = read_header(SHARD)
    hdr.pop("__metadata__", None)
    print(f"[v4hdr] {name}")
    print(f"[v4hdr] header {hlen} bytes, sha256 "
          f"{hashlib.sha256(raw).hexdigest()[:16]}, {len(hdr)} tensors")

    agg = collections.defaultdict(
        lambda: {"n": 0, "elems": 0, "bytes": 0, "dtypes": set()})
    for n, t in hdr.items():
        g = agg[group_of(n)]
        elems = 1
        for d in t["shape"]:
            elems *= d
        lo, hi = t["data_offsets"]
        g["n"] += 1
        g["elems"] += elems
        g["bytes"] += hi - lo
        g["dtypes"].add(t["dtype"])
    print(f"\n{'group':18s} {'tensors':>8s} {'elems':>14s} "
          f"{'MB':>9s}  dtypes")
    for k in sorted(agg, key=lambda k: -agg[k]["bytes"]):
        v = agg[k]
        print(f"{k:18s} {v['n']:8d} {v['elems']:14,d} "
              f"{v['bytes']/1e6:9.1f}  {','.join(sorted(v['dtypes']))}")

    # pair each expert weight with its scale to read the granularity
    print("\n[v4hdr] expert weight/scale pairs (granularity check):")
    shown = 0
    for n, t in hdr.items():
        if group_of(n) != "expert.weight":
            continue
        for cand in (n + "_scale_inv", n + "_scale", n + ".scale",
                     re.sub(r"\.weight$", ".weight_scale_inv", n),
                     re.sub(r"\.weight$", ".scale", n)):
            if cand in hdr:
                ws, ss = t["shape"], hdr[cand]["shape"]
                ratio = [round(a / b, 3) if b else None
                         for a, b in zip(ws, ss)]
                print(f"  w {n}")
                print(f"    weight {ws} {t['dtype']} | scale {ss} "
                      f"{hdr[cand]['dtype']} | per-axis ratio {ratio}")
                shown += 1
                break
        if shown >= 3:
            break
    if not shown:
        print("  (no name-matched scale tensor — dumping 6 raw names)")
        for n in list(hdr)[:6]:
            print(f"    {n}  {hdr[n]['shape']}  {hdr[n]['dtype']}")


if __name__ == "__main__":
    main()
