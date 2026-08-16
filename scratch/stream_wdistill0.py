"""STREAM-WDISTILL-0 (pre-reg RESULTS 2026-08-16, L30921).

PASS 0 ONLY in this revision: shard HEADERS, no weight bytes. Reports
the exact dims, the checkpoint revision, the full vendor payload
accounting for one routed expert layer INCLUDING SCALES, and the
pinned layer byte budget the race is matched against.

Arms A-E are deliberately NOT built here. Artin's gate: PASS 0
lands, the pre-reg is amended (the released FP4 artifact cannot
simultaneously be the reconstruction target AND a nonzero-error
baseline), representation dtypes/overheads are frozen, and only
then does the race get written.

NEW SIBLING, not an edit: scratch/v4flash_f1c.py / f1d.py /
v4flash_census.py / v4flash_rungA.py are CODEMAP-frozen evidence.
Their header/byte-range logic is the reference and is reimplemented
here, never modified. Unlike f1c.fetch(), nothing here writes a
weight cache — PASS 0 reads headers and stops.

    .venv/bin/python -u scratch/stream_wdistill0.py
    LAYER=22 SHARD_LO=20 SHARD_HI=28 .venv/bin/python -u scratch/stream_wdistill0.py
"""
import json
import os
import struct
import sys
import time
import urllib.request
from collections import defaultdict

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REPO = f"https://huggingface.co/{MODEL}/resolve/main"
API = f"https://huggingface.co/api/models/{MODEL}"
NSHARD = 48
LAYER = int(os.environ.get("LAYER", "22"))
SHARD_LO = int(os.environ.get("SHARD_LO", "20"))
SHARD_HI = int(os.environ.get("SHARD_HI", "28"))
OUT = "logs/streamwd/pass0.jsonl"

# bytes per element. I8 carries TWO packed fp4 codes — the element
# count is 2x the byte count, handled explicitly below (the census
# convention, scratch/v4flash_census.py).
NEL = {"I8": 1, "F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2, "F16": 2,
       "F32": 4, "I32": 4, "I64": 8}
PACKED_FP4 = {"I8"}          # two codes per byte
SCALE_DTYPES = {"F8_E8M0", "F8_E4M3", "BF16", "F32"}


def _get(url, lo=None, hi=None):
    req = urllib.request.Request(
        url, headers={"User-Agent": "llmopt-streamwd/0 (research)"})
    if lo is not None:
        req.add_header("Range", f"bytes={lo}-{hi - 1}")
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    if lo is not None and len(raw) != hi - lo:
        raise AssertionError(f"truncated: {len(raw)} != {hi - lo}")
    return raw


def revision():
    """The checkpoint revision actually being read (provenance, derived)."""
    with urllib.request.urlopen(
            urllib.request.Request(
                API, headers={"User-Agent": "llmopt-streamwd/0"}),
            timeout=60) as r:
        meta = json.loads(r.read())
    return {"sha": meta.get("sha"), "lastModified": meta.get("lastModified")}


def header(shard):
    """(name -> entry) for one shard, from its safetensors header only."""
    fn = f"model-{shard:05d}-of-{NSHARD:05d}.safetensors"
    hlen = struct.unpack("<Q", _get(f"{REPO}/{fn}", 0, 8))[0]
    hdr = json.loads(_get(f"{REPO}/{fn}", 8, 8 + hlen))
    hdr.pop("__metadata__", None)
    return fn, 8 + hlen, hdr


def is_routed_layer(name, layer):
    return (f"layers.{layer}.ffn.experts." in name
            and ".shared_experts." not in name)


def main():
    os.makedirs("logs/streamwd", exist_ok=True)
    t0 = time.time()
    rev = revision()
    print(f"[pass0] model {MODEL}", flush=True)
    print(f"[pass0] revision {rev}", flush=True)

    found = {}          # name -> dict
    hdr_bytes = 0
    shards_hit = []
    for s in range(SHARD_LO, SHARD_HI + 1):
        fn, base, hdr = header(s)
        hdr_bytes += base
        hits = {k: v for k, v in hdr.items() if is_routed_layer(k, LAYER)}
        if hits:
            shards_hit.append(fn)
            for k, v in hits.items():
                lo, hi = v["data_offsets"]
                found[k] = {"shard": fn, "base": base, "lo": lo, "hi": hi,
                            "dtype": v["dtype"], "shape": v["shape"],
                            "bytes": hi - lo}
        print(f"[pass0] shard {s:02d}: header {base}B, "
              f"layer-{LAYER} expert tensors {len(hits)}", flush=True)

    if not found:
        raise SystemExit(f"no layer-{LAYER} routed expert tensors in "
                         f"shards {SHARD_LO}..{SHARD_HI}")

    # --- group by projection and by expert
    by_proj = defaultdict(list)
    experts = set()
    for k, v in found.items():
        tail = k.split(f"layers.{LAYER}.ffn.experts.", 1)[1]
        eid, rest = tail.split(".", 1)
        experts.add(int(eid))
        by_proj[rest].append(v)

    print(f"\n[pass0] experts found: {len(experts)} "
          f"(ids {min(experts)}..{max(experts)})", flush=True)
    print(f"[pass0] shards carrying them: {shards_hit}", flush=True)

    # --- payload accounting, weights vs scales, per projection
    print("\n[pass0] PER-PROJECTION ACCOUNTING", flush=True)
    tot_bytes = tot_elems = 0
    acct = {}
    for rest in sorted(by_proj):
        vs = by_proj[rest]
        dt = {v["dtype"] for v in vs}
        shapes = {tuple(v["shape"]) for v in vs}
        nby = sum(v["bytes"] for v in vs)
        assert len(dt) == 1 and len(shapes) == 1, (rest, dt, shapes)
        dt, shape = dt.pop(), shapes.pop()
        n_per = 1
        for d in shape:
            n_per *= d
        elems = n_per * len(vs) * (2 if dt in PACKED_FP4 else 1)
        kind = "SCALE" if ("scale" in rest or dt in {"F8_E8M0"}) else "WEIGHT"
        acct[rest] = {"dtype": dt, "shape": list(shape), "count": len(vs),
                      "bytes": nby, "elems": elems, "kind": kind}
        tot_bytes += nby
        tot_elems += elems if kind == "WEIGHT" else 0
        print(f"  {rest:28s} {kind:6s} {dt:8s} shape {str(shape):16s} "
              f"x{len(vs):4d}  {nby/2**20:9.2f} MiB", flush=True)

    w_bytes = sum(a["bytes"] for a in acct.values() if a["kind"] == "WEIGHT")
    s_bytes = sum(a["bytes"] for a in acct.values() if a["kind"] == "SCALE")
    print(f"\n[pass0] WEIGHT payload {w_bytes/2**20:10.2f} MiB", flush=True)
    print(f"[pass0] SCALE  payload {s_bytes/2**20:10.2f} MiB "
          f"({100*s_bytes/max(tot_bytes,1):.2f}% of layer)", flush=True)
    print(f"[pass0] TOTAL  payload {tot_bytes/2**20:10.2f} MiB "
          f"= {tot_bytes} B  <-- PINNED LAYER BUDGET REFERENCE", flush=True)
    print(f"[pass0] weight elements {tot_elems:,}", flush=True)
    if tot_elems:
        print(f"[pass0] vendor effective rate "
              f"{8*tot_bytes/tot_elems:.4f} bits/weight "
              f"(scales included)", flush=True)

    row = {"pass": 0, "model": MODEL, "revision": rev, "layer": LAYER,
           "n_experts": len(experts), "shards": shards_hit,
           "header_bytes_read": hdr_bytes,
           "projections": acct,
           "weight_bytes": w_bytes, "scale_bytes": s_bytes,
           "total_bytes": tot_bytes, "weight_elems": tot_elems,
           "bits_per_weight_incl_scales": (
               8 * tot_bytes / tot_elems if tot_elems else None),
           "wall_s": round(time.time() - t0, 1)}
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"\n[pass0] -> {OUT} (headers only, {hdr_bytes/2**20:.1f} MiB "
          f"read, ZERO weight bytes) wall {row['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
