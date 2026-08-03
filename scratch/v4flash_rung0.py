"""RUNGS 0/0b/1/2c/3 (pre-reg V4-RUNG-0/1): entropy and lossless rANS
of DeepSeek-V4-Flash's shipped fp4 expert stream, from byte-range
fetches only.

Readouts, all from the same bytes:
  R0   order-0 entropy of the 16-symbol fp4 code stream (bits/param)
  R0b  the same split into sign (2 symbols) and magnitude (8)
  R2c  mean KL(expert || pooled) — does one global table serve all?
  R3   entropy of the E8M0 scale stream
  R1   rANS bytes per tensor, verify=True pinned ON

Per-TENSOR coding only: a 3.5 GB shard is ~28 GB as int32 (the C7 OOM
lesson). Every blob is sha256'd at write and re-asserted at load (the
K3-D1 protocol). Rows stream to logs/opus/v4_rung0.jsonl as they land
so a killed run leaves its measurements behind.

Env: SHARDS (comma list, default "6,24,42"), NEXP (experts per shard,
default 3), CACHE (default checkpoints/v4flash_sample).
Usage: .venv/bin/python scratch/v4flash_rung0.py
"""
import hashlib
import json
import os
import struct
import sys
import urllib.request

import numpy as np

sys.path.insert(0, ".")
from llmopt.quantize.pack import rans_size  # noqa: E402

REPO = ("https://huggingface.co/deepseek-ai/"
        "DeepSeek-V4-Flash-0731/resolve/main")
NSHARD = 48
SHARDS = [int(s) for s in os.environ.get("SHARDS", "6,24,42").split(",")]
NEXP = int(os.environ.get("NEXP", "3"))
CACHE = os.environ.get("CACHE", "checkpoints/v4flash_sample")
OUT = "logs/opus/v4_rung0.jsonl"
# e2m1 magnitudes {0,.5,1,1.5,2,3,4,6} as 2x-integers; sign bit 0x8.
# Verified identical to the vendor's inference/convert.py FP4_TABLE
# (RECEIPT V4-RUNG-MINUS-1).
LUT2X = np.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=np.int64)


def _get(url, lo=None, hi=None):
    req = urllib.request.Request(url)
    if lo is not None:
        req.add_header("Range", f"bytes={lo}-{hi - 1}")
    with urllib.request.urlopen(req) as r:
        return r.read()


def header(shard):
    url = f"{REPO}/model-{shard:05d}-of-{NSHARD:05d}.safetensors"
    hlen = struct.unpack("<Q", _get(url, 0, 8))[0]
    hdr = json.loads(_get(url, 8, 8 + hlen))
    hdr.pop("__metadata__", None)
    return url, 8 + hlen, hdr


def fetch(url, base, name, spec, cache):
    """Byte-range fetch one tensor, sha-pinned on disk."""
    os.makedirs(cache, exist_ok=True)
    safe = name.replace("/", "_")
    blob = os.path.join(cache, safe + ".bin")
    meta = os.path.join(cache, safe + ".sha")
    lo, hi = spec["data_offsets"]
    if not (os.path.exists(blob) and os.path.exists(meta)):
        raw = _get(url, base + lo, base + hi)
        # a 200 (whole file) or a truncated 206 would otherwise sail
        # through: nibbles() accepts any length and the wrong sha would
        # be written as canonical, permanently (reviewer catch).
        assert len(raw) == hi - lo, (
            f"{name}: got {len(raw)} bytes, expected {hi - lo}")
        with open(blob, "wb") as f:
            f.write(raw)
        with open(meta, "w") as f:
            f.write(hashlib.sha256(raw).hexdigest())
    raw = open(blob, "rb").read()
    want = open(meta).read().strip()
    got = hashlib.sha256(raw).hexdigest()
    assert got == want, f"{name}: sha DISAGREE {got} != {want}"
    return raw, got


def nibbles(raw):
    """Unpack packed fp4 bytes to 16-symbol codes, low nibble first."""
    b = np.frombuffer(raw, dtype=np.uint8)
    out = np.empty(b.size * 2, dtype=np.uint8)
    out[0::2] = b & 0x0F
    out[1::2] = b >> 4
    return out


def entropy(sym, k):
    """Order-0 empirical entropy in bits, and the probability vector."""
    cnt = np.bincount(sym.ravel(), minlength=k).astype(np.float64)
    p = cnt / cnt.sum()
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum()), p


def kl(p, q):
    """KL(p || q) in bits, over the support of p."""
    m = p > 0
    assert (q[m] > 0).all(), "pooled table lacks support the expert uses"
    return float((p[m] * np.log2(p[m] / q[m])).sum())


def main():
    os.makedirs("logs/opus", exist_ok=True)
    import constriction
    ver = getattr(constriction, "__version__", "unknown")
    print(f"[v4r0] constriction {ver} | shards {SHARDS} | "
          f"{NEXP} experts/shard", flush=True)
    rows, pooled_counts, sink = [], np.zeros(16), open(OUT, "a")
    for shard in SHARDS:
        url, base, hdr = header(shard)
        experts = sorted({n.rsplit(".", 2)[0] for n in hdr
                          if ".experts." in n and "scale" not in n})
        for ename in experts[:NEXP]:
            for proj in ("w1", "w2", "w3"):
                wname, sname = f"{ename}.{proj}.weight", None
                for c in (f"{ename}.{proj}.weight_scale_inv",
                          f"{ename}.{proj}.scale",
                          f"{ename}.{proj}.weight_scale"):
                    if c in hdr:
                        sname = c
                        break
                if wname not in hdr or sname is None:
                    continue
                wraw, wsha = fetch(url, base, wname, hdr[wname], CACHE)
                sraw, _ = fetch(url, base, sname, hdr[sname], CACHE)
                assert hdr[wname]["dtype"] == "I8", hdr[wname]["dtype"]
                codes = nibbles(wraw)
                want = 2 * int(np.prod(hdr[wname]["shape"]))
                assert codes.size == want, (
                    f"{wname}: {codes.size} codes, header says {want}")
                sign = (codes >> 3).astype(np.uint8)      # 0/1
                mag = (codes & 0x7).astype(np.uint8)      # 0..7
                scales = np.frombuffer(sraw, dtype=np.uint8)
                h_code, p_code = entropy(codes, 16)
                h_sign, _ = entropy(sign, 2)
                h_mag, _ = entropy(mag, 8)
                h_scale, _ = entropy(scales, 256)
                nb, ent_chk = rans_size(codes.astype(np.int64),
                                        verify=True)
                sb, _ = rans_size(scales.astype(np.int64), verify=True)
                pooled_counts += np.bincount(codes, minlength=16)
                row = {
                    "shard": shard, "tensor": wname,
                    "sha": wsha[:16], "n_params": int(codes.size),
                    "n_scales": int(scales.size),
                    "h_code": h_code, "h_sign": h_sign,
                    "h_mag": h_mag, "h_scale": h_scale,
                    "rans_bits_per_param": 8 * nb / codes.size,
                    "rans_scale_bits_per_symbol": 8 * sb / scales.size,
                    "p_code": p_code.tolist(),
                }
                rows.append(row)
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                print(f"[v4r0] {wname.split('.', 2)[-1][:44]:44s} "
                      f"H={h_code:.3f} sign={h_sign:.4f} "
                      f"mag={h_mag:.3f} rANS={row['rans_bits_per_param']:.3f} "
                      f"scaleH={h_scale:.2f}", flush=True)
    sink.close()
    if not rows:
        print("[v4r0] no tensors matched — check naming")
        return
    pooled = pooled_counts / pooled_counts.sum()
    kls = [kl(np.array(r["p_code"]), pooled) for r in rows]
    n = sum(r["n_params"] for r in rows)
    print(f"\n[v4r0] tensors {len(rows)} | params {n:,}")
    print(f"[v4r0] R0  pooled-mean code entropy "
          f"{np.mean([r['h_code'] for r in rows]):.4f} bits/param "
          f"(stored 4.000)")
    print(f"[v4r0] R0b sign {np.mean([r['h_sign'] for r in rows]):.5f} "
          f"| magnitude {np.mean([r['h_mag'] for r in rows]):.4f} of 3")
    print(f"[v4r0] R1  rANS "
          f"{np.mean([r['rans_bits_per_param'] for r in rows]):.4f} "
          f"bits/param, round-trip verified on {len(rows)}/{len(rows)}")
    print(f"[v4r0] R2c mean KL(expert||pooled) {np.mean(kls):.6f} "
          f"bits/param, max {np.max(kls):.6f}")
    print(f"[v4r0] R3  scale entropy "
          f"{np.mean([r['h_scale'] for r in rows]):.3f} bits/symbol, "
          f"rANS {np.mean([r['rans_scale_bits_per_symbol'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
