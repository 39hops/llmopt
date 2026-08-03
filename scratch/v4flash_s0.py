"""RUNG S0 (pre-reg V4-RUNG-D + S0): is the entropy-coded form of a
DeepSeek-V4-Flash expert EXECUTABLE, or only an archive?

Registered to FAIL. House doctrine already answers in the negative --
P6-v2 never benched decode-side rANS because "the runtime twin remains
crystal5/int8", citing the C2b lesson that the bit-packed form is
DIRECTLY executable. And the prize is 15.6%: 13.37 MB raw against
11.29 MB coded. So this cell exists to convert an inference the spec has
been carrying into a number, not to discover a win.

Method: the SAME coder path llmopt/quantize/pack.py:108 uses --
constriction Categorical(perfect=False), AnsCoder -- on the cached,
sha-pinned nibble and E8M0 scale streams of one expert. Encode is timed
separately; the reported figure is DECODE alone, best of NREP,
single-threaded.

The throughput that matters is bytes of DELIVERED WEIGHT per second, not
symbols per second: a decoded nibble is half a byte of the packed fp4
form the runtime would actually execute. Both are reported, and the
comparison against the streaming bar uses the packed-byte figure.

FENCE: this bounds THIS implementation (one Python-bound coder, one
core, one machine). It is not a statement about rANS as a technique; a
SIMD or GPU decoder is a different instrument.

Env: EXPERT (default layers.22.ffn.experts.0), NREP (default 5),
     CACHE (default checkpoints/v4flash_sample).
Usage: .venv/bin/python scratch/v4flash_s0.py
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")

CACHE = os.environ.get("CACHE", "checkpoints/v4flash_sample")
EXPERT = os.environ.get("EXPERT", "layers.22.ffn.experts.0")
NREP = int(os.environ.get("NREP", "5"))
OUT = "logs/opus/v4_s0.jsonl"
# The streaming arithmetic in spec v3 assumes a 5 GB/s pipe; a decoder
# slower than the pipe is the binding constraint, not the network.
BAR_GBPS = 5.0


def blob(name):
    """Cached bytes with the sha re-asserted (the K3-D1 protocol)."""
    p = os.path.join(CACHE, name.replace("/", "_") + ".bin")
    raw = open(p, "rb").read()
    want = open(p[:-4] + ".sha").read().strip()
    got = hashlib.sha256(raw).hexdigest()
    assert got == want, f"{name}: sha DISAGREE {got} != {want}"
    return raw, got


def nibbles(raw):
    b = np.frombuffer(raw, dtype=np.uint8)
    out = np.empty(b.size * 2, dtype=np.uint8)
    out[0::2] = b & 0x0F
    out[1::2] = b >> 4
    return out


def bench(sym, nrep):
    """Encode once, decode nrep times; return (bytes, best decode s, enc s)."""
    import constriction
    sym0 = (sym - int(sym.min())).astype(np.int32)
    counts = np.bincount(sym0).astype(np.float64)
    probs = counts / counts.sum()
    model = constriction.stream.model.Categorical(probs, perfect=False)
    t0 = time.perf_counter()
    enc = constriction.stream.stack.AnsCoder()
    enc.encode_reverse(sym0, model)
    comp = enc.get_compressed()
    t_enc = time.perf_counter() - t0
    best, out = float("inf"), None
    for _ in range(nrep):
        t0 = time.perf_counter()
        dec = constriction.stream.stack.AnsCoder(comp)
        out = dec.decode(model, len(sym0))
        best = min(best, time.perf_counter() - t0)
    # verify=True is pinned, per hazard 1: a throughput number from an
    # unverified decode measures the wrong function.
    assert (np.asarray(out) == sym0).all(), "rANS roundtrip failed"
    return 4 * len(comp) + 4 * len(counts) + 8, best, t_enc


def main():
    os.makedirs("logs/opus", exist_ok=True)
    import constriction
    ver = getattr(constriction, "__version__", "unknown")
    print(f"[S0] constriction {ver} | expert {EXPERT} | best of {NREP}",
          flush=True)
    sink = open(OUT, "a")
    tot = {"code_bytes_out": 0, "code_bytes_in": 0, "code_t": 0.0,
           "scale_bytes_out": 0, "scale_bytes_in": 0, "scale_t": 0.0}
    for proj in ("w1", "w2", "w3"):
        wraw, wsha = blob(f"{EXPERT}.{proj}.weight")
        sraw, _ = blob(f"{EXPERT}.{proj}.scale")
        codes, scales = nibbles(wraw), np.frombuffer(sraw, dtype=np.uint8)
        cb, ct, cte = bench(codes.astype(np.int64), NREP)
        sb, st, ste = bench(scales.astype(np.int64), NREP)
        row = {
            "expert": EXPERT, "proj": proj, "sha": wsha[:16],
            "constriction": ver, "nrep": NREP,
            "n_codes": int(codes.size), "n_scales": int(scales.size),
            # delivered payload = the packed fp4 bytes a runtime executes
            "packed_bytes": len(wraw), "coded_bytes": cb,
            "scale_packed_bytes": len(sraw), "scale_coded_bytes": sb,
            "decode_s": ct, "encode_s": cte,
            "scale_decode_s": st, "scale_encode_s": ste,
            "decode_MBps_packed": len(wraw) / ct / 1e6,
            "decode_Msym_per_s": codes.size / ct / 1e6,
            "encode_MBps_packed": len(wraw) / cte / 1e6,
        }
        sink.write(json.dumps(row) + "\n")
        sink.flush()
        tot["code_bytes_out"] += len(wraw)
        tot["code_bytes_in"] += cb
        tot["code_t"] += ct
        tot["scale_bytes_out"] += len(sraw)
        tot["scale_bytes_in"] += sb
        tot["scale_t"] += st
        print(f"[S0] {proj}: {codes.size/1e6:.2f}M codes | "
              f"{len(wraw)/1e6:.2f} MB packed -> {cb/1e6:.2f} MB coded | "
              f"decode {ct*1e3:.0f} ms = {row['decode_MBps_packed']:.1f} MB/s "
              f"packed ({row['decode_Msym_per_s']:.1f} Msym/s) | "
              f"encode {cte*1e3:.0f} ms", flush=True)
    sink.close()
    raw_mb = (tot["code_bytes_out"] + tot["scale_bytes_out"]) / 1e6
    cod_mb = (tot["code_bytes_in"] + tot["scale_bytes_in"]) / 1e6
    t = tot["code_t"] + tot["scale_t"]
    mbps = raw_mb / t
    print(f"\n[S0] whole expert: {raw_mb:.2f} MB stored -> {cod_mb:.2f} MB "
          f"coded ({100*(1-cod_mb/raw_mb):.1f}% saved)")
    print(f"[S0] whole-expert decode {t*1e3:.0f} ms = {mbps:.1f} MB/s of "
          f"delivered weight")
    print(f"[S0] against the {BAR_GBPS:.0f} GB/s streaming bar: "
          f"{BAR_GBPS*1000/mbps:.0f}x SHORT")
    print(f"[S0] a 6-expert layer batch would cost "
          f"{6*t*1e3:.0f} ms of decode alone; 43 layers = {43*6*t:.1f} s/token")


if __name__ == "__main__":
    main()
