"""RUNG S0 (pre-reg V4-RUNG-D + S0): is the entropy-coded form of a
DeepSeek-V4-Flash expert EXECUTABLE, or only an archive?

Registered to FAIL. House doctrine already answers in the negative --
P6-v2 never benched decode-side rANS because "the runtime twin remains
crystal5/int8", citing the C2b lesson that the bit-packed form is
DIRECTLY executable. And the prize is 8.3%: 13.37 MB raw against
12.26 MB coded, MEASURED here. (Spec v3 said 15.6% against 11.29 MB --
that was the merged-lattice rate, which is weight-exact but byte-LOSSY,
substituted for the byte-lossless one.) So this cell exists to convert
an inference the spec has been carrying into a number, not to find a win.

Method: the coder path of llmopt/quantize/pack.py:rans_size --
constriction Categorical(perfect=False), AnsCoder -- on the cached
nibble and E8M0 scale streams of one expert, with this file's byte count
asserted equal to rans_size's. Encode is timed separately; the reported
figure is DECODE alone, best of NREP, single-threaded. The parallel
projection is printed too, because the streams are independent and
"131x" is a property of one Python-bound core, not of the format.

The throughput that matters is bytes of DELIVERED WEIGHT per second, not
symbols per second: a decoded nibble is half a byte of the packed fp4
form the runtime would actually execute. Both are reported, and the
comparison against the streaming bar uses the packed-byte figure.

FENCE: this bounds THIS implementation (one Python-bound coder, one
machine). It is not a statement about rANS as a technique; a SIMD or GPU
decoder is a different instrument. Even the free order of magnitude --
all cores, perfectly parallel -- leaves the coded form ~12x under the
measured 3.5-4.5 GB/s NVMe rate, so the conclusion has real headroom.

Env: EXPERT (default layers.22.ffn.experts.0), NREP (default 5),
     CACHE (default checkpoints/v4flash_sample).
Usage: .venv/bin/python scratch/v4flash_s0.py
"""
import hashlib
import importlib.metadata
import json
import os
import struct
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
import v4flash_rungA as RA  # noqa: E402
from llmopt.quantize.pack import rans_size  # noqa: E402

CACHE = os.environ.get("CACHE", "checkpoints/v4flash_sample")
EXPERT = os.environ.get("EXPERT", "layers.22.ffn.experts.0")
NREP = int(os.environ.get("NREP", "5"))
OUT = "logs/opus/v4_s0.jsonl"
# The streaming arithmetic in spec v3 assumes a 5 GB/s pipe; a decoder
# slower than the pipe is the binding constraint, not the network.
# Measured on this machine, not assumed: Apple-silicon NVMe sequential
# read is 3.5-4.5 GB/s (F_NOCACHE), so 5 GB/s is 15-30% optimistic. Kept
# as the spec's stated bar; the honest range is printed alongside.
BAR_GBPS = 5.0
CONFIG = f"{RA.REPO}/config.json"


def vendor_shape():
    """Route width, layer count and shared-expert bytes -- READ, not typed.

    The first version hardcoded 6 and 43 in an f-string. Both happened to
    be right, but n_shared_experts=1 was sitting next to them in the same
    config and was missed, which undercounted per-token traffic by 31%
    (reviewer catch, 2026-08-03). A literal in a measurement script is an
    unasserted claim.
    """
    with urllib.request.urlopen(CONFIG) as r:
        cfg = json.loads(r.read())
    RA.SHARD = 24
    hdr, _, _ = RA.header()
    shared = sum(hi - lo for k, v in hdr.items()
                 if ".shared_experts." in k and k.startswith("layers.22.")
                 for lo, hi in [v["data_offsets"]])
    assert shared > 0, "no shared-expert tensors found in shard 24"
    return (int(cfg["num_experts_per_tok"]), int(cfg["num_hidden_layers"]),
            int(cfg["n_shared_experts"]), shared)


def blob(name, nbytes=None):
    """Cached bytes, cache-integrity checked, cold-fetching if absent.

    NOT "sha-pinned" in the strong sense: the .sha file is derived from
    these same bytes at write time, so the comparison detects disk rot
    and nothing else. There is no independent vendor pin to compare
    against (contrast RA.SILU_SHA, which is a real one). The expected
    LENGTH from the shard header is the check that has teeth.
    """
    p = os.path.join(CACHE, name.replace("/", "_") + ".bin")
    if not os.path.exists(p) or not os.path.exists(p[:-4] + ".sha"):
        RA.SHARD = 24
        hdr, url, base = RA.header()
        return RA.cached(name, hdr, url, base), None
    raw = open(p, "rb").read()
    want = open(p[:-4] + ".sha").read().strip()
    got = hashlib.sha256(raw).hexdigest()
    assert got == want, f"{name}: sha DISAGREE {got} != {want}"
    assert nbytes is None or len(raw) == nbytes, (
        f"{name}: {len(raw)} bytes, header says {nbytes}")
    return raw, got


def nibbles(raw):
    b = np.frombuffer(raw, dtype=np.uint8)
    out = np.empty(b.size * 2, dtype=np.uint8)
    out[0::2] = b & 0x0F
    out[1::2] = b >> 4
    return out


def bench(sym, nrep):
    """Encode once, decode nrep times; return (bytes, best decode s, enc s).

    The coder path is byte-for-byte llmopt/quantize/pack.py:rans_size --
    same rebase, same Categorical(perfect=False), same AnsCoder, same
    size formula. It is re-implemented here ONLY because rans_size
    returns a size and not a timing; the size it returns is asserted
    against this function's below, so the two cannot silently diverge.
    """
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
    # Hand-rolled equivalent of rans_size's verify=True (hazard 1): a
    # throughput number from an unverified decode measures the wrong
    # function. Checks the LAST rep; all reps are identical by
    # construction (same model, same stream).
    assert (np.asarray(out) == sym0).all(), "rANS roundtrip failed"
    nb = 4 * len(comp) + 4 * len(counts) + 8
    ref, _ = rans_size(sym, verify=False)      # the in-tree coder agrees
    assert nb == ref, f"local coder {nb} != pack.rans_size {ref}"
    return nb, best, t_enc


def main():
    os.makedirs("logs/opus", exist_ok=True)
    # constriction exposes no __version__; getattr(..., "unknown") always
    # fired, so every logged row carried "unknown" while the pre-reg
    # registered "constriction 0.5.0" (reviewer catch, 2026-08-03).
    ver = importlib.metadata.version("constriction")
    assert ver.startswith("0.5."), f"coder version drifted: {ver}"
    topk, nlayer, nshared, shared_bytes = vendor_shape()
    print(f"[S0] constriction {ver} | expert {EXPERT} | best of {NREP} | "
          f"topk {topk} | layers {nlayer} | shared experts {nshared} "
          f"({shared_bytes/1e6:.2f} MB/layer)", flush=True)
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
            # codes only: the E8M0 scales must also decode before the
            # weight is deliverable, so this is ~2% optimistic. The
            # whole-expert figure below is the one that is comparable.
            "decode_MBps_codes_only": len(wraw) / ct / 1e6,
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
              f"decode {ct*1e3:.0f} ms = {row['decode_MBps_codes_only']:.1f} MB/s "
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
    print(f"[S0] cores {os.cpu_count()}: streams are independent, so a "
          f"parallel decoder is ~{mbps*os.cpu_count()/1000:.2f} GB/s -- still "
          f"{BAR_GBPS*1000/(mbps*os.cpu_count()):.0f}x short")
    print(f"[S0] a {topk}-expert layer batch costs {topk*t*1e3:.0f} ms of "
          f"decode alone; {nlayer} layers = {nlayer*topk*t:.1f} s/token")
    # Per-token traffic. n_shared_experts runs on EVERY token in EVERY
    # layer and is not routed, so it belongs in the total.
    routed = topk * raw_mb
    print(f"[S0] per-token traffic, packed fp4: {nlayer} x ({topk} x "
          f"{raw_mb:.2f} + {shared_bytes/1e6:.2f} shared) = "
          f"{nlayer*(routed + shared_bytes/1e6)/1000:.2f} GB "
          f"({nlayer*routed/1000:.2f} GB routed only)")
    for bar in (BAR_GBPS, 4.0, 3.5):
        print(f"[S0]   at {bar:.1f} GB/s -> "
              f"{bar/(nlayer*(routed + shared_bytes/1e6)/1000):.2f} tok/s")


if __name__ == "__main__":
    main()
