"""Track B2 of the GO of 2026-10-22 (LOSSLESS-BF16-FEASIBILITY-0): practical
REVERSIBLE encodings of the original BF16 weights, measured on the exact
bytes of the two retained, sha-verified vendor shards, with every
compressor/decompressor pair proved exact by comparing the recovered
bytes with the original bytes (never a float tolerance). Deterministic:
single-threaded zstd, static models built from the tensor's own
histogram, fixed block sizes, no randomness.

Methods (per tensor; compressed bytes INCLUDE every dictionary, block
offset table and header this driver would need to decode the tensor
alone, random-access per block of BLOCK elements):
  raw          2 bytes per weight (the baseline)
  zstd-L       zstd --single-thread at level L on the raw bytes
  byteplane+zstd-L   the high byte plane (sign and the top 7 exponent
               bits) and the low byte plane (the exponent's last bit and
               the 7 mantissa bits) compressed separately
  bitshuffle+zstd-L  16 bit planes (np.packbits of each bit across the
               tensor) compressed as one stream
  dfloat11     the exponent byte entropy-coded with a canonical Huffman
               code built from the tensor's histogram (scratch/bf16_huff.c,
               compiled at run time, block-structured with a 64-bit byte
               offset per block), sign and mantissa stored raw as one byte
               per weight (DFloat11's split: entropy code the exponent,
               keep the rest verbatim); dictionary 256 code lengths
  rans-exp     the same split with the exponent coded by a static rANS
               coder (constriction, Categorical model from the tensor's
               histogram, one stream per block of BLOCK elements, 64-bit
               offsets); dictionary 256 x 32-bit frequencies
  rans-16      the full 16-bit pattern coded by a static rANS coder with
               a 65536-way Categorical model (captures the sign and
               mantissa dependence on the exponent that the split
               coders give up); dictionary 65536 x 32-bit frequencies
Per method: compressed bytes, bits per weight, ratio, compression and
decompression wall and throughput (raw bytes per second), peak RSS of
the process so far, and exact (bytes-equal) recovery.
Tensors: every tensor of shard 7 and lm_head from shard 18 (the mtp.*
tensors are not in the text model and are skipped); SMOKE=1 runs the
first two tensors of shard 7 and the first 32 M elements of lm_head.
Receipt logs/bf16_lossless/bench_<commit>.json (refuse-if-exists). Env:
SHARD_DIR, LEVELS ("3,19"), BLOCK (65536), CHUNK (67108864), SMOKE=1.
Entropy figures are the census's (scratch/bf16_entropy_census.py), not
this driver's; this driver measures achieved bytes.
"""
from __future__ import annotations

import ctypes
import heapq
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from bf16_entropy_census import SHARDS, classify, safetensors_header, tensor_view  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
SHARD_DIR = os.path.expanduser(os.environ.get("SHARD_DIR", "~/qwen_vendor_shards/1d4bf0f2"))
LEVELS = [int(v) for v in os.environ.get("LEVELS", "3,19").split(",") if v]
BLOCK = int(os.environ.get("BLOCK", "65536"))
CHUNK = int(os.environ.get("CHUNK", str(1 << 26)))
ZSTD = os.environ.get("ZSTD", "zstd")


def rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def build_huff_lib() -> ctypes.CDLL:
    src = os.path.join(HERE, "bf16_huff.c")
    build = os.environ.get("HUFF_BUILD_DIR") or os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp")
    os.makedirs(build, exist_ok=True)
    so = os.path.join(build, "bf16_huff.dylib")
    subprocess.run(["clang", "-O2", "-shared", "-fPIC", "-o", so, src], check=True)
    lib = ctypes.CDLL(so)
    lib.huff_encode.restype = ctypes.c_int64
    lib.huff_encode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    lib.huff_decode.restype = ctypes.c_int
    lib.huff_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    return lib


def huffman_lengths(counts: np.ndarray, max_len: int = 56) -> np.ndarray:
    """Code lengths of a Huffman code for the 256 symbols (zero for unused
    symbols; a single used symbol gets length 1). Deterministic tie-break
    by (count, symbol)."""
    syms = [int(s) for s in np.flatnonzero(counts)]
    if not syms:
        return np.zeros(256, np.uint8)
    if len(syms) == 1:
        lens = np.zeros(256, np.uint8)
        lens[syms[0]] = 1
        return lens
    heap = [(int(counts[s]), s, (s,)) for s in syms]
    heapq.heapify(heap)
    depth = {s: 0 for s in syms}
    while len(heap) > 1:
        c1, k1, m1 = heapq.heappop(heap)
        c2, k2, m2 = heapq.heappop(heap)
        for s in m1 + m2:
            depth[s] += 1
        heapq.heappush(heap, (c1 + c2, min(k1, k2), m1 + m2))
    lens = np.zeros(256, np.uint8)
    for s, d in depth.items():
        lens[s] = d
    if lens.max() > max_len:
        raise RuntimeError(f"Huffman code length {lens.max()} exceeds {max_len}")
    return lens


def run_zstd(data: bytes, level: int) -> tuple[bytes, float]:
    t0 = time.perf_counter()
    p = subprocess.run([ZSTD, f"-{level}", "--single-thread", "-q", "-c"], input=data, capture_output=True, check=True)
    return p.stdout, time.perf_counter() - t0


def run_unzstd(comp: bytes) -> tuple[bytes, float]:
    t0 = time.perf_counter()
    p = subprocess.run([ZSTD, "-d", "--single-thread", "-q", "-c"], input=comp, capture_output=True, check=True)
    return p.stdout, time.perf_counter() - t0


def method_zstd(u: np.ndarray, level: int) -> dict:
    raw = u.tobytes()
    comp, tc = run_zstd(raw, level)
    back, td = run_unzstd(comp)
    return {"compressed_bytes": len(comp), "metadata_bytes": 0, "compress_s": tc, "decompress_s": td, "exact": back == raw}


def method_byteplane(u: np.ndarray, level: int) -> dict:
    t0 = time.perf_counter()
    hi = (u >> 8).astype(np.uint8).tobytes()
    lo = (u & 0xFF).astype(np.uint8).tobytes()
    tsplit = time.perf_counter() - t0
    ch, tc1 = run_zstd(hi, level)
    cl, tc2 = run_zstd(lo, level)
    bh, td1 = run_unzstd(ch)
    bl, td2 = run_unzstd(cl)
    t0 = time.perf_counter()
    back = (np.frombuffer(bh, np.uint8).astype(np.uint16) << 8) | np.frombuffer(bl, np.uint8).astype(np.uint16)
    tjoin = time.perf_counter() - t0
    return {"compressed_bytes": len(ch) + len(cl) + 16, "metadata_bytes": 16, "compress_s": tsplit + tc1 + tc2,
            "decompress_s": td1 + td2 + tjoin, "exact": bool(np.array_equal(back, u))}


def method_bitshuffle(u: np.ndarray, level: int) -> dict:
    n = u.shape[0]
    t0 = time.perf_counter()
    planes = [np.packbits(((u >> k) & 1).astype(np.uint8)) for k in range(16)]
    data = b"".join(p.tobytes() for p in planes)
    tsh = time.perf_counter() - t0
    comp, tc = run_zstd(data, level)
    back, td = run_unzstd(comp)
    t0 = time.perf_counter()
    plen = planes[0].shape[0]
    arr = np.frombuffer(back, np.uint8)
    rec = np.zeros(n, np.uint16)
    for k in range(16):
        bits = np.unpackbits(arr[k * plen:(k + 1) * plen])[:n].astype(np.uint16)
        rec |= bits << k
    tun = time.perf_counter() - t0
    return {"compressed_bytes": len(comp) + 8, "metadata_bytes": 8, "compress_s": tsh + tc, "decompress_s": td + tun,
            "exact": bool(np.array_equal(rec, u))}


def split_exp_sm(u: np.ndarray):
    ex = ((u >> 7) & 0xFF).astype(np.uint8)
    sm = (((u >> 8) & 0x80) | (u & 0x7F)).astype(np.uint8)
    return ex, sm


def join_exp_sm(ex: np.ndarray, sm: np.ndarray) -> np.ndarray:
    return ((sm.astype(np.uint16) & 0x80) << 8) | (ex.astype(np.uint16) << 7) | (sm.astype(np.uint16) & 0x7F)


def method_dfloat11(u: np.ndarray, lib) -> dict:
    n = u.shape[0]
    t0 = time.perf_counter()
    ex, sm = split_exp_sm(u)
    counts = np.bincount(ex, minlength=256)
    lens = huffman_lengths(counts)
    nblocks = (n + BLOCK - 1) // BLOCK
    cap = int(n * (int(lens.max()) + 8) // 8 + nblocks * 8 + 64)
    out = np.empty(cap, np.uint8)
    offsets = np.empty(nblocks + 1, np.int64)
    nbytes = lib.huff_encode(ex.ctypes.data, n, lens.ctypes.data, BLOCK, out.ctypes.data, cap, offsets.ctypes.data)
    if nbytes < 0:
        raise RuntimeError("huff_encode failed")
    tc = time.perf_counter() - t0
    t0 = time.perf_counter()
    rec_ex = np.empty(n, np.uint8)
    rc = lib.huff_decode(out.ctypes.data, offsets.ctypes.data, n, lens.ctypes.data, BLOCK, rec_ex.ctypes.data)
    if rc != 0:
        raise RuntimeError("huff_decode failed")
    rec = join_exp_sm(rec_ex, sm)
    td = time.perf_counter() - t0
    meta = 256 + 8 * (nblocks + 1) + 16
    return {"compressed_bytes": int(nbytes) + n + meta, "metadata_bytes": meta, "huffman_bytes": int(nbytes),
            "raw_sign_mantissa_bytes": n, "max_code_len": int(lens.max()), "compress_s": tc, "decompress_s": td,
            "exact": bool(np.array_equal(rec, u))}


def _rans_streams(sym: np.ndarray, probs: np.ndarray):
    import constriction
    model = constriction.stream.model.Categorical(probs, perfect=False)
    streams = []
    n = sym.shape[0]
    for s in range(0, n, BLOCK):
        coder = constriction.stream.stack.AnsCoder()
        coder.encode_reverse(sym[s:s + BLOCK].astype(np.int32), model)
        streams.append(coder.get_compressed())
    return model, streams


def _rans_decode(model, streams, n):
    import constriction
    out = np.empty(n, np.int32)
    pos = 0
    for st in streams:
        m = min(BLOCK, n - pos)
        out[pos:pos + m] = constriction.stream.stack.AnsCoder(st).decode(model, m)
        pos += m
    return out


def _probs(counts: np.ndarray) -> np.ndarray:
    p = counts.astype(np.float64)
    p[p == 0] = 0.0
    return p / p.sum()


def method_rans_exp(u: np.ndarray) -> dict:
    n = u.shape[0]
    t0 = time.perf_counter()
    ex, sm = split_exp_sm(u)
    counts = np.bincount(ex, minlength=256)
    model, streams = _rans_streams(ex, _probs(counts))
    tc = time.perf_counter() - t0
    t0 = time.perf_counter()
    rec_ex = _rans_decode(model, streams, n).astype(np.uint8)
    rec = join_exp_sm(rec_ex, sm)
    td = time.perf_counter() - t0
    body = sum(int(s.nbytes) for s in streams)
    meta = 256 * 4 + 8 * (len(streams) + 1) + 16
    return {"compressed_bytes": body + n + meta, "metadata_bytes": meta, "rans_bytes": body, "raw_sign_mantissa_bytes": n,
            "compress_s": tc, "decompress_s": td, "exact": bool(np.array_equal(rec, u))}


def method_rans_16(u: np.ndarray) -> dict:
    n = u.shape[0]
    t0 = time.perf_counter()
    counts = np.bincount(u.astype(np.int64), minlength=65536)
    model, streams = _rans_streams(u, _probs(counts))
    tc = time.perf_counter() - t0
    t0 = time.perf_counter()
    rec = _rans_decode(model, streams, n).astype(np.uint16)
    td = time.perf_counter() - t0
    body = sum(int(s.nbytes) for s in streams)
    meta = 65536 * 4 + 8 * (len(streams) + 1) + 16
    return {"compressed_bytes": body + meta, "metadata_bytes": meta, "rans_bytes": body,
            "compress_s": tc, "decompress_s": td, "exact": bool(np.array_equal(rec, u))}


def main() -> None:
    from llmopt.lab.hash import require_pinned_file
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    out = f"logs/bf16_lossless/{'smoke/' if SMOKE else ''}bench_{head}.json"
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    zv = subprocess.run([ZSTD, "--version"], capture_output=True, text=True).stdout.strip()
    lib = build_huff_lib()
    import constriction
    receipt = {"commit": head, "tree_dirty": dirty, "smoke": SMOKE, "shard_dir_basename": os.path.basename(SHARD_DIR),
               "zstd_version": zv, "levels": LEVELS, "block_elements": BLOCK,
               "constriction_version": getattr(constriction, "__version__", "unknown"),
               "huffman_codec": "scratch/bf16_huff.c (canonical, MSB-first, byte-aligned blocks, compiled at run time)",
               "shards": {}, "tensors": [],
               "note": "compressed_bytes include every dictionary, block offset table and header needed to decode the "
                       "tensor alone; exact means the recovered bytes equal the original BF16 bytes; throughput is "
                       "single-process wall on the raw byte count; peak RSS is the process's running maximum."}
    for shard, pin in SHARDS.items():
        path = os.path.join(SHARD_DIR, shard)
        chk = require_pinned_file(path, pin["sha256"], pin["bytes"])
        hdr, base = safetensors_header(path)
        names = [k for k in hdr if k != "__metadata__" and not k.startswith("mtp.")]
        receipt["shards"][shard] = {"pin_check": chk, "tensors_used": len(names)}
        if SMOKE:
            names = names[:2] if "00007" in shard else ["lm_head.weight"]
        for name in names:
            mm = tensor_view(path, hdr, base, name)
            n = min(mm.shape[0], 32 * 1024 * 1024) if SMOKE else mm.shape[0]
            u = np.ascontiguousarray(np.asarray(mm[:n]))
            del mm
            row = {"shard": shard, "name": name, "shape": hdr[name]["shape"], "class": classify(name), "n": int(n),
                   "raw_bytes": int(2 * n), "methods": {}}
            print(f"[bench] {name} {hdr[name]['shape']} n {n:,}", flush=True)

            def rec(label, r):
                r = dict(r)
                r["bits_per_weight"] = 8.0 * r["compressed_bytes"] / n
                r["ratio_compressed_over_raw"] = r["compressed_bytes"] / (2 * n)
                r["compress_MBps_raw"] = 2 * n / max(r["compress_s"], 1e-9) / 1e6
                r["decompress_MBps_raw"] = 2 * n / max(r["decompress_s"], 1e-9) / 1e6
                r["peak_rss_bytes_so_far"] = rss_bytes()
                row["methods"][label] = r
                print(f"[bench]   {label:22s} {r['bits_per_weight']:6.3f} bits/w  ratio {r['ratio_compressed_over_raw']:.4f}  "
                      f"c {r['compress_MBps_raw']:7.1f} MB/s  d {r['decompress_MBps_raw']:7.1f} MB/s  exact {r['exact']}", flush=True)
                if not r["exact"]:
                    raise SystemExit(f"REFUSING: {label} did not recover {name} exactly")

            rec("raw", {"compressed_bytes": 2 * n, "metadata_bytes": 0, "compress_s": 0.0, "decompress_s": 0.0, "exact": True})
            for L in LEVELS:
                rec(f"zstd-{L}", method_zstd(u, L))
                rec(f"byteplane+zstd-{L}", method_byteplane(u, L))
                rec(f"bitshuffle+zstd-{L}", method_bitshuffle(u, L))
            rec("dfloat11", method_dfloat11(u, lib))
            rec("rans-exp", method_rans_exp(u))
            rec("rans-16", method_rans_16(u))
            receipt["tensors"].append(row)
            del u
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"[bench] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
