"""Per-position and paired statistics for a head-substitution study.

A study compares two candidate heads applied to the same hidden states
against a teacher's stored logits, position by position. Everything
here is pure numpy over one position or over aligned per-position
arrays, so the formulas can be pinned by tests on tiny vocabularies
and reused by any driver. Conventions:

- logits are compared over the slice the caller passes (the live
  vocabulary); the caller slices before calling.
- teacher_rank is the number of candidate logits strictly greater than
  the candidate's logit at the teacher token (0 = the teacher token is
  the candidate argmax).
- kl_teacher_cand is the forward KL(teacher || candidate) in nats, in
  float64 log-softmax.
- slope_on_teacher is the census formula (scratch/qwen_e0_disagreement_census.py):
  the centered least-squares slope of candidate logits on teacher
  logits over the teacher's top ids of one position; NaN when the
  teacher logits are constant.
- paired_counts / block_deltas / block_bootstrap_delta_points read two
  boolean per-position agreement arrays (candidate A, candidate B) and
  the block each position belongs to (a prompt, a paragraph, a
  prefix); positions inside a block are not treated as independent.

Also here: bfloat16 bit conversions (round to nearest even), the
safetensors header reader for a bf16 tensor, and the row slice of an
S16 payload (the lab codec's layout: block exponents, 16 fp16 levels,
nibble codes) so a CPU oracle can decode a row range without decoding
the whole matrix.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np

S16_BLOCK = 128
S16_LEVELS = 16


def log_softmax64(z) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    m = z.max()
    return z - (m + math.log(np.exp(z - m).sum()))


def position_stats(z_cand, z_teacher, teacher_token: int, top_ids, k_top: int = 8) -> dict:
    zc = np.asarray(z_cand, dtype=np.float32)
    zt = np.asarray(z_teacher, dtype=np.float32)
    if zc.shape != zt.shape or zc.ndim != 1:
        raise ValueError("candidate and teacher logits must be aligned 1-d arrays")
    order = np.argsort(-zc.astype(np.float64), kind="stable")
    lc = log_softmax64(zc)
    lt = log_softmax64(zt)
    pt = np.exp(lt)
    z_tok = float(zc[teacher_token])
    z64 = zc.astype(np.float64)
    m = z64.max()
    lse = float(m + math.log(np.exp(z64 - m).sum()))
    return {
        "argmax": int(order[0]),
        "mean_logit": float(z64.mean()),
        "margin": float(zc[order[0]] - zc[order[1]]) if zc.size > 1 else float("nan"),
        "teacher_rank": int((zc > zc[teacher_token]).sum()),
        "lse": lse,
        "z_teacher_token": z_tok,
        "ce_teacher_token": float(-lc[teacher_token]),
        "kl_teacher_cand": float((pt * (lt - lc)).sum()),
        "z_top_ids": zc[np.asarray(top_ids)].astype(np.float32),
        "top_ids": order[:k_top].astype(np.int32),
        "top_logits": zc[order[:k_top]].astype(np.float32),
    }


def slope_on_teacher(z_cand_top, z_teacher_top) -> float:
    a = np.asarray(z_cand_top, dtype=np.float64)
    b = np.asarray(z_teacher_top, dtype=np.float64)
    bc = b - b.mean()
    den = float(bc @ bc)
    if den <= 0:
        return float("nan")
    return float(((a - a.mean()) @ bc) / den)


def paired_counts(agree_a, agree_b) -> dict:
    a = np.asarray(agree_a, dtype=bool)
    b = np.asarray(agree_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired arrays differ in length")
    n = int(a.size)
    corrected = int((~a & b).sum())
    broken = int((a & ~b).sum())
    return {"n": n, "a_right": int(a.sum()), "b_right": int(b.sum()), "corrected": corrected, "broken": broken,
            "both_right": int((a & b).sum()), "both_wrong": int((~a & ~b).sum()), "net": corrected - broken,
            "delta_points": 100.0 * (corrected - broken) / n if n else float("nan")}


def _binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided sign-test p-value: P(X <= min(k, n-k)) + P(X >= max(k, n-k)) under X ~ Bin(n, 1/2)."""
    if n == 0:
        return float("nan")
    lo = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, lo + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def block_deltas(agree_a, agree_b, blocks) -> dict:
    a = np.asarray(agree_a, dtype=bool)
    b = np.asarray(agree_b, dtype=bool)
    blocks = np.asarray(blocks)
    ids = sorted(set(blocks.tolist()), key=lambda x: (str(type(x)), x))
    per = []
    for bid in ids:
        m = blocks == bid
        per.append(int((~a[m] & b[m]).sum() - (a[m] & ~b[m]).sum()))
    up = sum(1 for d in per if d > 0)
    down = sum(1 for d in per if d < 0)
    tied = len(per) - up - down
    return {"block_ids": [bid if isinstance(bid, (int, str)) else bid.item() for bid in ids], "per_block_net": per,
            "blocks_up": up, "blocks_down": down, "blocks_tied": tied,
            "sign_test_p_two_sided": _binom_two_sided(up, up + down)}


def block_bootstrap_delta_points(agree_a, agree_b, blocks, n_boot: int = 10000, seed: int = 0) -> dict:
    """Percentile bootstrap of the paired agreement delta (points of the
    pooled position count) resampling BLOCKS with replacement."""
    a = np.asarray(agree_a, dtype=bool)
    b = np.asarray(agree_b, dtype=bool)
    blocks = np.asarray(blocks)
    ids = sorted(set(blocks.tolist()), key=lambda x: (str(type(x)), x))
    nets = np.array([((~a[blocks == i] & b[blocks == i]).sum() - (a[blocks == i] & ~b[blocks == i]).sum()) for i in ids], np.float64)
    sizes = np.array([(blocks == i).sum() for i in ids], np.float64)
    rng = np.random.default_rng(seed)
    nb = len(ids)
    draws = rng.integers(0, nb, size=(n_boot, nb))
    boot = 100.0 * nets[draws].sum(1) / sizes[draws].sum(1)
    return {"delta_points": float(100.0 * nets.sum() / sizes.sum()), "n_blocks": nb, "n_boot": int(n_boot), "seed": int(seed),
            "ci95_low": float(np.percentile(boot, 2.5)), "ci95_high": float(np.percentile(boot, 97.5))}


# ---- bfloat16 ----

def bf16_bits_to_f32(u16) -> np.ndarray:
    u = np.ascontiguousarray(np.asarray(u16, dtype=np.uint16))
    return (u.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x, chunk: int = 1 << 24) -> np.ndarray:
    """Round to nearest even in the bf16 grid; NaN payloads keep the quiet
    bit. Processed in chunks so the uint64 temporaries stay bounded (a
    5 GB head would otherwise need 30 GB of temporaries)."""
    f = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    flat = f.reshape(-1)
    out = np.empty(flat.shape, np.uint16)
    for s in range(0, flat.size, chunk):
        part = flat[s:s + chunk]
        u = part.view(np.uint32).astype(np.uint64)
        lsb = (u >> 16) & 1
        o = ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)
        nan = np.isnan(part)
        o[nan] = ((u[nan] >> 16) | 0x40).astype(np.uint16)
        out[s:s + chunk] = o
    return out.reshape(f.shape)


def round_to_bf16(x) -> np.ndarray:
    return bf16_bits_to_f32(f32_to_bf16_bits(x))


def safetensors_header(path):
    """(header dict, byte offset of the data section)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n).decode("utf-8"))
    return header, 8 + n


def read_bf16_bits(path, name: str, expect_shape=None) -> np.ndarray:
    """The named BF16 tensor of a safetensors file as its raw uint16 bits,
    read by its header offsets only."""
    header, start = safetensors_header(path)
    if name not in header:
        raise ValueError(f"{name} not in {Path(path).name}")
    ent = header[name]
    if ent["dtype"] != "BF16":
        raise ValueError(f"{name} dtype {ent['dtype']} is not BF16")
    shape = tuple(int(s) for s in ent["shape"])
    if expect_shape is not None and shape != tuple(expect_shape):
        raise ValueError(f"{name} shape {shape} != expected {tuple(expect_shape)}")
    o0, o1 = (int(x) for x in ent["data_offsets"])
    n = int(np.prod(shape))
    if o1 - o0 != 2 * n:
        raise ValueError(f"{name} byte span {o1 - o0} != 2 * {n}")
    with open(path, "rb") as f:
        f.seek(start + o0)
        raw = f.read(o1 - o0)
    if len(raw) != o1 - o0:
        raise ValueError(f"{name}: short read {len(raw)} of {o1 - o0} bytes")
    return np.frombuffer(raw, np.uint16).reshape(shape)


def read_bf16_tensor(path, name: str, expect_shape=None) -> np.ndarray:
    """The named BF16 tensor as float32 (an exact widening)."""
    return bf16_bits_to_f32(read_bf16_bits(path, name, expect_shape))


def s16_row_slice_payload(buf: bytes, shape, r0: int, r1: int) -> bytes:
    """The S16 payload of rows [r0, r1) of a [R, C] matrix, in the same
    layout as the whole (block exponents, the 16 fp16 levels, nibble
    codes), so qcodec.dec_s16(sub, (r1 - r0, C)) equals the row slice of
    the whole decode."""
    R, C = (int(s) for s in shape)
    if not (0 <= r0 < r1 <= R):
        raise ValueError(f"row range [{r0},{r1}) outside [0,{R})")
    if C % S16_BLOCK:
        raise ValueError("C must be a multiple of the block")
    nb = R * C // S16_BLOCK
    bpr = C // S16_BLOCK
    mv = memoryview(buf)                 # slices without copying the whole codes section
    if len(mv) - nb - 2 * S16_LEVELS != R * C // 2:
        raise ValueError("payload length does not match the shape")
    c0 = nb + 2 * S16_LEVELS
    return bytes(mv[r0 * bpr:r1 * bpr]) + bytes(mv[nb:c0]) + bytes(mv[c0 + r0 * C // 2:c0 + r1 * C // 2])
