"""Track B1 of the GO of 2026-10-22 (LOSSLESS-BF16-FEASIBILITY-0): the
information-theoretic baseline of the ORIGINAL Qwen3.8-27B BF16 weights,
measured on the exact 16-bit patterns of the two retained, sha-verified
vendor shards (model-00007-of-00018: layers 21 (part), 22 and 23;
model-00018-of-00018: lm_head and the MTP tensors). No float32
conversion anywhere: every statistic is a function of the uint16 bit
pattern. No download, no write to the shards.

Per tensor (and pooled per architectural class):
  n, raw bytes (2 n), distinct 16-bit patterns, the 16-bit symbol
  histogram's zeroth-order entropy H16 (bits per weight), the exponent
  byte's entropy H_exp (the 8 bits at positions 7..14), the sign bit's
  entropy, the 7-bit mantissa's marginal entropy, the conditional
  entropy of sign and mantissa given the exponent (H16 - H_exp; a
  DFloat11-style coder that stores sign and mantissa raw pays 8 bits
  for them, so H_exp + 8 is that coder's floor), the high and low
  byte entropies (what a byte-plane coder sees), the first-order
  conditional entropy of the exponent given the previous exponent in
  memory order, and block-adaptive entropies at tile sizes TILES
  (contiguous runs in memory order): the per-block zeroth-order entropy
  of the exponent byte plus the cost of each block's own dictionary
  (distinct symbols x (8 + ceil(log2 tile)) bits), and at the two
  largest tiles the per-block entropy of the full 16-bit pattern plus
  its dictionary (distinct x (16 + ceil(log2 tile)) bits). Random-access
  metadata is accounted as a 64-bit offset per block.
Marginal exponent entropy is reported as the exponent's entropy, never
as the weight's; the weight-level figures are H16 and the block 16-bit
figures. The sampled tensors are the two shards' tensors and nothing
else; the receipt says which layers they come from.
Receipt logs/bf16_lossless/entropy_<commit>.json (refuse-if-exists).
Env: SHARD_DIR (~/qwen_vendor_shards/1d4bf0f2), CHUNK (67108864
elements), SMOKE=1 (first 2 tensors of shard 7, the first 32 M elements
of lm_head, receipt under logs/bf16_lossless/smoke/).
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

SMOKE = os.environ.get("SMOKE") == "1"
SHARD_DIR = os.path.expanduser(os.environ.get("SHARD_DIR", "~/qwen_vendor_shards/1d4bf0f2"))
SHARDS = {"model-00007-of-00018.safetensors": {"sha256": "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892", "bytes": 2108759344},
          "model-00018-of-00018.safetensors": {"sha256": "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06", "bytes": 3392197344}}
TILES = [256, 1024, 4096, 16384, 65536]
TILES16 = [4096, 65536]
CHUNK = int(os.environ.get("CHUNK", str(1 << 26)))


def classify(name: str) -> str:
    if name.startswith("mtp."):
        return "mtp (not in the text model budget)"
    if name == "lm_head.weight":
        return "lm_head"
    if re.search(r"self_attn\.(q|k|v|o)_proj\.weight", name):
        return "attention_proj"
    if re.search(r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)\.weight", name):
        return "deltanet_proj"
    if re.search(r"linear_attn\.(in_proj_a|in_proj_b)\.weight", name):
        return "deltanet_gate_proj"
    if re.search(r"mlp\.(gate|up|down)_proj\.weight", name):
        return "mlp"
    return "small (norms, conv, A_log, dt_bias)"


def safetensors_header(path: str):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def tensor_view(path: str, hdr: dict, base: int, name: str) -> np.ndarray:
    e = hdr[name]
    if e["dtype"] != "BF16":
        raise SystemExit(f"REFUSING: {name} is {e['dtype']}")
    s, t = e["data_offsets"]
    mm = np.memmap(path, dtype=np.uint16, mode="r", offset=base + s, shape=((t - s) // 2,))
    return mm


def H_bits(counts: np.ndarray) -> float:
    c = counts[counts > 0].astype(np.float64)
    n = c.sum()
    return float((c * (np.log2(n) - np.log2(c))).sum() / n) if n else 0.0


def block_entropy_bits(sym: np.ndarray, tile: int, nsym: int, width_bits: int) -> tuple[float, float]:
    """(sum of per-block zeroth-order entropies in bits, sum of dictionary
    bits) over the whole-tile blocks of sym (int64 symbols < nsym)."""
    nb = sym.shape[0] // tile
    if nb == 0:
        return 0.0, 0.0
    s = sym[: nb * tile]
    if nsym <= 256:
        b = np.repeat(np.arange(nb, dtype=np.int64), tile)
        counts = np.bincount(b * nsym + s, minlength=nb * nsym).reshape(nb, nsym)
        nz = counts > 0
        c = counts[nz].astype(np.float64)
        bits = float((c * (math.log2(tile) - np.log2(c))).sum())
        distinct = nz.sum(axis=1)
    else:
        srt = np.sort(s.reshape(nb, tile), axis=1)
        starts = np.ones((nb, tile), dtype=bool)
        starts[:, 1:] = srt[:, 1:] != srt[:, :-1]
        idx = np.flatnonzero(starts.ravel())
        runs = np.diff(np.append(idx, nb * tile)).astype(np.float64)
        bits = float((runs * (math.log2(tile) - np.log2(runs))).sum())
        distinct = starts.sum(axis=1)
    dict_bits = float(distinct.sum()) * (width_bits + math.ceil(math.log2(tile)))
    return bits, dict_bits


def census_tensor(mm: np.ndarray, n_limit: int | None = None) -> dict:
    n = mm.shape[0] if n_limit is None else min(mm.shape[0], n_limit)
    hist16 = np.zeros(65536, dtype=np.int64)
    pair = np.zeros(256 * 256, dtype=np.int64)
    blk = {t: [0.0, 0.0] for t in TILES}
    blk16 = {t: [0.0, 0.0] for t in TILES16}
    prev_exp = None
    step = (CHUNK // 65536) * 65536
    t0 = time.perf_counter()
    for s in range(0, n, step):
        u = np.asarray(mm[s: min(n, s + step)]).astype(np.int64)
        hist16 += np.bincount(u, minlength=65536)
        ex = (u >> 7) & 0xFF
        if prev_exp is not None:
            pair[prev_exp * 256 + ex[0]] += 1
        pair += np.bincount(ex[:-1] * 256 + ex[1:], minlength=65536)
        prev_exp = int(ex[-1])
        for t in TILES:
            b, d = block_entropy_bits(ex, t, 256, 8)
            blk[t][0] += b
            blk[t][1] += d
        for t in TILES16:
            b, d = block_entropy_bits(u, t, 65536, 16)
            blk16[t][0] += b
            blk16[t][1] += d
    wall = time.perf_counter() - t0
    h3 = hist16.reshape(2, 256, 128)          # sign, exponent, mantissa
    h_exp = h3.sum(axis=(0, 2))
    h_sign = h3.sum(axis=(1, 2))
    h_mant = h3.sum(axis=(0, 1))
    h_hi = hist16.reshape(256, 256).sum(axis=1)
    h_lo = hist16.reshape(256, 256).sum(axis=0)
    pr = pair.reshape(256, 256)
    row_tot = pr.sum(axis=1)
    cond = 0.0
    for i in np.flatnonzero(row_tot):
        cond += H_bits(pr[i]) * row_tot[i]
    cond /= max(pair.sum(), 1)
    nz_exp = np.flatnonzero(h_exp)
    out = {"n": int(n), "raw_bytes": int(2 * n), "distinct_patterns": int((hist16 > 0).sum()),
           "H16_bits": H_bits(hist16), "H_exp_bits": H_bits(h_exp), "H_sign_bits": H_bits(h_sign),
           "H_mant_marginal_bits": H_bits(h_mant), "H_sign_mant_given_exp_bits": H_bits(hist16) - H_bits(h_exp),
           "H_hi_byte_bits": H_bits(h_hi), "H_lo_byte_bits": H_bits(h_lo),
           "H_exp_given_prev_exp_bits": float(cond),
           "dfloat11_floor_bits": H_bits(h_exp) + 8.0,
           "exp_range": [int(nz_exp.min()), int(nz_exp.max())] if nz_exp.size else None,
           "exp_distinct": int(nz_exp.size),
           "zero_pattern_fraction": float(hist16[0] / n), "neg_zero_fraction": float(hist16[0x8000] / n),
           "sign_neg_fraction": float(h_sign[1] / n),
           "top8_patterns": [{"hex": f"{int(p):04x}", "frac": float(hist16[p] / n)} for p in np.argsort(-hist16)[:8]],
           "exp_hist": {int(i): int(h_exp[i]) for i in nz_exp},
           "mant_hist": [int(v) for v in h_mant],
           "block_exp": {str(t): {"blocks": int(n // t), "entropy_bits_per_weight": blk[t][0] / max((n // t) * t, 1),
                                  "dict_bits_per_weight": blk[t][1] / max((n // t) * t, 1),
                                  "offset_bits_per_weight": 64.0 / t} for t in TILES},
           "block_16": {str(t): {"blocks": int(n // t), "entropy_bits_per_weight": blk16[t][0] / max((n // t) * t, 1),
                                 "dict_bits_per_weight": blk16[t][1] / max((n // t) * t, 1),
                                 "offset_bits_per_weight": 64.0 / t} for t in TILES16},
           "census_wall_s": wall}
    return out, hist16


def main() -> None:
    from llmopt.lab.hash import require_pinned_file
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    out = f"logs/bf16_lossless/{'smoke/' if SMOKE else ''}entropy_{head}.json"
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    receipt = {"commit": head, "tree_dirty": dirty, "smoke": SMOKE, "shard_dir_basename": os.path.basename(SHARD_DIR),
               "tiles_exp": TILES, "tiles_16": TILES16, "chunk_elements": CHUNK, "shards": {}, "tensors": [], "classes": {},
               "note": "All statistics are functions of the exact BF16 uint16 bit patterns (no float conversion). "
                       "H16 is the weight-level zeroth-order entropy; H_exp is the exponent byte's entropy only. "
                       "Block figures add each block's own dictionary and a 64-bit offset per block. The tensors are "
                       "those of the two retained shards and are not a random sample of the 64 layers."}
    class_hist: dict = {}
    class_rows: dict = {}
    for shard, pin in SHARDS.items():
        path = os.path.join(SHARD_DIR, shard)
        chk = require_pinned_file(path, pin["sha256"], pin["bytes"])
        hdr, base = safetensors_header(path)
        names = [k for k in hdr if k != "__metadata__"]
        receipt["shards"][shard] = {"pin_check": chk, "header_bytes": base, "tensors": len(names)}
        if SMOKE:
            names = names[:2] if "00007" in shard else ["lm_head.weight"]
        for name in names:
            mm = tensor_view(path, hdr, base, name)
            cls = classify(name)
            print(f"[census] {shard[:11]} {name} {hdr[name]['shape']} ...", flush=True)
            row, hist = census_tensor(mm, n_limit=(32 * 1024 * 1024 if SMOKE else None))
            row = {"shard": shard, "name": name, "shape": hdr[name]["shape"], "class": cls, **row}
            receipt["tensors"].append(row)
            class_hist[cls] = class_hist.get(cls, np.zeros(65536, dtype=np.int64)) + hist
            class_rows.setdefault(cls, []).append(row)
            print(f"[census]   n {row['n']:,} distinct {row['distinct_patterns']:,} H16 {row['H16_bits']:.3f} "
                  f"H_exp {row['H_exp_bits']:.3f} H(sm|exp) {row['H_sign_mant_given_exp_bits']:.3f} "
                  f"block16@4096 {row['block_16']['4096']['entropy_bits_per_weight']:.3f}+{row['block_16']['4096']['dict_bits_per_weight']:.3f} "
                  f"({row['census_wall_s']:.0f}s)", flush=True)
            del mm
    for cls, rows in class_rows.items():
        n = sum(r["n"] for r in rows)
        w = lambda key: sum(r[key] * r["n"] for r in rows) / n  # noqa: E731
        hist = class_hist[cls]
        h3 = hist.reshape(2, 256, 128)
        receipt["classes"][cls] = {
            "tensors": len(rows), "n": int(n), "raw_bytes": int(2 * n),
            "H16_bits_weighted_mean": w("H16_bits"), "H16_bits_pooled_static_model": H_bits(hist),
            "H_exp_bits_weighted_mean": w("H_exp_bits"), "H_exp_bits_pooled": H_bits(h3.sum(axis=(0, 2))),
            "H_sign_mant_given_exp_bits_weighted_mean": w("H_sign_mant_given_exp_bits"),
            "dfloat11_floor_bits_weighted_mean": w("dfloat11_floor_bits"),
            "H_hi_byte_bits_weighted_mean": w("H_hi_byte_bits"), "H_lo_byte_bits_weighted_mean": w("H_lo_byte_bits"),
            "H_exp_given_prev_exp_bits_weighted_mean": w("H_exp_given_prev_exp_bits"),
            "block_exp_total_bits_per_weight": {t: sum((r["block_exp"][t]["entropy_bits_per_weight"] + r["block_exp"][t]["dict_bits_per_weight"]
                                                        + r["block_exp"][t]["offset_bits_per_weight"]) * r["n"] for r in rows) / n for t in map(str, TILES)},
            "block_16_total_bits_per_weight": {t: sum((r["block_16"][t]["entropy_bits_per_weight"] + r["block_16"][t]["dict_bits_per_weight"]
                                                       + r["block_16"][t]["offset_bits_per_weight"]) * r["n"] for r in rows) / n for t in map(str, TILES16)},
            "distinct_patterns_pooled": int((hist > 0).sum())}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"[census] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
