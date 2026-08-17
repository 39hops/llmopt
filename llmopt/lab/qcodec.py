"""Canonical WHOLE-0T payload decoders (the one shared decode path).

Frozen layouts, produced by scratch/qwen_whole0t.py (results-cited,
never edited) and consumed by the runtime reference, the cache
sidecar, and the MODEL-1 scorer THROUGH THIS MODULE ONLY. The
compiler's encoders stay where they are; this module is the single
decode implementation those independent encoders are tested
against on golden fixtures (tests/test_qwen_codec.py) before any
model-scale execution.

Layouts (n elements, block 128, nb = n // 128):
  w4 : exps u8[nb] ++ codebook fp16[256,4] ++ idxs u8[n/4]
       value = codebook[idx] * 2^(exp - 127), groups of 4
       consecutive along the flattened row order.
  s16: exps u8[nb] ++ levels fp16[16] ++ codes u8[n/2]
       HIGH nibble = EVEN element (opposite of the GPTQ
       convention), value = levels[code] * 2^(exp - 127).
  raw: vendor bytes; BF16 -> fp32 via u16 << 16.

Expected payload lengths (exact, asserted by the qualifier):
  w4 : nb + 2048 + n // 4
  s16: nb + 32 + n // 2
  raw: n * 2 (BF16)
"""
from __future__ import annotations

import numpy as np

BLOCK = 128


def expected_len(codec: str, shape) -> int:
    n = 1
    for d in shape:
        n *= d
    if codec == "w4":
        return n // BLOCK + 2048 + n // 4
    if codec == "s16":
        return n // BLOCK + 32 + n // 2
    if codec == "raw":
        return n * 2
    if codec == "excluded":
        return 0
    raise ValueError(f"unknown codec {codec!r}")


def dec_raw(buf: bytes, shape, dtype: str = "BF16") -> np.ndarray:
    u16 = np.frombuffer(buf, np.uint16)
    if dtype == "BF16":
        return ((u16.astype(np.uint32) << 16).view(np.float32)
                .reshape(shape))
    return u16.view(np.float16).astype(np.float32).reshape(shape)


def _scales(buf: bytes, nb: int) -> np.ndarray:
    exps = np.frombuffer(buf, np.uint8, nb, 0).astype(np.int32)
    return np.exp2(exps - 127).astype(np.float32)


def dec_w4(buf: bytes, shape) -> np.ndarray:
    n = 1
    for d in shape:
        n *= d
    nb = n // BLOCK
    assert len(buf) == expected_len("w4", shape), \
        f"w4 payload {len(buf)} != {expected_len('w4', shape)}"
    scale = _scales(buf, nb)
    cb = np.frombuffer(buf, np.float16, 256 * 4, nb).reshape(256, 4)
    idx = np.frombuffer(buf, np.uint8, n // 4, nb + 2048)
    Wn = cb.astype(np.float32)[idx].reshape(nb, BLOCK)
    return (Wn * scale[:, None]).reshape(shape)


def dec_s16(buf: bytes, shape) -> np.ndarray:
    n = 1
    for d in shape:
        n *= d
    nb = n // BLOCK
    assert len(buf) == expected_len("s16", shape), \
        f"s16 payload {len(buf)} != {expected_len('s16', shape)}"
    scale = _scales(buf, nb)
    lv = np.frombuffer(buf, np.float16, 16, nb).astype(np.float32)
    codes = np.frombuffer(buf, np.uint8, n // 2, nb + 32)
    c = np.empty(n, np.uint8)
    c[0::2] = codes >> 4          # HIGH nibble = even element
    c[1::2] = codes & 0xF
    Wn = lv[c].reshape(nb, BLOCK)
    return (Wn * scale[:, None]).reshape(shape)


def decode_entry(buf: bytes, entry: dict) -> np.ndarray:
    """Decode one manifest entry {codec, shape, dtype?}."""
    codec = entry["codec"]
    if codec == "excluded":
        raise ValueError("excluded tensor has no payload")
    if codec == "raw":
        return dec_raw(buf, entry["shape"], entry.get("dtype", "BF16"))
    if codec == "w4":
        return dec_w4(buf, entry["shape"])
    if codec == "s16":
        return dec_s16(buf, entry["shape"])
    raise ValueError(f"unknown codec {codec!r}")
