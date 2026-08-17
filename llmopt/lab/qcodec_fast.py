"""Optimized WHOLE-0T decoders, parity-fixtured against qcodec.

Rule (2026-08-17 review): ONE canonical reference decoder
(llmopt.lab.qcodec); optimized decoders are permitted only with
mandatory parity fixtures against it — first/middle/last rows,
single-row and multi-row slices, several column widths, random
slices (tests/test_qwen_qualify.py). The CUDA W4 kernel gets the
same oracle. Codec semantics never hide inside a runtime.

W4Rows: row-sliced decode of a w4 payload without materializing
the tensor — blocks align to rows (C/128 blocks and C/4 index
bytes per row), so any [lo, hi) row range decodes independently.
"""
from __future__ import annotations

import numpy as np

BLOCK = 128


class W4Rows:
    def __init__(self, buf: bytes, shape):
        self.R, self.C = shape
        if self.C % BLOCK or self.C % 4:
            raise ValueError(f"invalid w4 row width {self.C}")
        nb = (self.R * self.C) // BLOCK
        self.exps = np.frombuffer(buf, np.uint8, nb, 0)
        self.cb = np.frombuffer(buf, np.float16, 1024, nb) \
            .reshape(256, 4).astype(np.float32)
        self.idx = np.frombuffer(buf, np.uint8,
                                 (self.R * self.C) // 4, nb + 2048)

    def rows(self, lo: int, hi: int) -> np.ndarray:
        if not (0 <= lo < hi <= self.R):
            raise ValueError(f"row range [{lo},{hi}) outside "
                             f"[0,{self.R})")
        bpr = self.C // BLOCK
        sc = np.exp2(self.exps[lo * bpr:hi * bpr]
                     .astype(np.int32) - 127).astype(np.float32)
        ix = self.idx[lo * self.C // 4:hi * self.C // 4]
        Wn = self.cb[ix].reshape(-1, BLOCK)
        return (Wn * sc[:, None]).reshape(hi - lo, self.C)
