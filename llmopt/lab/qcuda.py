"""CUDA decode primitives for the compressed-artifact tower.

ORIGIN: adopted verbatim from scratch/qwen_cuda_rung4.py (the
2026-08-17 CUDA ladder, rung 4 "fused tower"). The scratch driver is
the receipt-cited original and stays frozen in place; while the two
files coexist, tests/test_lab_qcuda_adoption.py asserts each adopted
symbol is byte-identical between them, so a fix lands in BOTH in the
same commit or in neither.

Adopted: SUB127, w4_decode_kernel, s16_gemv_kernel, w4_gemv_kernel,
W4Gpu, S16Gpu, FusedW4Linear.

PARITY REGIME (as gated in-process by the drivers, never by this
module):
  - decode kernel: BIT-EXACT against llmopt.lab.qcodec (dec_w4) on the
    rung-3 fixture set (random payloads plus the exponent edges
    e=0/127/254); np.array_equal, equal_nan=True.
  - GEMV kernels (w4 and s16): rel error <= 1e-5 against the canonical
    llmopt.lab.qcodec decode matmul'd in float64. Bit exactness is NOT
    expected for GEMV — the fused accumulation order differs from a
    decode-then-matmul.
  - the gates run in-process in the driver (scratch/qwen_cuda_rung4.py
    _gates()) before any artifact is touched or receipt written.

SCALE BIT-CONSTRUCTION: block scales are powers of two stored as a
single uint8 exponent per 128-element block, so the kernels rebuild
the fp32 scale without a lookup: (e << 23) bitcast to float32 gives
2^(e-127) for 1 <= e <= 254. Two specials — e == 0 is the subnormal
boundary and uses the exact constant 2^-127 (SUB127), and e == 255
bitcasts to inf, which is the intended carry-through of an inf scale.

OPTIONAL DEPS: triton is imported through a guard, so this module
imports cleanly on machines without CUDA/triton (the Mac lab venv has
no triton). When triton is absent the kernels are defined against a
stand-in whose `jit` and `constexpr` are pass-throughs; touching any
other triton/tl attribute, or launching a kernel, raises a clear
RuntimeError at USE time rather than at import time.
"""
from __future__ import annotations

import numpy as np
import torch

from llmopt.lab.qcodec import BLOCK

HAVE_TRITON: bool

_NEED = ("llmopt.lab.qcuda needs triton (and a CUDA device): "
         "these are GPU decode kernels. Install triton on the CUDA "
         "machine; this module imports on triton-less hosts only so "
         "the source-identity guard and the ledger tooling can read "
         "it.")


class _NoTriton:
    """Import-time stand-in; every real use raises RuntimeError."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attr: str):
        raise RuntimeError(f"{_NEED} (touched {self._name}.{attr})")

    @staticmethod
    def constexpr(value):
        return value

    @staticmethod
    def jit(fn):
        return _NoTritonKernel(fn)


class _NoTritonKernel:
    """A kernel that exists only to be looked at, never launched."""

    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

    def __getitem__(self, grid):
        raise RuntimeError(f"{_NEED} (launch of {self.__name__})")

    def __call__(self, *a, **k):
        raise RuntimeError(f"{_NEED} (call of {self.__name__})")


try:  # pragma: no cover - environment dependent
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:  # pragma: no cover - environment dependent
    triton = _NoTriton("triton")  # type: ignore[assignment]
    tl = _NoTriton("tl")  # type: ignore[assignment]
    HAVE_TRITON = False


def require_triton() -> None:
    """Raise the clear error before doing any GPU work."""
    if not HAVE_TRITON:
        raise RuntimeError(_NEED)


# --- adopted verbatim from scratch/qwen_cuda_rung4.py (do not edit
# --- one copy without the other; see the module docstring) ---

SUB127 = tl.constexpr(5.877471754111438e-39)  # exact fp32 2^-127


@triton.jit
def w4_decode_kernel(idx_ptr, cb_ptr, exp_ptr, out_ptr, n,
                     BLK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLK + tl.arange(0, BLK)
    m = offs < n
    e = tl.load(exp_ptr + offs // 128, mask=m, other=127).to(tl.int32)
    s = tl.where(e == 0, SUB127,
                 (e << 23).to(tl.float32, bitcast=True))
    byte = tl.load(idx_ptr + offs // 4, mask=m, other=0)
    val = tl.load(cb_ptr + byte.to(tl.int32) * 4 + offs % 4,
                  mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, val * s, mask=m)


@triton.jit
def s16_gemv_kernel(code_ptr, lv_ptr, exp_ptr, x_ptr, y_ptr, C,
                    BLK_C: tl.constexpr):
    """s16 fused GEMV: HIGH nibble = EVEN element (qcodec
    convention — opposite of GPTQ); scale by bit-construction."""
    r = tl.program_id(0)
    acc = tl.zeros((BLK_C,), tl.float32)
    for c0 in range(0, C, BLK_C):
        offs = c0 + tl.arange(0, BLK_C)
        m = offs < C
        flat = r * C + offs
        e = tl.load(exp_ptr + flat // 128, mask=m,
                    other=127).to(tl.int32)
        s = tl.where(e == 0, SUB127,
                     (e << 23).to(tl.float32, bitcast=True))
        byte = tl.load(code_ptr + flat // 2, mask=m,
                       other=0).to(tl.int32)
        nib = tl.where(offs % 2 == 0, byte >> 4, byte & 0xF)
        val = tl.load(lv_ptr + nib, mask=m,
                      other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=m, other=0.0)
        acc += val * s * x
    tl.store(y_ptr + r, tl.sum(acc, 0))


@triton.jit
def w4_gemv_kernel(idx_ptr, cb_ptr, exp_ptr, x_ptr, y_ptr, C,
                   BLK_C: tl.constexpr):
    """One program per output row, fused decode+dot, fp32 acc.
    Scales rebuilt in-kernel (bit-construction, rung-3 validated)."""
    r = tl.program_id(0)
    acc = tl.zeros((BLK_C,), tl.float32)
    for c0 in range(0, C, BLK_C):
        offs = c0 + tl.arange(0, BLK_C)
        m = offs < C
        flat = r * C + offs
        e = tl.load(exp_ptr + flat // 128, mask=m,
                    other=127).to(tl.int32)
        s = tl.where(e == 0, SUB127,
                     (e << 23).to(tl.float32, bitcast=True))
        byte = tl.load(idx_ptr + flat // 4, mask=m, other=0)
        val = tl.load(cb_ptr + byte.to(tl.int32) * 4 + offs % 4,
                      mask=m, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=m, other=0.0)
        acc += val * s * x
    tl.store(y_ptr + r, tl.sum(acc, 0))


class W4Gpu:
    def __init__(self, buf: bytes, shape):
        n = int(np.prod(shape))
        nb = n // BLOCK
        self.shape, self.n = list(shape), n
        self.exps = torch.from_numpy(
            np.frombuffer(buf, np.uint8, nb, 0).copy()).cuda()
        self.cb = torch.from_numpy(
            np.frombuffer(buf, np.float16, 1024, nb).copy()).cuda()
        self.idx = torch.from_numpy(
            np.frombuffer(buf, np.uint8, n // 4,
                          nb + 2048).copy()).cuda()

    def decode_rows(self, lo, hi):
        R, C = self.shape
        n = (hi - lo) * C
        out = torch.empty(n, dtype=torch.float32, device="cuda")
        w4_decode_kernel[(triton.cdiv(n, 1024),)](
            self.idx[lo * C // 4:], self.cb,
            self.exps[lo * C // BLOCK:], out, n, BLK=1024)
        return out.reshape(hi - lo, C)

    def gemv(self, x):
        R, C = self.shape
        y = torch.empty(R, dtype=torch.float32, device="cuda")
        w4_gemv_kernel[(R,)](self.idx, self.cb, self.exps,
                             x.contiguous(), y, C, BLK_C=512)
        return y


class S16Gpu:
    """One s16 payload resident on device. GEMV-only (io use); the
    prefill path decodes rows via S16Rows on CPU."""

    def __init__(self, buf: bytes, shape):
        n = int(np.prod(shape))
        nb = n // BLOCK
        self.shape = list(shape)
        self.exps = torch.from_numpy(
            np.frombuffer(buf, np.uint8, nb, 0).copy()).cuda()
        self.lv = torch.from_numpy(
            np.frombuffer(buf, np.float16, 16, nb).copy()).cuda()
        self.codes = torch.from_numpy(
            np.frombuffer(buf, np.uint8, n // 2,
                          nb + 32).copy()).cuda()

    def gemv(self, x):
        R, C = self.shape
        y = torch.empty(R, dtype=torch.float32, device="cuda")
        s16_gemv_kernel[(R,)](self.codes, self.lv, self.exps,
                              x.contiguous(), y, C, BLK_C=512)
        return y


class FusedW4Linear(torch.nn.Module):
    CHUNK = 8192

    def __init__(self, pay: W4Gpu):
        super().__init__()
        self.pay = pay
        self.out_features, self.in_features = pay.shape

    def forward(self, x):
        lead = x.shape[:-1]
        C = x.shape[-1]
        flat = x.reshape(-1, C)
        if flat.shape[0] == 1:
            y = self.pay.gemv(flat[0].float())
            return y.reshape(*lead, -1).to(x.dtype)
        outs = []
        R = self.pay.shape[0]
        for lo in range(0, R, self.CHUNK):
            hi = min(lo + self.CHUNK, R)
            W = self.pay.decode_rows(lo, hi)
            outs.append(flat.float() @ W.T)
        return torch.cat(outs, -1).reshape(*lead, -1).to(x.dtype)
