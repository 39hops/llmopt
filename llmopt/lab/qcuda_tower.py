"""Reusable CUDA tower: codec dispatch + FusedS16Linear + residency plan.

Born from OBSERVATION QWEN-BLE-FREEGEN-1-ABORT (2026-08-19): the
frozen rung4 driver routed only w4 Linears to compressed execution,
so an arm with 48 promoted s16 decoder tensors silently expanded
0.87 GiB of payload into 6.875 GiB of dense FP32 and generation
collapsed to 0.56 tok/s. Spec:
docs/superpowers/specs/2026-08-19-qcuda-tower-runtime.md.

The executable invariant: a compressed 2D layer tensor must NEVER
fall through to ordinary dense nn.Linear. route_codec() is total
over the manifest codecs and REFUSES anything unrouted;
assert_no_fallthrough() sweeps the built model against the manifest
and raises on any surviving dense compressed tensor.

New symbols only — the adopted historical kernels/classes in
llmopt.lab.qcuda are consumed, never modified.
"""
from __future__ import annotations

import torch

from llmopt.lab.qcodec import BLOCK, expected_len
from llmopt.lab.qcuda import (HAVE_TRITON, S16Gpu, W4Gpu,  # noqa: F401
                              FusedW4Linear, require_triton, tl, triton)

SUB127_F32 = 5.877471754111438e-39  # exact fp32 2^-127 (qcuda.SUB127)


@triton.jit
def s16_decode_kernel(code_ptr, lv_ptr, exp_ptr, out_ptr, n,
                      BLK: tl.constexpr):
    """Row-decode for s16 payload slices: HIGH nibble = EVEN element
    (qcodec convention), scale by bit-construction, fp32 out."""
    pid = tl.program_id(0)
    offs = pid * BLK + tl.arange(0, BLK)
    m = offs < n
    e = tl.load(exp_ptr + offs // 128, mask=m, other=127).to(tl.int32)
    s = tl.where(e == 0, SUB127_F32,
                 (e << 23).to(tl.float32, bitcast=True))
    byte = tl.load(code_ptr + offs // 2, mask=m, other=0).to(tl.int32)
    nib = tl.where(offs % 2 == 0, byte >> 4, byte & 0xF)
    val = tl.load(lv_ptr + nib, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, val * s, mask=m)


def s16_decode_rows(pay: S16Gpu, lo: int, hi: int):
    """Decode rows [lo, hi) of an S16Gpu payload to fp32 on device.
    Slice offsets follow W4Gpu.decode_rows: rows are C-contiguous, C
    is a multiple of BLOCK (qcodec shape contract), so both the code
    stream (2 elems/byte) and the exp stream slice cleanly."""
    require_triton()
    R, C = pay.shape
    n = (hi - lo) * C
    out = torch.empty(n, dtype=torch.float32, device="cuda")
    s16_decode_kernel[(triton.cdiv(n, 1024),)](
        pay.codes[lo * C // 2:], pay.lv,
        pay.exps[lo * C // BLOCK:], out, n, BLK=1024)
    return out.reshape(hi - lo, C)


class FusedS16Linear(torch.nn.Module):
    """s16 sibling of FusedW4Linear: decode-step GEMV through the
    adopted parity-gated S16Gpu.gemv; prefill through transient
    row-chunk decode + matmul."""
    CHUNK = 8192

    def __init__(self, pay: S16Gpu):
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
            W = s16_decode_rows(self.pay, lo, hi)
            outs.append(flat.float() @ W.T)
        return torch.cat(outs, -1).reshape(*lead, -1).to(x.dtype)


# ------------------------------------------------------------ routing
# Pure functions (no CUDA) so the invariant is testable everywhere.

ROUTES = {"w4": "fused_w4", "s16": "fused_s16", "raw": "dense_raw"}


def route_codec(codec: str) -> str:
    """Total, fail-closed codec routing. 'excluded' tensors are not
    materialized at all and must be handled (or refused) by the
    caller before routing."""
    r = ROUTES.get(codec)
    if r is None:
        raise ValueError(f"REFUSING: no route for codec {codec!r} — "
                         "a compressed tensor must never fall through "
                         "to dense nn.Linear")
    return r


def runtime_bytes(entry: dict) -> int:
    """Bytes the SELECTED representation occupies at runtime (the
    abort's lesson: artifact bytes != runtime bytes)."""
    n = 1
    for d in entry["shape"]:
        n *= d
    route = route_codec(entry["codec"])
    if route in ("fused_w4", "fused_s16"):
        return expected_len(entry["codec"], entry["shape"])
    return n * 4  # dense_raw: fp32 resident


def plan_residency(entries, free_bytes: int, reserve_frac: float = 0.15):
    """Sum planned runtime bytes over manifest entries and refuse if
    the plan exceeds free_bytes * (1 - reserve_frac). Returns the
    plan dict for the receipt (planned bytes per route + total)."""
    per_route: dict[str, int] = {}
    for e in entries:
        if e["codec"] == "excluded":
            continue
        r = route_codec(e["codec"])
        per_route[r] = per_route.get(r, 0) + runtime_bytes(e)
    total = sum(per_route.values())
    budget = int(free_bytes * (1.0 - reserve_frac))
    plan = {"per_route": per_route, "total_bytes": total,
            "free_bytes": int(free_bytes), "reserve_frac": reserve_frac,
            "budget_bytes": budget, "fits": total <= budget}
    if not plan["fits"]:
        raise MemoryError(f"REFUSING: planned residency {total} B "
                          f"exceeds budget {budget} B "
                          f"(free {free_bytes}, reserve {reserve_frac})")
    return plan


def assert_no_fallthrough(model, manifest: dict, name_fn=None) -> int:
    """The executable invariant: after module surgery, no module that
    the manifest marks w4/s16 may remain an ordinary nn.Linear.
    name_fn maps a module path to its manifest key (identity when
    None). Returns the number of compressed-routed modules checked."""
    checked = 0
    for path, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        key = (name_fn(path) if name_fn else path)
        e = manifest.get(key if isinstance(key, str) else "")
        if e is None:
            continue
        if e.get("codec") in ("w4", "s16"):
            raise RuntimeError(
                f"REFUSING: compressed tensor {key} ({e['codec']}) "
                f"fell through to dense nn.Linear at {path}")
    for path, mod in model.named_modules():
        if isinstance(mod, (FusedW4Linear, FusedS16Linear)):
            checked += 1
    return checked


def fused_module(entry: dict, buf: bytes):
    """Materialize the fused module for a manifest entry (CUDA)."""
    route = route_codec(entry["codec"])
    if route == "fused_w4":
        return FusedW4Linear(W4Gpu(buf, entry["shape"]))
    if route == "fused_s16":
        return FusedS16Linear(S16Gpu(buf, entry["shape"]))
    raise ValueError(f"REFUSING: fused_module on route {route!r}")
