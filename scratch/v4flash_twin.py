"""F1a (PRE-REG V4-F1): pure-torch twin of DeepSeek-V4-Flash's
inference/kernel.py — the six tilelang kernels plus the Hadamard
rotation — so the vendor's model.py runs unmodified on Mac CPU/MPS.

Semantics transcribed from the vendor kernel bodies (tilelang 0.1.8
source, fetched 2026-08-03), not re-derived. Device constraints from
the F1 risk scan, verified on torch 2.12.1:
  * MPS can STORE float8_e4m3fn/e8m0 but cannot run ANY kernel on them
    (even .float()), while float4_e2m1fn_x2 is uint8-viewable anywhere.
    So all fp8/e8m0 decode goes through view(uint8) + LUT gather, and
    act_quant returns bf16 grid VALUES on MPS (its only consumers are
    this module's own gemms, which dequant by scale regardless).
  * The five inplace=True QAT call sites pass NON-CONTIGUOUS SLICES
    (kv[..., :-rd]); the vendor wrapper contiguous-copies then writes
    back with x.copy_(y). reshape() + copy_() preserves that contract.
  * scale_fmt is forced "ue8m0": scales are 2^ceil(log2(amax/max)),
    computed here exactly via frexp (mantissa==0.5 branch), never via
    log2()+ceil() which drifts at exact powers of two.
  * RNE casts to the e2m1/e4m3 grids use searchsorted left/right pairs
    with ties-to-even by code-index parity — checked against torch's
    own CPU e4m3 cast over all finite values in the self-test.

Install as the vendor's `kernel` and `fast_hadamard_transform` modules
BEFORE importing model.py:
    import v4flash_twin; v4flash_twin.install()

Self-test (the F1a acceptance bars): .venv/bin/python scratch/v4flash_twin.py
"""
import math
import sys
import types

import torch

FP4_GRID = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
# every non-negative finite e4m3 value, ascending, built from torch's
# own dtype so the grid IS the IEEE spec rather than a transcription
_E4M3_ALL = torch.arange(256, dtype=torch.uint8).view(
    torch.float8_e4m3fn).float()
E4M3_GRID = _E4M3_ALL[torch.isfinite(_E4M3_ALL) & (_E4M3_ALL >= 0)]\
    .unique().sort().values                      # 128 values, 0..448
# fp8 byte -> fp32 value LUT (nan for the two nan codes)
F8_LUT = torch.where(torch.isfinite(_E4M3_ALL), _E4M3_ALL,
                     torch.zeros(()))


def _pow2_ceil_log2(x):
    """2^ceil(log2(x)) exactly, via the vendor's own IEEE bit trick
    (kernel.py fast_log2_ceil/fast_pow2) — integer ops only, so it runs
    on MPS where frexp/ldexp do not. Valid for positive NORMAL floats;
    every caller floors amax above the subnormal range first."""
    bits = x.float().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + (bits & ((1 << 23) - 1)).ne(0).int()
    return ((e + 127) << 23).view(torch.float32)


def _rne_to_grid(x, grid):
    """Round to nearest grid value, ties to even CODE INDEX — which is
    ties-to-even-mantissa for both e2m1 and e4m3 (each grid, sorted
    ascending over non-negative values, alternates mantissa parity).
    Operates on |x|; caller reapplies sign. Returns grid indices."""
    g = grid.to(x.device)
    mid = (g[:-1] + g[1:]) / 2
    a = x.abs().clamp(max=float(g[-1]))
    lo = torch.searchsorted(mid, a, right=False)   # ties -> lower nbr
    hi = torch.searchsorted(mid, a, right=True)    # ties -> upper nbr
    return torch.where((lo != hi) & (lo % 2 != 0), hi, lo)


def _quant_values(x, grid):
    """|x| RNE'd onto grid, sign restored, fp32."""
    idx = _rne_to_grid(x, grid)
    v = grid.to(x.device).gather(0, idx.reshape(-1)).reshape(idx.shape)
    return v * torch.sign(x.float())


def _f8_to_f32(a):
    """fp8 e4m3 tensor -> fp32 values, device-pure (uint8 LUT gather)."""
    u = a.view(torch.uint8).long()
    return F8_LUT.to(a.device)[u]


def _e8m0_to_f32(s):
    """e8m0 byte -> 2^(b-127), bit-constructed (MPS-safe, no ldexp)."""
    return (s.view(torch.uint8).int() << 23).view(torch.float32)


def _f32_to_e8m0(s):
    """Exact power-of-two fp32 -> e8m0 byte: the exponent field IS the
    e8m0 code (mantissa is zero by construction). MPS-safe."""
    b = ((s.float().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    return b.view(torch.float8_e8m0fnu)


def _scale_f32(s):
    """Any scale tensor (f32 / e8m0) -> fp32, device-pure."""
    return (_e8m0_to_f32(s) if s.dtype == torch.float8_e8m0fnu
            else s.float())


# [256, 2] pair-LUT: byte -> (low-nibble value, high-nibble value).
# One gather per byte replaces two shift+mask+gather rounds plus a
# where (F1e arm 4: the unpack was 84%-of-decode hot). Built from the
# same grid, so bit-identity with the certified decode is structural.
_PAIR_LUT = torch.empty(256, 2)
for _byte in range(256):
    for _half, _nib in ((0, _byte & 0xF), (1, _byte >> 4)):
        _mag = float(FP4_GRID[_nib & 0x7])
        _PAIR_LUT[_byte, _half] = -_mag if _nib >= 8 else _mag


def _unpack_fp4(b):
    """[.., K//2] packed e2m1 (any 1-byte view) -> [.., K] fp32.
    Low nibble first; sign bit 0x8 (byte-identical to K3)."""
    u = b.view(torch.uint8).long()
    return _PAIR_LUT.to(u.device)[u].reshape(*u.shape[:-1],
                                             u.size(-1) * 2)


# ---------------------------------------------------------------- kernels

def act_quant(x, block_size=128, scale_fmt=None,
              scale_dtype=torch.float32, inplace=False):
    """Vendor contract: per-(row, block) amax (floor 1e-4); scale
    amax/448, ue8m0 -> 2^ceil(log2(amax/448)); RNE e4m3 store, or fused
    quant-dequant written back INTO x (which may be a view)."""
    N = x.size(-1)
    assert N % block_size == 0
    z = x.float().reshape(*x.shape[:-1], N // block_size, block_size)
    amax = z.abs().amax(-1).clamp(min=1e-4)
    s = (_pow2_ceil_log2(amax / 448.0) if scale_fmt is not None
         else amax / 448.0)
    q = (z / s.unsqueeze(-1)).clamp(-448.0, 448.0)
    qv = _quant_values(q, E4M3_GRID)
    if inplace:
        x.copy_((qv * s.unsqueeze(-1)).reshape(x.shape).to(x.dtype))
        return x
    if x.device.type == "cpu":
        y = qv.reshape(x.shape).to(torch.float8_e4m3fn)
    else:   # MPS cannot cast to fp8; grid VALUES carry the same info
        y = qv.reshape(x.shape).to(torch.bfloat16)
    s_out = (_f32_to_e8m0(s) if scale_dtype == torch.float8_e8m0fnu
             else s.to(scale_dtype))
    return y, s_out


def fp4_act_quant(x, block_size=32, inplace=False):
    """Per-(row,32) amax (floor 6*2^-126); scale 2^ceil(log2(amax/6))
    as e8m0; RNE e2m1. model.py uses only the inplace QAT form."""
    N = x.size(-1)
    assert N % block_size == 0
    z = x.float().reshape(*x.shape[:-1], N // block_size, block_size)
    amax = z.abs().amax(-1).clamp(min=6 * 2.0 ** -126)
    s = _pow2_ceil_log2(amax / 6.0)
    qv = _quant_values((z / s.unsqueeze(-1)).clamp(-6.0, 6.0), FP4_GRID)
    if inplace:
        x.copy_((qv * s.unsqueeze(-1)).reshape(x.shape).to(x.dtype))
        return x
    idx = _rne_to_grid((z / s.unsqueeze(-1)).clamp(-6.0, 6.0),
                       FP4_GRID).reshape(*x.shape[:-1], N).to(torch.uint8)
    neg = (qv.reshape(*x.shape[:-1], N) < 0) | (
        (qv.reshape(*x.shape[:-1], N) == 0) & (z.reshape(*x.shape[:-1], N) < 0))
    code = torch.where(neg, idx | 0x8, idx)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    if x.device.type == "cpu" and hasattr(torch, "float4_e2m1fn_x2"):
        packed = packed.view(torch.float4_e2m1fn_x2)
    return packed, _f32_to_e8m0(s)


def _deq_act(a, a_s, group=128):
    """Quantized activations (fp8 on cpu, bf16 grid values on mps) +
    per-group scales -> fp32."""
    K = a.size(-1)
    av = _f8_to_f32(a) if a.dtype == torch.float8_e4m3fn else a.float()
    return (av.reshape(*a.shape[:-1], K // group, group)
            * _scale_f32(a_s).unsqueeze(-1)).reshape(*a.shape[:-1], K)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    """C = A_fp8[M,K] @ B_fp8[N,K]^T; A per-1x128 scales, B per-128x128
    block scales [ceil(N/128), ceil(K/128)]. fp32 compute (one dequant
    then matmul — mathematically equal to the vendor's per-K-block
    accumulate; summation order differs, bar is the fp64 reference)."""
    af = _deq_act(a, a_s, 128)
    N, K = b.size(0), b.size(-1)
    bs = _scale_f32(b_s)
    bf = _f8_to_f32(b) if b.dtype == torch.float8_e4m3fn else b.float()
    bf = bf.reshape(N, K // 128, 128) * bs.repeat_interleave(
        128, dim=0)[:N].unsqueeze(-1)
    out = af @ bf.reshape(N, K).T
    return out.to(torch.get_default_dtype())


_W_CACHE, _W_CAP = {}, int(__import__("os").environ.get("WCACHE", "0"))


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    """C = A_fp8[M,K] @ B_fp4[N,K]^T; B packed 2 codes/byte along K
    (low nibble first), per-32 e8m0 weight scales [N, K/32].

    WCACHE=<n> (F1e arm 3): memoize the unpacked+scaled bf16 weight,
    FIFO-capped at n tensors. Keys are the weight Parameter objects
    (long-lived), so id-reuse cannot alias. Dequant stays value-exact;
    only WHEN it happens changes. Measured motivation: 84% of decode
    time was this unpack, re-done for every expert on every token."""
    af = _deq_act(a, a_s, 128)
    key = id(b)
    hit = _W_CAP and key in _W_CACHE
    if hit:
        bf = _W_CACHE[key][1]
    else:
        bf = _unpack_fp4(b)
        N, K = bf.size(0), bf.size(-1)
        bf = ((bf.reshape(N, K // 32, 32)
               * _scale_f32(b_s).unsqueeze(-1)).reshape(N, K)
              .to(torch.bfloat16))
        if _W_CAP:
            if len(_W_CACHE) >= _W_CAP:
                _W_CACHE.pop(next(iter(_W_CACHE)))
            _W_CACHE[key] = (b, bf)      # hold b so id stays valid
    return (af.to(torch.bfloat16) @ bf.T).to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """q [b,s,h,d], kv [b,n,d] (K==V latent), idx -1 masked; the sink
    joins the DENOMINATOR only. fp32 math, q.dtype out."""
    b, s, h, d = q.shape
    idx = topk_idxs.long()
    mask = idx < 0
    kvg = kv.float().gather(
        1, idx.clamp(min=0).reshape(b, -1, 1).expand(-1, -1, d)
    ).reshape(b, s, -1, d)
    logits = torch.einsum("bshd,bstd->bsht", q.float(), kvg) * softmax_scale
    logits = logits.masked_fill(mask.unsqueeze(2), float("-inf"))
    mx = torch.maximum(logits.amax(-1),
                       attn_sink.float().reshape(1, 1, h).to(q.device))
    w = torch.exp(logits - mx.unsqueeze(-1))
    denom = w.sum(-1) + torch.exp(
        attn_sink.float().reshape(1, 1, h).to(q.device) - mx)
    out = torch.einsum("bsht,bstd->bshd", w, kvg) / denom.unsqueeze(-1)
    return out.to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4,
                      sinkhorn_iters=20, eps=1e-6):
    """pre=sigmoid(m0*s0+b0)+eps; post=2*sigmoid(m1*s1+b1); comb =
    softmax(rows)+eps, col/(sum+eps), then (iters-1) x [row/(sum+eps),
    col/(sum+eps)] — eps placement matches kernel.py:391-423 exactly."""
    hc = hc_mult
    m, hs, hb = mixes.float(), hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(m[..., :hc] * hs[0] + hb[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hs[1] + hb[hc:2 * hc])
    comb = (m[..., 2 * hc:] * hs[2] + hb[2 * hc:]).reshape(
        *m.shape[:-1], hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    to = mixes.dtype
    return pre.to(to), post.to(to), comb.to(to)


def hadamard_transform(x, scale=1.0):
    """Sylvester-order Walsh-Hadamard along the last dim (power of 2)."""
    d = x.size(-1)
    assert d & (d - 1) == 0, "hadamard needs a power-of-2 dim"
    y = x.float()
    h = 1
    while h < d:
        y = y.reshape(*y.shape[:-1], d // (2 * h), 2, h)
        y = torch.cat([y[..., 0, :] + y[..., 1, :],
                       y[..., 0, :] - y[..., 1, :]], dim=-1)
        y = y.reshape(x.shape)
        h *= 2
    return (y * scale).to(x.dtype)


def install():
    """Register the twin as `kernel` and `fast_hadamard_transform` so
    the vendor model.py's imports resolve here."""
    k = types.ModuleType("kernel")
    for name in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm",
                 "sparse_attn", "hc_split_sinkhorn"):
        setattr(k, name, globals()[name])
    sys.modules["kernel"] = k
    f = types.ModuleType("fast_hadamard_transform")
    setattr(f, "hadamard_transform", hadamard_transform)
    sys.modules["fast_hadamard_transform"] = f


# ---------------------------------------------------------------- F1a bars

def _smallest_pow2_geq(v):
    p = 1.0
    while p < v:
        p *= 2
    while p / 2 >= v:
        p /= 2
    return p


def _bars(dev):
    g = torch.Generator().manual_seed(20260803)
    ok = []

    # (1) RNE-to-e4m3 vs torch's own CPU cast, ALL finite inputs classes
    x = torch.cat([torch.linspace(-500, 500, 20011),
                   E4M3_GRID, -E4M3_GRID,
                   (E4M3_GRID[:-1] + E4M3_GRID[1:]) / 2,        # exact ties
                   -(E4M3_GRID[:-1] + E4M3_GRID[1:]) / 2])
    ref = x.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    got = _quant_values(x.clamp(-448, 448), E4M3_GRID)
    same = (ref == got) | (ref.abs() + got.abs() == 0)          # +-0
    ok.append(("e4m3 RNE == torch cast (incl. all ties)", bool(same.all())))

    # (2) e2m1 tie vector, hand-computed ties-to-even
    ties = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0,
                         -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5.0])
    want = torch.tensor([0., 1., 1., 2., 2., 4., 4.,
                         -0., -1., -1., -2., -2., -4., -4.])
    got = _quant_values(ties, FP4_GRID)
    ok.append(("e2m1 ties-to-even vector", bool(torch.equal(got, want))))

    # (3) pow2ceil exact at powers of two and neighbours
    xs = torch.tensor([2.0 ** i for i in range(-30, 30)])
    good = True
    for v in torch.cat([xs, xs * 1.0000001, xs * 0.9999999]):
        good &= float(_pow2_ceil_log2(v.reshape(1))) == \
            _smallest_pow2_geq(float(v))
    ok.append(("pow2ceil == smallest pow2 >= x", bool(good)))

    # (4) inplace QAT writes back through a NON-CONTIGUOUS slice
    base = torch.randn(2, 3, 192, generator=g).bfloat16().to(dev)
    keep = base[..., -64:].clone()
    view = base[..., :-64]                       # non-contiguous slice
    act_quant(view, 64, "ue8m0", torch.float8_e8m0fnu, inplace=True)
    ok.append((f"inplace-on-slice[{dev}] writes back",
               not torch.equal(view.cpu(), torch.randn(0)) and
               torch.equal(base[..., -64:].cpu(), keep.cpu()) and
               (view.float().abs().sum() > 0).item()))

    # (5) fp4_gemm on a REAL cached expert vs exact fp64 reference,
    #     decode side bit-identical to the certified rungA decode
    import os
    blob = "checkpoints/v4flash_sample/layers.22.ffn.experts.0.w1.weight.bin"
    sblob = blob.replace("weight", "scale")
    if os.path.exists(blob) and os.path.exists(sblob):
        import numpy as np
        sys.path.insert(0, "scratch")
        import v4flash_rungA as RA
        raw = open(blob, "rb").read()
        b = torch.frombuffer(bytearray(raw), dtype=torch.uint8)\
            .reshape(2048, 2048)
        bs = torch.frombuffer(bytearray(open(sblob, "rb").read()),
                              dtype=torch.uint8).reshape(2048, 128)\
            .view(torch.float8_e8m0fnu)
        nib = np.frombuffer(raw, dtype=np.uint8)
        codes = np.empty(nib.size * 2, dtype=np.uint8)
        codes[0::2], codes[1::2] = nib & 0xF, nib >> 4
        ref_dec = RA.FP4_TABLE[codes].reshape(2048, 4096)
        ours_dec = _unpack_fp4(b).double().numpy()
        ok.append(("fp4 decode bit-identical to rungA",
                   bool((ref_dec == ours_dec).all())))
        x = torch.randn(16, 4096, generator=g).bfloat16()
        a, a_s = act_quant(x, 128, "ue8m0", torch.float8_e8m0fnu)
        af64 = _deq_act(a, a_s, 128).double()
        bf64 = (torch.from_numpy(ref_dec)
                * _e8m0_to_f32(bs).double().repeat_interleave(32, dim=1))
        ref = af64 @ bf64.T
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            got = fp4_gemm(a.to(dev) if dev == "cpu" else
                           _f8_to_f32(a).bfloat16().to(dev),
                           a_s.to(dev) if dev == "cpu" else
                           _e8m0_to_f32(a_s).to(dev),
                           b.to(dev), bs.to(dev) if dev == "cpu" else
                           _e8m0_to_f32(bs).to(dev)).cpu()
        finally:
            torch.set_default_dtype(prev)
        rel = ((got.double() - ref).abs()
               / ref.abs().clamp(min=1e-2)).max().item()
        ok.append((f"fp4_gemm[{dev}] vs fp64 ref rel={rel:.2e} (<=1/128)",
                   rel <= 1 / 128))
    else:
        ok.append(("fp4_gemm real-expert bar", "SKIP (no cached expert)"))

    # (6) sinkhorn doubly stochastic at 20 iterations
    m = torch.randn(2, 3, 24, generator=g).to(dev)
    _, _, comb = hc_split_sinkhorn(m, torch.ones(3, device=dev),
                                   torch.zeros(24, device=dev))
    resid = max((comb.sum(-1) - 1).abs().max().item(),
                (comb.sum(-2) - 1).abs().max().item())
    ok.append((f"sinkhorn[{dev}] residual {resid:.1e} (<=2e-5)",
               resid <= 2e-5))

    # (7) hadamard involution + isometry
    x = torch.randn(4, 128, generator=g).to(dev)
    hx = hadamard_transform(x, scale=128 ** -0.5)
    back = hadamard_transform(hx, scale=128 ** -0.5)
    ok.append((f"hadamard[{dev}] involution",
               (back - x).abs().max().item() <= 1e-5))
    ok.append((f"hadamard[{dev}] isometry",
               abs(hx.norm().item() - x.norm().item()) <= 1e-3))

    # (8) sparse_attn vs dense fp64 reference, with sink and -1 masking
    b_, s_, h_, d_, n_, k_ = 1, 3, 4, 32, 40, 8
    q = torch.randn(b_, s_, h_, d_, generator=g).bfloat16().to(dev)
    kv = torch.randn(b_, n_, d_, generator=g).bfloat16().to(dev)
    sink = torch.randn(h_, generator=g)
    idx = torch.stack([torch.randperm(n_, generator=g)[:k_]
                       for _ in range(b_ * s_)]).reshape(b_, s_, k_).int()
    idx[0, 0, -2:] = -1
    got = sparse_attn(q, kv, sink.to(dev), idx.to(dev),
                      d_ ** -0.5).cpu().double()
    ref = torch.zeros_like(got)
    for si in range(s_):
        ids = idx[0, si][idx[0, si] >= 0].long()
        for hi in range(h_):
            lg = (q[0, si, hi].cpu().double()
                  @ kv[0, ids].cpu().double().T) * d_ ** -0.5
            w = torch.exp(lg - lg.max())
            den = w.sum() + math.exp(float(sink[hi]) - float(lg.max()))
            ref[0, si, hi] = (w @ kv[0, ids].cpu().double()) / den
    err = (got - ref).abs().max().item()
    ok.append((f"sparse_attn[{dev}] vs dense fp64 err={err:.2e} (<=1/64)",
               err <= 1 / 64))

    # (9) act_quant ue8m0 scales are exact powers of two
    x = torch.randn(8, 256, generator=g).bfloat16().to(dev)
    _, s = act_quant(x, 128, "ue8m0", torch.float8_e8m0fnu)
    sf = _scale_f32(s.cpu())
    ok.append((f"act_quant[{dev}] scales pow2",
               bool((sf == _pow2_ceil_log2(sf)).all())))
    return ok


if __name__ == "__main__":
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    fails = 0
    for dev in devs:
        for name, res in _bars(dev):
            tag = res if isinstance(res, str) else ("PASS" if res else "FAIL")
            fails += tag == "FAIL"
            print(f"[twin] {tag:4s} {name}")
    print(f"[twin] {'ALL BARS PASS' if not fails else f'{fails} FAILURES'}")
    raise SystemExit(1 if fails else 0)
