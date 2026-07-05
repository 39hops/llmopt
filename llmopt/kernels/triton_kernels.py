"""Readable Triton kernels (CUDA) — mirrors kernels/metal.py.

Same ops, same reasoning: each kernel fuses a memory-bound op chain so the
data makes one trip through HBM instead of several. Written for readability;
the classic patterns visible in plain Triton:

- rmsnorm: one program per row; blocked sum-of-squares in registers
  (tl.sum does the tree reduction), then normalize+scale in the same pass.
- swiglu: silu(gate) * up fused elementwise.
- rope: one program per (token, head) row rotating interleaved pairs
  (x[2p], x[2p+1]).
- attention_decode: single-query attention, split-K flash pattern — each
  program streams one chunk of keys keeping a running (max, sumexp,
  weighted-V accumulator) and writes its partial; a second tiny kernel
  merges the partials with the same online rule. Never materializes the
  [T] score vector. Unlike the Metal version (one threadgroup,
  educational), split-K actually parallelizes across the GPU.
- paged_attention: batched decode attention straight over the paged KV
  pool from cache/paged.py — the block table *is* the memory layout, so
  fork/COW-shared blocks are attended in place with no gather. One
  program per (sequence, KV head, chunk): the KV head's whole GQA query
  group rides along as one tensor-core tile, so each K/V block is read
  from HBM exactly once for the group; chunks are split-K partials
  folded by a merge kernel. 2.7x over gather-then-SDPA (B=8, T=4096,
  kv_heads=8, group=4, dim=128) — the win is skipping the gather
  entirely, and it grows with fragmentation.
- flash_attention: the full tiled kernel (Flash-2 style) — one program per
  (head, query-block), inner loop over key blocks with online softmax over
  a [BLOCK_Q, BLOCK_K] score tile, optional causal mask. The optimizations
  that matter, in the order they were measured: dot in fp16 (tensor cores,
  fp32 accumulate) instead of upcasting first; masking only compiled into
  diagonal/ragged-end tiles so interior tiles run mask-free; softmax in the
  exp2 domain (one fewer multiply in the hot loop); autotuned tile sizes.
  Together: 1901 -> 749 us on the benchmark shape below (2.5x).

All kernels are verified elementwise against pure-torch references in
tests; benchmark with scripts/bench_triton_kernels.py. Measured (RTX 3080,
4096x4096 fp16): rmsnorm 3.7x over unfused ops and 1.5x over
torch.nn.RMSNorm; swiglu 2.7x over unfused; attention_decode 1.1x over
naive softmax@V at T=8192 (the split-K + fused merge is what it takes to
beat a cuBLAS GEMV pipeline); flash_attention 1.4x over torch's fused SDPA
(heads=32, T=2048, dim=128, causal); attention_decode_quant int8 ~1.6-2.3x
and packed int4 ~1.7-2.5x over fp16 KV at T=262k (int4 unpack is
instruction-bound, so it never reaches the roofline 4x; below T~64k a
~75 us launch/merge floor hides the bandwidth win — bench_kv_quant_decode.py).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------- rmsnorm


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, dim, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < dim

    x = tl.load(x_ptr + row * dim + cols, mask=mask, other=0.0).to(tl.float32)
    scale = 1.0 / tl.sqrt(tl.sum(x * x) / dim + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * dim + cols, (x * scale * w).to(out_ptr.dtype.element_ty), mask=mask)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x: [rows, dim], w: [dim]."""
    rows, dim = x.shape
    out = torch.empty_like(x)
    _rmsnorm_kernel[(rows,)](x, w, out, dim, eps, BLOCK=triton.next_power_of_2(dim))
    return out


# ----------------------------------------------------------------- swiglu


@triton.jit
def _swiglu_kernel(gate_ptr, up_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask).to(tl.float32)
    silu = g * tl.sigmoid(g)
    tl.store(out_ptr + offs, (silu * u).to(out_ptr.dtype.element_ty), mask=mask)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(gate)
    n = gate.numel()
    BLOCK = 1024
    _swiglu_kernel[(triton.cdiv(n, BLOCK),)](gate, up, out, n, BLOCK=BLOCK)
    return out


# ------------------------------------------------------------------- rope


@triton.jit
def _rope_kernel(x_ptr, pos_ptr, out_ptr, heads, base, dim: tl.constexpr, BLOCK: tl.constexpr):
    nh = tl.program_id(0)  # flattened (token, head)
    tok = nh // heads
    pair = tl.arange(0, BLOCK)
    mask = pair < dim // 2

    theta = tl.exp2(-tl.log2(base) * 2.0 * pair.to(tl.float32) / dim)
    angle = tl.load(pos_ptr + tok).to(tl.float32) * theta
    c, s = tl.cos(angle), tl.sin(angle)

    x0 = tl.load(x_ptr + nh * dim + 2 * pair, mask=mask).to(tl.float32)
    x1 = tl.load(x_ptr + nh * dim + 2 * pair + 1, mask=mask).to(tl.float32)
    ty = out_ptr.dtype.element_ty
    tl.store(out_ptr + nh * dim + 2 * pair, (x0 * c - x1 * s).to(ty), mask=mask)
    tl.store(out_ptr + nh * dim + 2 * pair + 1, (x0 * s + x1 * c).to(ty), mask=mask)


def rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """x: [n, heads, dim] (dim even), positions: [n] int."""
    n, heads, dim = x.shape
    out = torch.empty_like(x)
    _rope_kernel[(n * heads,)](
        x, positions, out, heads, base, dim=dim, BLOCK=triton.next_power_of_2(dim // 2)
    )
    return out


# ------------------------------------------------------- attention_decode


@triton.jit
def _attn_decode_kernel(
    q_ptr, k_ptr, v_ptr, m_ptr, l_ptr, acc_ptr,
    tlen, scale2, dim: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # one program per key chunk: online softmax (log2 domain) over its
    # BLOCK_T keys, writes partial (m, l, acc[dim]) for the merge kernel.
    chunk = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    dmask = d < dim
    q = tl.load(q_ptr + d, mask=dmask, other=0.0).to(tl.float32)

    t = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < tlen
    k = tl.load(k_ptr + t[:, None] * dim + d[None, :],
                mask=tmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
    score = tl.sum(k * q[None, :], axis=1) * scale2
    score = tl.where(tmask, score, float("-inf"))

    m = tl.max(score)
    p = tl.exp2(score - m)
    l = tl.sum(p)
    v = tl.load(v_ptr + t[:, None] * dim + d[None, :],
                mask=tmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
    acc = tl.sum(p[:, None] * v, axis=0)

    tl.store(m_ptr + chunk, m)
    tl.store(l_ptr + chunk, l)
    tl.store(acc_ptr + chunk * dim + d, acc, mask=dmask)


@triton.jit
def _attn_merge_kernel(
    m_ptr, l_ptr, acc_ptr, out_ptr,
    nchunks, dim: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # single program: merge the per-chunk partials with the same online rule.
    c = tl.arange(0, BLOCK_C)
    cmask = c < nchunks
    m = tl.load(m_ptr + c, mask=cmask, other=float("-inf"))
    l = tl.load(l_ptr + c, mask=cmask, other=0.0)
    corr = tl.exp2(m - tl.max(m))
    d = tl.arange(0, BLOCK_D)
    acc = tl.load(acc_ptr + c[:, None] * dim + d[None, :],
                  mask=cmask[:, None] & (d < dim)[None, :], other=0.0)
    out = tl.sum(acc * corr[:, None], axis=0) / tl.sum(corr * l)
    tl.store(out_ptr + d, out.to(out_ptr.dtype.element_ty), mask=d < dim)


def attention_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Single-query attention: q [dim], k/v [T, dim] -> [dim]."""
    t, dim = k.shape
    BLOCK_T = 512
    nchunks = triton.cdiv(t, BLOCK_T)
    dev = q.device
    m = torch.empty(nchunks, device=dev, dtype=torch.float32)
    l = torch.empty(nchunks, device=dev, dtype=torch.float32)
    acc = torch.empty(nchunks, dim, device=dev, dtype=torch.float32)
    out = torch.empty_like(q)
    BLOCK_D = triton.next_power_of_2(dim)
    _attn_decode_kernel[(nchunks,)](
        q, k, v, m, l, acc, t, _LOG2E / dim**0.5,
        dim=dim, BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
    )
    _attn_merge_kernel[(1,)](
        m, l, acc, out, nchunks,
        dim=dim, BLOCK_C=triton.next_power_of_2(nchunks), BLOCK_D=BLOCK_D,
    )
    return out


# ------------------------------------------------- attention_decode_quant


@triton.jit
def _attn_decode_int8_kernel(
    q_ptr, kc_ptr, ks_ptr, vc_ptr, vs_ptr, m_ptr, l_ptr, acc_ptr,
    tlen, scale2, dim: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # attention_decode with int8 K/V codes + per-token fp32 scales,
    # dequantized in registers: half the HBM traffic of fp16 K/V.
    chunk = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    dmask = d < dim
    q = tl.load(q_ptr + d, mask=dmask, other=0.0).to(tl.float32)

    t = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < tlen
    ks = tl.load(ks_ptr + t, mask=tmask, other=0.0)
    kc = tl.load(kc_ptr + t[:, None] * dim + d[None, :],
                 mask=tmask[:, None] & dmask[None, :], other=0)
    score = tl.sum(kc.to(tl.float32) * q[None, :], axis=1) * ks * scale2
    score = tl.where(tmask, score, float("-inf"))

    m = tl.max(score)
    p = tl.exp2(score - m)
    l = tl.sum(p)
    vs = tl.load(vs_ptr + t, mask=tmask, other=0.0)
    vc = tl.load(vc_ptr + t[:, None] * dim + d[None, :],
                 mask=tmask[:, None] & dmask[None, :], other=0)
    acc = tl.sum((p * vs)[:, None] * vc.to(tl.float32), axis=0)

    tl.store(m_ptr + chunk, m)
    tl.store(l_ptr + chunk, l)
    tl.store(acc_ptr + chunk * dim + d, acc, mask=dmask)


@triton.jit
def _attn_decode_int4_kernel(
    q_ptr, kc_ptr, ks_ptr, vc_ptr, vs_ptr, m_ptr, l_ptr, acc_ptr,
    tlen, scale2, dim: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # packed int4: two offset-coded nibbles per byte (code = value + 7, so
    # a nibble is unsigned in [0, 14]). Byte b holds elements (2b, 2b+1),
    # so k . q = lo . q_even + hi . q_odd — each byte is loaded exactly
    # once and unpacked in registers on half-width tiles.
    chunk = tl.program_id(0)
    half: tl.constexpr = dim // 2
    HBLOCK: tl.constexpr = BLOCK_D // 2
    dh = tl.arange(0, HBLOCK)
    hmask = dh < half
    q_even = tl.load(q_ptr + 2 * dh, mask=hmask, other=0.0).to(tl.float32)
    q_odd = tl.load(q_ptr + 2 * dh + 1, mask=hmask, other=0.0).to(tl.float32)

    t = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < tlen
    ldmask = tmask[:, None] & hmask[None, :]
    byte_off = t[:, None] * half + dh[None, :]

    kb = tl.load(kc_ptr + byte_off, mask=ldmask, other=0).to(tl.int32)
    k_lo = ((kb & 0xF) - 7).to(tl.float32)
    k_hi = ((kb >> 4) - 7).to(tl.float32)
    ks = tl.load(ks_ptr + t, mask=tmask, other=0.0)
    score = tl.sum(k_lo * q_even[None, :] + k_hi * q_odd[None, :], axis=1)
    score = tl.where(tmask, score * ks * scale2, float("-inf"))

    m = tl.max(score)
    p = tl.exp2(score - m)
    l = tl.sum(p)
    vb = tl.load(vc_ptr + byte_off, mask=ldmask, other=0).to(tl.int32)
    vs = tl.load(vs_ptr + t, mask=tmask, other=0.0)
    pw = (p * vs)[:, None]
    acc_even = tl.sum(pw * ((vb & 0xF) - 7).to(tl.float32), axis=0)
    acc_odd = tl.sum(pw * ((vb >> 4) - 7).to(tl.float32), axis=0)

    tl.store(m_ptr + chunk, m)
    tl.store(l_ptr + chunk, l)
    tl.store(acc_ptr + chunk * dim + 2 * dh, acc_even, mask=hmask)
    tl.store(acc_ptr + chunk * dim + 2 * dh + 1, acc_odd, mask=hmask)


def quantize_kv_rows(x: torch.Tensor, bits: int):
    """Per-token symmetric quantization of K or V [T, dim] for the decode
    kernels: int8 -> (codes int8 [T, dim], scale fp32 [T]); int4 -> codes
    packed two-per-byte as uint8 [T, dim // 2] (offset code value + 7)."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.float().abs().amax(dim=-1).clamp(min=1e-8) / qmax
    codes = (x.float() / scale[:, None]).round().clamp(-qmax, qmax).to(torch.int8)
    if bits == 8:
        return codes, scale
    u = (codes + 7).to(torch.uint8).view(x.shape[0], -1, 2)
    return u[:, :, 0] | (u[:, :, 1] << 4), scale


def attention_decode_quant(
    q, k_codes, k_scale, v_codes, v_scale, bits: int = 8, block_t: int | None = None
):
    """attention_decode over quantized K/V (see quantize_kv_rows).

    Decode attention is HBM-bound on the KV read, so int8 targets ~2x and
    packed int4 ~4x over fp16 K/V; scales add 8 bytes/token of overhead.
    Measured (RTX 3080, T=262k): the nibble unpack is instruction-bound
    enough that int4 lands ~2.5x, not 4x. Chunk sizes swept per bit width.
    """
    t = k_codes.shape[0]
    dim = q.shape[0]
    BLOCK_T = block_t if block_t is not None else (512 if bits == 8 else 256)
    nchunks = triton.cdiv(t, BLOCK_T)
    dev = q.device
    m = torch.empty(nchunks, device=dev, dtype=torch.float32)
    l = torch.empty(nchunks, device=dev, dtype=torch.float32)
    acc = torch.empty(nchunks, dim, device=dev, dtype=torch.float32)
    out = torch.empty_like(q)
    BLOCK_D = triton.next_power_of_2(dim)
    kernel = _attn_decode_int8_kernel if bits == 8 else _attn_decode_int4_kernel
    kernel[(nchunks,)](
        q, k_codes, k_scale, v_codes, v_scale, m, l, acc, t, _LOG2E / dim**0.5,
        dim=dim, BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
    )
    _attn_merge_kernel[(1,)](
        m, l, acc, out, nchunks,
        dim=dim, BLOCK_C=triton.next_power_of_2(nchunks), BLOCK_D=BLOCK_D,
    )
    return out


# -------------------------------------------------------- flash_attention


_LOG2E = 1.4426950408889634  # softmax via exp2: exp(x) == exp2(x * log2(e))


@triton.jit
def _flash_inner(q, k_base, v_base, m, l, acc, kstart, qi, offs,
                 tk, scale2, causal_diag: tl.constexpr, boundary: tl.constexpr,
                 dim: tl.constexpr, BLOCK_K: tl.constexpr):
    # one K/V tile: dot in the input dtype (fp16 -> tensor cores, fp32 acc);
    # masking only compiled in for diagonal (causal) and ragged-end tiles.
    ki = kstart + tl.arange(0, BLOCK_K)
    d = tl.arange(0, dim)
    if boundary:
        kv_mask = (ki < tk)[:, None]
        k = tl.load(k_base + ki[:, None] * dim + d[None, :], mask=kv_mask, other=0.0)
    else:
        k = tl.load(k_base + ki[:, None] * dim + d[None, :])

    score = tl.dot(q, tl.trans(k)) * scale2  # [BLOCK_Q, BLOCK_K], log2 domain
    if boundary:
        score = tl.where((ki < tk)[None, :], score, float("-inf"))
    if causal_diag:
        score = tl.where(offs + qi[:, None] >= ki[None, :], score, float("-inf"))

    m_new = tl.maximum(m, tl.max(score, axis=1))
    corr = tl.exp2(m - m_new)                # rescale old state
    p = tl.exp2(score - m_new[:, None])
    l = l * corr + tl.sum(p, axis=1)
    if boundary:
        v = tl.load(v_base + ki[:, None] * dim + d[None, :], mask=(ki < tk)[:, None], other=0.0)
    else:
        v = tl.load(v_base + ki[:, None] * dim + d[None, :])
    acc = acc * corr[:, None] + tl.dot(p.to(v.dtype), v)
    return m_new, l, acc


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": bq, "BLOCK_K": bk}, num_warps=w, num_stages=s)
        for bq, bk, w, s in [
            (64, 64, 4, 2), (64, 64, 4, 3), (128, 64, 4, 2), (128, 64, 8, 2),
            (64, 128, 4, 2), (128, 32, 4, 2), (32, 64, 4, 2),
        ]
    ],
    key=["dim", "causal"],
)
@triton.jit
def _flash_attn_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    tq, tk, scale2, causal: tl.constexpr,
    dim: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # one program per (head, query block); K/V streamed in BLOCK_K tiles.
    # tiles come in three flavors so masking never touches the hot loop:
    # interior (no masks), causal-diagonal (position mask), ragged end.
    qblock, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, dim)
    qi = qblock * BLOCK_Q + tl.arange(0, BLOCK_Q)

    q = tl.load(q_ptr + head * tq * dim + qi[:, None] * dim + d[None, :],
                mask=(qi < tq)[:, None], other=0.0)
    k_base = k_ptr + head * tk * dim
    v_base = v_ptr + head * tk * dim

    m = tl.full((BLOCK_Q,), float("-inf"), tl.float32)  # running max (log2)
    l = tl.zeros((BLOCK_Q,), tl.float32)                # running sumexp2
    acc = tl.zeros((BLOCK_Q, dim), tl.float32)          # running p @ V

    # causal: query qi attends keys ki <= qi + offs (offs aligns the ends)
    offs = tk - tq
    hi = tk if not causal else tl.minimum((qblock + 1) * BLOCK_Q + offs, tk)
    # interior tiles: fully in range, fully visible to every query in the block
    hi_safe = tl.maximum(qblock * BLOCK_Q + offs + 1, 0) if causal else tk
    hi_safe = (tl.minimum(hi_safe, tk) // BLOCK_K) * BLOCK_K

    for kstart in range(0, hi_safe, BLOCK_K):
        m, l, acc = _flash_inner(q, k_base, v_base, m, l, acc, kstart, qi, offs,
                                 tk, scale2, causal_diag=False, boundary=False,
                                 dim=dim, BLOCK_K=BLOCK_K)
    for kstart in range(hi_safe, hi, BLOCK_K):
        m, l, acc = _flash_inner(q, k_base, v_base, m, l, acc, kstart, qi, offs,
                                 tk, scale2, causal_diag=causal, boundary=True,
                                 dim=dim, BLOCK_K=BLOCK_K)

    out = acc / l[:, None]
    tl.store(out_ptr + head * tq * dim + qi[:, None] * dim + d[None, :],
             out.to(out_ptr.dtype.element_ty), mask=(qi < tq)[:, None])


# -------------------------------------------------------- paged_attention


@triton.jit
def _paged_attn_kernel(
    q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, m_ptr, l_ptr, acc_ptr,
    max_blocks, nchunks, scale2,
    group: tl.constexpr, kv_heads: tl.constexpr, dim: tl.constexpr,
    BS: tl.constexpr, BG: tl.constexpr, CHUNK_BLOCKS: tl.constexpr,
):
    # one program per (sequence, KV head, chunk of blocks): all `group`
    # query heads of the KV head ride along as a [BG, dim] tile (padded to
    # tensor-core width), so each K/V block is read from HBM exactly once
    # for the whole group. The kernel never sees a contiguous cache — the
    # block table *is* the layout, so fork/COW-shared blocks are read in
    # place (no gather). Chunks write (m, l, acc) partials; a second
    # kernel merges them (same split-K pattern as attention_decode).
    seq, kvh, chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d = tl.arange(0, dim)
    g = tl.arange(0, BG)
    q = tl.load(q_ptr + (seq * kv_heads * group + kvh * group + g[:, None]) * dim
                + d[None, :], mask=(g < group)[:, None], other=0.0)

    seqlen = tl.load(len_ptr + seq)
    m = tl.full((BG,), float("-inf"), tl.float32)  # running max (log2)
    l = tl.zeros((BG,), tl.float32)                # running sumexp2
    acc = tl.zeros((BG, dim), tl.float32)

    offs = tl.arange(0, BS)
    nblocks = (seqlen + BS - 1) // BS
    lo = chunk * CHUNK_BLOCKS
    hi = tl.minimum(lo + CHUNK_BLOCKS, nblocks)
    for i in range(lo, hi):
        blk = tl.load(bt_ptr + seq * max_blocks + i)
        valid = i * BS + offs < seqlen  # partial tail block
        addr = ((blk * BS + offs[:, None]) * kv_heads + kvh) * dim + d[None, :]
        k = tl.load(k_ptr + addr, mask=valid[:, None], other=0.0)
        score = tl.dot(q, tl.trans(k)) * scale2  # [BG, BS]
        score = tl.where(valid[None, :], score, float("-inf"))

        m_new = tl.maximum(m, tl.max(score, axis=1))
        corr = tl.exp2(m - m_new)
        p = tl.exp2(score - m_new[:, None])
        l = l * corr + tl.sum(p, axis=1)
        v = tl.load(v_ptr + addr, mask=valid[:, None], other=0.0)
        acc = acc * corr[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new

    base = ((seq * kv_heads + kvh) * nchunks + chunk) * BG
    tl.store(m_ptr + base + g, m)
    tl.store(l_ptr + base + g, l)
    tl.store(acc_ptr + base * dim + g[:, None] * dim + d[None, :], acc)


@triton.jit
def _paged_merge_kernel(
    m_ptr, l_ptr, acc_ptr, out_ptr, nchunks,
    group: tl.constexpr, kv_heads: tl.constexpr, dim: tl.constexpr,
    BG: tl.constexpr,
):
    # one program per (sequence, KV head): fold chunk partials together
    # with the online rule (empty chunks carry m=-inf, l=0 — no-ops).
    seq, kvh = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, dim)
    g = tl.arange(0, BG)

    M = tl.full((BG,), float("-inf"), tl.float32)
    L = tl.zeros((BG,), tl.float32)
    ACC = tl.zeros((BG, dim), tl.float32)
    for c in range(0, nchunks):
        base = ((seq * kv_heads + kvh) * nchunks + c) * BG
        m = tl.load(m_ptr + base + g)
        l = tl.load(l_ptr + base + g)
        acc = tl.load(acc_ptr + base * dim + g[:, None] * dim + d[None, :])
        M_new = tl.maximum(M, m)
        c0 = tl.exp2(M - M_new)
        c1 = tl.exp2(m - M_new)
        L = L * c0 + l * c1
        ACC = ACC * c0[:, None] + acc * c1[:, None]
        M = M_new

    out = ACC / L[:, None]
    tl.store(out_ptr + (seq * kv_heads * group + kvh * group + g[:, None]) * dim
             + d[None, :], out.to(out_ptr.dtype.element_ty),
             mask=(g < group)[:, None])


_PAGED_CHUNK_TOKENS = 512  # split-K granularity along the sequence


def paged_attention(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Decode attention over a paged KV pool (cache/paged.py layout).

    q: [B, q_heads, dim]; k/v_pool: [num_blocks, block_size, kv_heads, dim];
    block_tables: [B, max_blocks] int32 (rows padded past each seq's
    blocks); seq_lens: [B] int32. q_heads must be a multiple of kv_heads
    (GQA). Returns [B, q_heads, dim].
    """
    batch, q_heads, dim = q.shape
    _, block_size, kv_heads, _ = k_pool.shape
    assert q_heads % kv_heads == 0
    assert dim & (dim - 1) == 0 and block_size & (block_size - 1) == 0
    group = q_heads // kv_heads
    BG = max(16, triton.next_power_of_2(group))  # tensor-core minimum
    max_blocks = block_tables.shape[1]
    chunk_blocks = max(_PAGED_CHUNK_TOKENS // block_size, 1)
    nchunks = triton.cdiv(max_blocks, chunk_blocks)

    dev = q.device
    m = torch.empty(batch * kv_heads * nchunks * BG, device=dev, dtype=torch.float32)
    l = torch.empty_like(m)
    acc = torch.empty(m.shape[0] * dim, device=dev, dtype=torch.float32)
    out = torch.empty_like(q)
    _paged_attn_kernel[(batch, kv_heads, nchunks)](
        q, k_pool, v_pool, block_tables, seq_lens, m, l, acc,
        max_blocks, nchunks, _LOG2E / dim**0.5,
        group=group, kv_heads=kv_heads, dim=dim, BS=block_size, BG=BG,
        CHUNK_BLOCKS=chunk_blocks,
    )
    _paged_merge_kernel[(batch, kv_heads)](
        m, l, acc, out, nchunks,
        group=group, kv_heads=kv_heads, dim=dim, BG=BG,
    )
    return out


def flash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True
) -> torch.Tensor:
    """Tiled flash attention: q [heads, Tq, dim], k/v [heads, Tk, dim].

    dim must be a power of two (16..128 — the usual head dims). Causal
    alignment matches SDPA: the last query attends all Tk keys.
    """
    heads, tq, dim = q.shape
    tk = k.shape[1]
    assert dim & (dim - 1) == 0 and dim >= 16, "head dim must be a power of 2 >= 16"
    out = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(tq, meta["BLOCK_Q"]), heads)
    _flash_attn_kernel[grid](
        q, k, v, out, tq, tk, _LOG2E / dim**0.5, causal=causal, dim=dim,
    )
    return out
