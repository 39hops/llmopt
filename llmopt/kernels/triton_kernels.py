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
  weighted-V accumulator) and writes its partial; partials merge on the
  host with the same online-softmax rule. Never materializes the [T]
  score vector. Unlike the Metal version (one threadgroup, educational),
  split-K actually parallelizes across the GPU.
- flash_attention: the full tiled kernel (Flash-2 style) — one program per
  (head, query-block), inner loop over key blocks with online softmax over
  a [BLOCK_Q, BLOCK_K] score tile, optional causal mask.

All kernels are verified elementwise against pure-torch references in
tests; benchmark with scripts/bench_triton_kernels.py. Measured (RTX 3080,
4096x4096 fp16): rmsnorm 3.7x over unfused ops and 1.5x over
torch.nn.RMSNorm; swiglu 2.7x over unfused. attention_decode *loses* to
naive softmax@V at T=8192 (same lesson as Metal: a cuBLAS GEMV that
saturates the GPU beats a readable split-K at decode sizes — the split-K
win appears when the [T] score vector no longer fits nicely, or fused into
larger kernels). flash_attention runs at ~0.5x of torch's fused SDPA
(heads=32, T=2048, dim=128) — the remaining 2x is what FlashAttention's
hand-tuned pipelining/layout work buys over a readable two-loop kernel.
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
    tlen, scale, dim: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # one program per key chunk: online softmax over its BLOCK_T keys,
    # writes partial (m, l, acc[dim]) for the host-side merge.
    chunk = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    dmask = d < dim
    q = tl.load(q_ptr + d, mask=dmask, other=0.0).to(tl.float32)

    t = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < tlen
    k = tl.load(k_ptr + t[:, None] * dim + d[None, :],
                mask=tmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
    score = tl.sum(k * q[None, :], axis=1) * scale
    score = tl.where(tmask, score, float("-inf"))

    m = tl.max(score)
    p = tl.exp(score - m)
    l = tl.sum(p)
    v = tl.load(v_ptr + t[:, None] * dim + d[None, :],
                mask=tmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
    acc = tl.sum(p[:, None] * v, axis=0)

    tl.store(m_ptr + chunk, m)
    tl.store(l_ptr + chunk, l)
    tl.store(acc_ptr + chunk * dim + d, acc, mask=dmask)


def attention_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Single-query attention: q [dim], k/v [T, dim] -> [dim]."""
    t, dim = k.shape
    BLOCK_T = 512
    nchunks = triton.cdiv(t, BLOCK_T)
    dev = q.device
    m = torch.empty(nchunks, device=dev, dtype=torch.float32)
    l = torch.empty(nchunks, device=dev, dtype=torch.float32)
    acc = torch.empty(nchunks, dim, device=dev, dtype=torch.float32)
    _attn_decode_kernel[(nchunks,)](
        q, k, v, m, l, acc, t, 1.0 / dim**0.5,
        dim=dim, BLOCK_T=BLOCK_T, BLOCK_D=triton.next_power_of_2(dim),
    )
    # merge partials with the same online rule (tiny: nchunks x dim)
    M = m.max()
    corr = torch.exp(m - M)
    return ((corr[:, None] * acc).sum(0) / (corr * l).sum()).to(q.dtype)


# -------------------------------------------------------- flash_attention


@triton.jit
def _flash_attn_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    tq, tk, scale, causal: tl.constexpr,
    dim: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # one program per (head, query block); K/V streamed in BLOCK_K tiles.
    qblock, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    dmask = d < dim
    qi = qblock * BLOCK_Q + tl.arange(0, BLOCK_Q)
    qmask = qi < tq

    q_base = q_ptr + head * tq * dim
    q = tl.load(q_base + qi[:, None] * dim + d[None, :],
                mask=qmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)

    m = tl.full((BLOCK_Q,), float("-inf"), tl.float32)  # running max
    l = tl.zeros((BLOCK_Q,), tl.float32)                # running sumexp
    acc = tl.zeros((BLOCK_Q, BLOCK_D), tl.float32)      # running p @ V

    # causal: query qi only attends keys <= qi + (tk - tq); later blocks skipped
    hi = tk if not causal else (qblock + 1) * BLOCK_Q + (tk - tq)
    for kstart in range(0, hi, BLOCK_K):
        ki = kstart + tl.arange(0, BLOCK_K)
        kmask = ki < tk
        kv_mask = kmask[:, None] & dmask[None, :]
        k_base = k_ptr + head * tk * dim
        k = tl.load(k_base + ki[:, None] * dim + d[None, :], mask=kv_mask, other=0.0).to(tl.float32)

        score = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_Q, BLOCK_K]
        score = tl.where(kmask[None, :], score, float("-inf"))
        if causal:
            score = tl.where(qi[:, None] + (tk - tq) >= ki[None, :], score, float("-inf"))

        m_new = tl.maximum(m, tl.max(score, axis=1))
        corr = tl.exp(m - m_new)                # rescale old state
        p = tl.exp(score - m_new[:, None])
        l = l * corr + tl.sum(p, axis=1)
        v_base = v_ptr + head * tk * dim
        v = tl.load(v_base + ki[:, None] * dim + d[None, :], mask=kv_mask, other=0.0).to(tl.float32)
        acc = acc * corr[:, None] + tl.dot(p, v)
        m = m_new

    out = acc / l[:, None]
    out_base = out_ptr + head * tq * dim
    tl.store(out_base + qi[:, None] * dim + d[None, :], out.to(out_ptr.dtype.element_ty),
             mask=qmask[:, None] & dmask[None, :])


def flash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True
) -> torch.Tensor:
    """Tiled flash attention: q [heads, Tq, dim], k/v [heads, Tk, dim].

    Causal alignment matches SDPA: the last query attends all Tk keys.
    """
    heads, tq, dim = q.shape
    tk = k.shape[1]
    out = torch.empty_like(q)
    # bigger head dims need smaller tiles to fit shared memory (~100KB on Ampere)
    BLOCK_Q = 64 if dim <= 64 else 32
    BLOCK_K = 64
    _flash_attn_kernel[(triton.cdiv(tq, BLOCK_Q), heads)](
        q, k, v, out, tq, tk, 1.0 / dim**0.5, causal=causal,
        dim=dim, BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=triton.next_power_of_2(dim),
        num_stages=2,
    )
    return out
