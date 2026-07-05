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
(heads=32, T=2048, dim=128, causal).
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
