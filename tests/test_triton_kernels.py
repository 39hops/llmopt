"""Triton kernels vs pure-torch references (CUDA only)."""

import pytest
import torch

triton = pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA required for Triton kernels", allow_module_level=True)

from llmopt.kernels.triton_kernels import (
    attention_decode,
    flash_attention,
    paged_attention,
    rmsnorm,
    rope,
    swiglu,
)


def _cuda(*shapes, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(s, device="cuda", generator=g) for s in shapes]


def _rmsnorm_ref(x, w, eps=1e-6):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


def test_rmsnorm_matches_reference():
    x, w = _cuda((17, 384), (384,))  # dim not a power of 2 exercises masking
    torch.testing.assert_close(rmsnorm(x, w), _rmsnorm_ref(x, w), atol=1e-5, rtol=1e-5)


def test_swiglu_matches_reference():
    g, u = _cuda((33, 128), (33, 128), seed=1)
    ref = (g * torch.sigmoid(g)) * u
    torch.testing.assert_close(swiglu(g, u), ref, atol=1e-5, rtol=1e-5)


def _rope_ref(x, pos, base=10000.0):
    n, h, d = x.shape
    pair = torch.arange(d // 2, device=x.device, dtype=torch.float32)
    theta = (1.0 / base) ** (2.0 * pair / d)
    ang = pos[:, None, None].float() * theta[None, None, :]
    c, s = torch.cos(ang), torch.sin(ang)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = torch.zeros_like(x)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    return out


def test_rope_matches_reference():
    (x,) = _cuda((5, 3, 64), seed=2)
    pos = torch.tensor([0, 1, 2, 100, 1000], device="cuda")
    torch.testing.assert_close(rope(x, pos), _rope_ref(x, pos), atol=1e-4, rtol=1e-4)


def test_attention_decode_matches_softmax():
    t, dim = 500, 64  # T >> BLOCK_T exercises the split-K merge
    q, k, v = _cuda((dim,), (t, dim), (t, dim), seed=3)
    ref = torch.softmax((k @ q) / dim**0.5, dim=0) @ v
    torch.testing.assert_close(attention_decode(q, k, v), ref, atol=1e-4, rtol=1e-4)


def test_attention_decode_extreme_scores_stable():
    # online softmax must survive score ranges that overflow naive exp
    dim = 32
    q = torch.ones(dim, device="cuda") * 6.0
    k = torch.cat([torch.ones(1, dim, device="cuda") * 10,
                   -torch.ones(63, dim, device="cuda") * 10])
    (v,) = _cuda((64, dim), seed=4)
    out = attention_decode(q, k, v)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, v[0], atol=1e-3, rtol=1e-3)


# tl.dot uses tf32 tensor cores on fp32 inputs (~1e-3 abs error vs ieee);
# that's the intended fast path, so flash tests use a tf32-sized tolerance.
_FLASH_TOL = dict(atol=5e-3, rtol=0.0)


def test_flash_attention_matches_sdpa_causal():
    heads, t, dim = 4, 300, 64  # t not a multiple of block size
    q, k, v = _cuda((heads, t, dim), (heads, t, dim), (heads, t, dim), seed=5)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(flash_attention(q, k, v), ref, **_FLASH_TOL)


def test_flash_attention_noncausal():
    heads, t, dim = 2, 129, 64
    q, k, v = _cuda((heads, t, dim), (heads, t, dim), (heads, t, dim), seed=6)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(flash_attention(q, k, v, causal=False), ref, **_FLASH_TOL)


def _paged_setup(seq_lens, block_size=16, kv_heads=2, dim=32, seed=0):
    """Real BlockTable/PagedTensorStore plumbing, moved to CUDA: write
    random KV per sequence, fork+COW the last one so the kernel reads
    genuinely shared and diverged blocks."""
    from llmopt.cache.paged import BlockAllocator, BlockTable, PagedTensorStore

    g = torch.Generator().manual_seed(seed)
    alloc = BlockAllocator(256)
    store = PagedTensorStore(256, block_size, kv_heads, dim)
    tables = []
    for t in seq_lens[:-1]:
        table = store.bind(BlockTable(alloc, block_size))
        for _ in range(t):
            store.write(table, torch.randn(kv_heads, dim, generator=g),
                        torch.randn(kv_heads, dim, generator=g))
        tables.append(table)
    # last sequence: fork of the first + extra tokens (exercises COW + sharing)
    child = tables[0].fork()
    for _ in range(seq_lens[-1] - tables[0].length):
        store.write(child, torch.randn(kv_heads, dim, generator=g),
                    torch.randn(kv_heads, dim, generator=g))
    tables.append(child)

    max_blocks = max(len(t.blocks) for t in tables)
    bt = torch.zeros(len(tables), max_blocks, dtype=torch.int32, device="cuda")
    for i, t in enumerate(tables):
        bt[i, : len(t.blocks)] = torch.tensor(t.blocks, dtype=torch.int32)
    lens = torch.tensor([t.length for t in tables], dtype=torch.int32, device="cuda")
    return store, tables, bt, lens


def test_paged_attention_matches_gather_reference():
    kv_heads, group, dim = 2, 4, 32
    seq_lens = [50, 17, 70]  # last = fork of first extended past it
    store, tables, bt, lens = _paged_setup(seq_lens, kv_heads=kv_heads, dim=dim)
    k_pool, v_pool = store.k.cuda(), store.v.cuda()
    q = torch.randn(len(tables), kv_heads * group, dim, device="cuda")

    out = paged_attention(q, k_pool, v_pool, bt, lens)

    for s, table in enumerate(tables):
        ks, vs = store.gather(table)  # [T, kv_heads, dim] contiguous truth
        ks, vs = ks.cuda(), vs.cuda()
        for h in range(kv_heads * group):
            kh = h // group
            scores = (ks[:, kh] @ q[s, h]) / dim**0.5
            ref = torch.softmax(scores, dim=0) @ vs[:, kh]
            # tf32 dot tolerance, same as the flash tests
            torch.testing.assert_close(out[s, h], ref, atol=5e-3, rtol=0.0)


def test_paged_attention_cow_divergence_isolated():
    # after COW, parent and forked child must attend different tails
    store, tables, bt, lens = _paged_setup([32, 17, 40], block_size=16)
    parent, child = tables[0], tables[-1]
    assert parent.blocks[:2] == child.blocks[:2]  # full blocks shared
    k_pool, v_pool = store.k.cuda(), store.v.cuda()
    q = torch.randn(1, 8, 32, device="cuda").expand(3, -1, -1).contiguous()

    out = paged_attention(q, k_pool, v_pool, bt, lens)
    assert not torch.allclose(out[0], out[2])  # different lengths/tails


def test_flash_attention_cross_lengths_causal():
    # tq < tk (decode-with-context shape): last query sees all keys
    heads, tq, tk, dim = 2, 65, 200, 64
    q, k, v = _cuda((heads, tq, dim), (heads, tk, dim), (heads, tk, dim), seed=7)
    mask = torch.ones(tq, tk, device="cuda", dtype=torch.bool).tril(diagonal=tk - tq)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(flash_attention(q, k, v), ref, **_FLASH_TOL)
