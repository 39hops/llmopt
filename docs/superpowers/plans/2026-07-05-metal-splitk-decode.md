# Split-K Attention Decode (Metal/MLX) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `attention_decode_splitk` (and a GQA multi-head variant) to `llmopt/kernels/metal.py` that parallelizes single-query attention across the whole GPU, beating the existing single-threadgroup kernel.

**Architecture:** Two Metal kernels mirroring the Triton split-K decode in `llmopt/kernels/triton_kernels.py`: a partial kernel (one threadgroup per BLOCK_T key chunk, exp2-domain online softmax, writes float32 `(m, l, acc)` partials) and a merge kernel (one threadgroup combines partials with the rescale rule). The existing educational `attention_decode` stays untouched as baseline.

**Tech Stack:** MLX `mx.fast.metal_kernel`, pytest (with `pytest.importorskip("mlx.core")`), Apple silicon required for tests/benches.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-05-metal-splitk-decode-design.md`.
- Partial buffers (`m`, `l`, `acc`) are **float32** regardless of input dtype.
- Softmax in exp2 domain: fold `_LOG2E = 1.4426950408889634` into the scale (`scale2 = _LOG2E / sqrt(dim)`), use `metal::exp2`.
- Old `attention_decode` is NOT modified or removed.
- All tests must pass on Apple silicon: `python -m pytest tests/test_metal_kernels.py -v`.
- Repo convention: benchmarks report losses honestly; kernel docstrings state measured results only after measuring.

---

### Task 1: Single-head split-K kernel

**Files:**
- Modify: `llmopt/kernels/metal.py` (append after `attention_decode`, plus two new `_SRC` strings after `_ATTN_DECODE_SRC` and kernel objects after `_attn`)
- Test: `tests/test_metal_kernels.py`

**Interfaces:**
- Consumes: existing `_kernel(name, src, inputs)` helper in `metal.py`.
- Produces: `attention_decode_splitk(q: mx.array, k: mx.array, v: mx.array, block_t: int = 512) -> mx.array` — `q [dim]`, `k/v [T, dim]` → `[dim]`, same contract as `attention_decode`. Also module-level `_LOG2E = 1.4426950408889634`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_metal_kernels.py` (extend the import line to include `attention_decode_splitk`):

```python
def _decode_ref(q, k, v):
    return mx.softmax((k @ q) / q.shape[0] ** 0.5) @ v


def test_splitk_matches_softmax():
    mx.random.seed(3)
    t, dim = 3000, 64  # several chunks + a ragged final chunk
    q = mx.random.normal((dim,))
    k = mx.random.normal((t, dim))
    v = mx.random.normal((t, dim))
    assert mx.allclose(attention_decode_splitk(q, k, v), _decode_ref(q, k, v), atol=1e-4)


@pytest.mark.parametrize("t", [1, 7, 512, 513])
def test_splitk_boundary_lengths(t):
    mx.random.seed(4)
    dim = 32
    q = mx.random.normal((dim,))
    k = mx.random.normal((t, dim))
    v = mx.random.normal((t, dim))
    assert mx.allclose(attention_decode_splitk(q, k, v), _decode_ref(q, k, v), atol=1e-4)


def test_splitk_fp16():
    mx.random.seed(5)
    t, dim = 2000, 128
    q = mx.random.normal((dim,)).astype(mx.float16)
    k = mx.random.normal((t, dim)).astype(mx.float16)
    v = mx.random.normal((t, dim)).astype(mx.float16)
    ref = _decode_ref(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32))
    out = attention_decode_splitk(q, k, v)
    assert out.dtype == mx.float16
    assert mx.allclose(out.astype(mx.float32), ref, atol=2e-3)


def test_splitk_extreme_scores_stable():
    dim = 32
    q = mx.ones((dim,)) * 6.0
    k = mx.concatenate([mx.ones((1, dim)) * 10, -mx.ones((1100, dim)) * 10])
    v = mx.random.normal((1101, dim))
    out = attention_decode_splitk(q, k, v)
    assert mx.isfinite(out).all()
    assert mx.allclose(out, v[0], atol=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metal_kernels.py -v -k splitk`
Expected: FAIL at import — `ImportError: cannot import name 'attention_decode_splitk'`

- [ ] **Step 3: Implement the kernels**

In `llmopt/kernels/metal.py`, add after `_ATTN_DECODE_SRC`:

```python
_LOG2E = 1.4426950408889634  # softmax via exp2: exp(x) == exp2(x * log2(e))

_ATTN_PARTIAL_SRC = """
    // split-K phase 1: one threadgroup per BLOCK_T-key chunk.
    // online softmax in exp2 domain; writes float32 partials
    // (m, l, acc[DIM]) for the merge kernel. Mirrors the Triton
    // _attn_decode_kernel in kernels/triton_kernels.py.
    constexpr uint TG = 256;
    uint chunk = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint t0 = chunk * (uint)BLOCK_T;

    // scores for this chunk into shared memory (thread tid takes
    // keys t0+tid, t0+tid+TG, ...), tracking a per-thread max.
    threadgroup float sc[(uint)BLOCK_T];
    threadgroup float red[TG];
    float local_max = -INFINITY;
    for (uint j = tid; j < (uint)BLOCK_T; j += TG) {
        float s = -INFINITY;
        uint t = t0 + j;
        if (t < (uint)TLEN) {
            s = 0.0f;
            for (uint d = 0; d < (uint)DIM; d++)
                s += (float)q[d] * (float)k[t * (uint)DIM + d];
            s *= scale2[0];
        }
        sc[j] = s;
        local_max = metal::max(local_max, s);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = TG / 2; r > 0; r >>= 1) {
        if (tid < r) red[tid] = metal::max(red[tid], red[tid + r]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // p = exp2(score - m) in place; l = sum(p) by tree reduction.
    // an all -inf chunk (can't happen: nchunks = ceil(T/BLOCK_T))
    // would give m = -inf; guard exp2(-inf - -inf) via l == 0 path
    // in the merge (corr * l = 0 contribution).
    float local_sum = 0.0f;
    for (uint j = tid; j < (uint)BLOCK_T; j += TG) {
        float p = (sc[j] == -INFINITY) ? 0.0f : metal::exp2(sc[j] - m);
        sc[j] = p;
        local_sum += p;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = TG / 2; r > 0; r >>= 1) {
        if (tid < r) red[tid] += red[tid + r];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) { m_out[chunk] = m; l_out[chunk] = red[0]; }

    // acc[d] = sum_j p[j] * V[t0+j][d]; threads stride over d so each
    // V element is read exactly once by exactly one thread.
    uint chunk_len = metal::min((uint)BLOCK_T, (uint)TLEN - t0);
    for (uint d = tid; d < (uint)DIM; d += TG) {
        float a = 0.0f;
        for (uint j = 0; j < chunk_len; j++)
            a += sc[j] * (float)v[(t0 + j) * (uint)DIM + d];
        acc_out[chunk * (uint)DIM + d] = a;
    }
"""

_ATTN_MERGE_SRC = """
    // split-K phase 2: one threadgroup merges C partials.
    // out[d] = sum_c corr_c * acc[c][d] / sum_c corr_c * l[c],
    // corr_c = exp2(m_c - M). Mirrors Triton _attn_merge_kernel.
    constexpr uint TG = 256;
    uint tid = thread_position_in_threadgroup.x;

    threadgroup float red[TG];
    float local_max = -INFINITY;
    for (uint c = tid; c < (uint)C; c += TG)
        local_max = metal::max(local_max, m_in[c]);
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = TG / 2; r > 0; r >>= 1) {
        if (tid < r) red[tid] = metal::max(red[tid], red[tid + r]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float M = red[0];

    // C is small (T / BLOCK_T): every thread recomputes the scalar
    // denominator serially rather than staging another reduction.
    float L = 0.0f;
    for (uint c = 0; c < (uint)C; c++)
        L += metal::exp2(m_in[c] - M) * l_in[c];

    for (uint d = tid; d < (uint)DIM; d += TG) {
        float a = 0.0f;
        for (uint c = 0; c < (uint)C; c++)
            a += metal::exp2(m_in[c] - M) * acc_in[c * (uint)DIM + d];
        out[d] = (T)(a / L);
    }
"""
```

Add kernel objects after `_attn = _kernel(...)`:

```python
_attn_partial = mx.fast.metal_kernel(
    name="llmopt_attn_decode_partial",
    input_names=["q", "k", "v", "scale2"],
    output_names=["m_out", "l_out", "acc_out"],
    source=_ATTN_PARTIAL_SRC,
)
_attn_merge = _kernel("llmopt_attn_decode_merge", _ATTN_MERGE_SRC,
                      ["m_in", "l_in", "acc_in"])
```

Add the wrapper after `attention_decode`:

```python
def attention_decode_splitk(
    q: mx.array, k: mx.array, v: mx.array, block_t: int = 512
) -> mx.array:
    """Single-query attention, split-K: q [dim], k/v [T, dim] -> [dim].

    Phase 1 gives each threadgroup a BLOCK_T chunk of keys (parallel
    across the device); phase 2 merges the per-chunk (m, l, acc)
    partials. Partials are float32 regardless of input dtype.
    """
    t, dim = k.shape
    nchunks = (t + block_t - 1) // block_t
    scale2 = mx.array([_LOG2E / dim**0.5], dtype=mx.float32)
    m, l, acc = _attn_partial(
        inputs=[q, k, v, scale2],
        template=[("T", q.dtype), ("DIM", dim), ("TLEN", t), ("BLOCK_T", block_t)],
        grid=(nchunks * _TG, 1, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(nchunks,), (nchunks,), (nchunks, dim)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (out,) = _attn_merge(
        inputs=[m, l, acc],
        template=[("T", q.dtype), ("DIM", dim), ("C", nchunks)],
        grid=(_TG, 1, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(dim,)],
        output_dtypes=[q.dtype],
    )
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metal_kernels.py -v`
Expected: all tests PASS (old + new). If a splitk test fails, debug with
`t=513, block_t=512` (two chunks, second nearly empty) — the most likely
bug sites are the `-INFINITY` padding of ragged chunks and the fp32/fp16
cast points.

- [ ] **Step 5: Commit**

```bash
git add llmopt/kernels/metal.py tests/test_metal_kernels.py
git commit -m "feat: kernels/metal — split-K attention decode (two-phase, exp2 softmax)"
```

---

### Task 2: Benchmark split-K vs old kernel, naive, and mx.fast SDPA

**Files:**
- Modify: `scripts/bench_metal_kernels.py` (attention section of `main()`)
- Modify: `llmopt/kernels/metal.py:19-25` (module docstring — record measured numbers)

**Interfaces:**
- Consumes: `attention_decode_splitk(q, k, v, block_t=512)` from Task 1.
- Produces: measured numbers in the bench output and updated docstring; a chosen default `block_t` if 512 is not best.

- [ ] **Step 1: Extend the benchmark**

In `scripts/bench_metal_kernels.py`, extend the import to include `attention_decode_splitk` and replace the attention section at the end of `main()` with:

```python
    for t in (2048, 8192, 32768):
        hd = 128
        q, k, v = (mx.random.normal(s) for s in ((hd,), (t, hd), (t, hd)))
        naive = lambda: mx.softmax((k @ q) / hd**0.5) @ v
        sdpa = lambda: mx.fast.scaled_dot_product_attention(
            q[None, None, None, :], k[None, None], v[None, None], scale=1.0 / hd**0.5
        )[0, 0, 0]
        print(f"attention decode T={t} dim={hd}:")
        print(f"  naive softmax     {bench(naive):8.1f} us")
        print(f"  mx.fast sdpa      {bench(sdpa):8.1f} us")
        print(f"  llmopt v1 (32thr) {bench(attention_decode, q, k, v):8.1f} us")
        for bt in (256, 512, 1024):
            us = bench(lambda: attention_decode_splitk(q, k, v, block_t=bt))
            print(f"  llmopt split-K bt={bt:<5d}{us:8.1f} us")
```

- [ ] **Step 2: Run the benchmark**

Run: `python scripts/bench_metal_kernels.py`
Expected: split-K beats `llmopt v1` at all T and beats naive softmax@V at
large T. Losing to `mx.fast sdpa` is expected and reported honestly.

- [ ] **Step 3: Record results**

Update the measured-results paragraph of the `metal.py` module docstring
(lines 19–25) with the actual numbers: keep the existing honest note about
v1 losing, add one sentence stating split-K's speedup over v1 and vs naive
at the largest T, and the chosen default `block_t`. If a `block_t` other
than 512 wins consistently, change the wrapper default and say so.

- [ ] **Step 4: Commit**

```bash
git add scripts/bench_metal_kernels.py llmopt/kernels/metal.py
git commit -m "bench: metal split-K decode vs v1/naive/mx.fast sdpa; record numbers"
```

---

### Task 3: GQA multi-head split-K (phase C)

**Files:**
- Modify: `llmopt/kernels/metal.py` (new `_SRC` pair + wrapper)
- Test: `tests/test_metal_kernels.py`

**Interfaces:**
- Consumes: Task 1's structure (`_LOG2E`, `_TG`, `_kernel`).
- Produces: `attention_decode_gqa(q: mx.array, k: mx.array, v: mx.array, block_t: int = 512) -> mx.array` — `q [H, dim]`, `k/v [T, KVH, dim]` with `H % KVH == 0` → `[H, dim]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_metal_kernels.py` (extend import with `attention_decode_gqa`):

```python
def _gqa_ref(q, k, v):
    h, dim = q.shape
    kvh = k.shape[1]
    group = h // kvh
    outs = []
    for i in range(h):
        ki, vi = k[:, i // group, :], v[:, i // group, :]
        outs.append(mx.softmax((ki @ q[i]) / dim**0.5) @ vi)
    return mx.stack(outs)


@pytest.mark.parametrize("h,kvh", [(8, 8), (8, 2), (4, 1)])
def test_gqa_matches_reference(h, kvh):
    mx.random.seed(6)
    t, dim = 1500, 64
    q = mx.random.normal((h, dim))
    k = mx.random.normal((t, kvh, dim))
    v = mx.random.normal((t, kvh, dim))
    assert mx.allclose(attention_decode_gqa(q, k, v), _gqa_ref(q, k, v), atol=1e-4)


def test_gqa_single_head_matches_splitk():
    mx.random.seed(7)
    t, dim = 700, 32
    q = mx.random.normal((dim,))
    k = mx.random.normal((t, dim))
    v = mx.random.normal((t, dim))
    a = attention_decode_gqa(q[None], k[:, None], v[:, None])[0]
    b = attention_decode_splitk(q, k, v)
    assert mx.allclose(a, b, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metal_kernels.py -v -k gqa`
Expected: FAIL at import — `cannot import name 'attention_decode_gqa'`

- [ ] **Step 3: Implement**

The kernel bodies are the Task 1 sources with head indexing; grid gains a
second dimension for heads. Add to `metal.py`:

```python
_GQA_PARTIAL_SRC = """
    // split-K + GQA: grid (chunks, q_heads). q head h reads kv head
    // h / (H / KVH); K/V layout [T, KVH, DIM].
    constexpr uint TG = 256;
    uint chunk = threadgroup_position_in_grid.x;
    uint h = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint t0 = chunk * (uint)BLOCK_T;
    uint kvh = h / ((uint)H / (uint)KVH);
    uint krow = (uint)KVH * (uint)DIM;          // stride of one timestep
    const device T* qh = q + h * (uint)DIM;

    threadgroup float sc[(uint)BLOCK_T];
    threadgroup float red[TG];
    float local_max = -INFINITY;
    for (uint j = tid; j < (uint)BLOCK_T; j += TG) {
        float s = -INFINITY;
        uint t = t0 + j;
        if (t < (uint)TLEN) {
            s = 0.0f;
            const device T* kt = k + t * krow + kvh * (uint)DIM;
            for (uint d = 0; d < (uint)DIM; d++)
                s += (float)qh[d] * (float)kt[d];
            s *= scale2[0];
        }
        sc[j] = s;
        local_max = metal::max(local_max, s);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = TG / 2; r > 0; r >>= 1) {
        if (tid < r) red[tid] = metal::max(red[tid], red[tid + r]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint j = tid; j < (uint)BLOCK_T; j += TG) {
        float p = (sc[j] == -INFINITY) ? 0.0f : metal::exp2(sc[j] - m);
        sc[j] = p;
        local_sum += p;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = TG / 2; r > 0; r >>= 1) {
        if (tid < r) red[tid] += red[tid + r];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint pidx = h * (uint)C + chunk;            // partials laid out [H, C]
    if (tid == 0) { m_out[pidx] = m; l_out[pidx] = red[0]; }

    uint chunk_len = metal::min((uint)BLOCK_T, (uint)TLEN - t0);
    for (uint d = tid; d < (uint)DIM; d += TG) {
        float a = 0.0f;
        for (uint j = 0; j < chunk_len; j++)
            a += sc[j] * (float)v[(t0 + j) * krow + kvh * (uint)DIM + d];
        acc_out[pidx * (uint)DIM + d] = a;
    }
"""

_GQA_MERGE_SRC = """
    // grid (1, q_heads): merge C partials per head; layout [H, C].
    constexpr uint TG = 256;
    uint h = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint base = h * (uint)C;

    threadgroup float red[TG];
    float local_max = -INFINITY;
    for (uint c = tid; c < (uint)C; c += TG)
        local_max = metal::max(local_max, m_in[base + c]);
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = TG / 2; r > 0; r >>= 1) {
        if (tid < r) red[tid] = metal::max(red[tid], red[tid + r]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float M = red[0];

    float L = 0.0f;
    for (uint c = 0; c < (uint)C; c++)
        L += metal::exp2(m_in[base + c] - M) * l_in[base + c];

    for (uint d = tid; d < (uint)DIM; d += TG) {
        float a = 0.0f;
        for (uint c = 0; c < (uint)C; c++)
            a += metal::exp2(m_in[base + c] - M)
                 * acc_in[(base + c) * (uint)DIM + d];
        out[h * (uint)DIM + d] = (T)(a / L);
    }
"""

_gqa_partial = mx.fast.metal_kernel(
    name="llmopt_attn_gqa_partial",
    input_names=["q", "k", "v", "scale2"],
    output_names=["m_out", "l_out", "acc_out"],
    source=_GQA_PARTIAL_SRC,
)
_gqa_merge = _kernel("llmopt_attn_gqa_merge", _GQA_MERGE_SRC,
                     ["m_in", "l_in", "acc_in"])


def attention_decode_gqa(
    q: mx.array, k: mx.array, v: mx.array, block_t: int = 512
) -> mx.array:
    """GQA decode attention: q [H, dim], k/v [T, KVH, dim] -> [H, dim].

    Same split-K structure as attention_decode_splitk with a 2D grid
    (key chunks x query heads); query head h attends to kv head
    h // (H // KVH).
    """
    h, dim = q.shape
    t, kvh, _ = k.shape
    assert h % kvh == 0, f"q heads {h} not a multiple of kv heads {kvh}"
    nchunks = (t + block_t - 1) // block_t
    scale2 = mx.array([_LOG2E / dim**0.5], dtype=mx.float32)
    tmpl = [("T", q.dtype), ("DIM", dim), ("TLEN", t), ("BLOCK_T", block_t),
            ("H", h), ("KVH", kvh), ("C", nchunks)]
    m, l, acc = _gqa_partial(
        inputs=[q, k, v, scale2],
        template=tmpl,
        grid=(nchunks * _TG, h, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(h, nchunks), (h, nchunks), (h, nchunks, dim)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (out,) = _gqa_merge(
        inputs=[m, l, acc],
        template=[("T", q.dtype), ("DIM", dim), ("C", nchunks), ("H", h)],
        grid=(_TG, h, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(h, dim)],
        output_dtypes=[q.dtype],
    )
    return out
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/test_metal_kernels.py -v`
Expected: all PASS.

- [ ] **Step 5: Add GQA line to the benchmark and run it**

In `scripts/bench_metal_kernels.py` after the split-K loop (inside the
`for t in ...` loop), add a GQA comparison at Llama-like shapes:

```python
        hq, hkv = 32, 8
        qg = mx.random.normal((hq, hd))
        kg = mx.random.normal((t, hkv, hd))
        vg = mx.random.normal((t, hkv, hd))
        per_head = lambda: mx.stack([
            mx.softmax((kg[:, i // (hq // hkv)] @ qg[i]) / hd**0.5)
            @ vg[:, i // (hq // hkv)] for i in range(hq)
        ])
        print(f"  gqa H={hq}/KVH={hkv}:")
        print(f"    per-head naive  {bench(per_head):8.1f} us")
        print(f"    llmopt gqa      {bench(attention_decode_gqa, qg, kg, vg):8.1f} us")
```

Run: `python scripts/bench_metal_kernels.py`. Append the GQA numbers to
the `metal.py` docstring results paragraph.

- [ ] **Step 6: Commit**

```bash
git add llmopt/kernels/metal.py tests/test_metal_kernels.py scripts/bench_metal_kernels.py
git commit -m "feat: kernels/metal — GQA multi-head split-K decode"
```
