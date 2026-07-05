# Split-K attention decode for Metal (MLX)

**Date:** 2026-07-05
**Status:** Approved

## Goal

Turn the documented loss in `kernels/metal.py` into a win: the current
`attention_decode` runs one 32-thread threadgroup on the whole device and
loses to naive softmax@V. Port the Triton split-K decode structure
(`kernels/triton_kernels.py:_attn_decode_kernel` / `_attn_merge_kernel`)
to Metal so the kernel parallelizes across the whole GPU.

## Scope

1. **This session (phase A):** single-head `attention_decode_splitk(q, k, v)`
   — `q [dim]`, `k/v [T, dim]` → `[dim]` — added *alongside* the existing
   educational kernel, which stays as the documented baseline.
2. **Follow-up (phase C, same session if A wins its benchmark):** GQA
   multi-head via 2D grid `(chunks, q_heads)`; K/V indexed by
   `head // group_size`.

## Design

### Phase 1 kernel — `llmopt_attn_decode_partial`

- Grid: `ceil(T / BLOCK_T)` threadgroups, TG = 256 threads each; each
  threadgroup owns one contiguous chunk of keys.
- Scores computed in the exp2 domain: fold `log2(e) / sqrt(dim)` into the
  scale (mirrors `_LOG2E` in the Triton version).
- Per chunk: threads score their strided keys, shared-memory tree
  reduction to a chunk-local max `m`, then accumulate
  `l = Σ exp2(score − m)` and `acc[dim] = Σ p·V` with a second reduction.
- Outputs (multiple outputs from one `mx.fast.metal_kernel`):
  `m_out [C]`, `l_out [C]`, `acc_out [C, dim]` — **all float32** even for
  fp16 inputs (rescale factors underflow fp16).

### Phase 2 kernel — `llmopt_attn_decode_merge`

- One threadgroup reads all `C` partials, applies the online rescale rule
  `corr = exp2(m − M)`, writes final `[dim]` in the input dtype.
- `C ≈ T / BLOCK_T` is small; a simple serial-ish merge is acceptable.

### Launch heuristics

- `BLOCK_T` chosen at launch so `nchunks` can saturate the GPU
  (start at 512 per the Triton port; tune in the benchmark).
- Strided inner loops handle arbitrary `dim`; no power-of-2 restriction.

## Testing

Extend `tests/test_metal_kernels.py`:

- Elementwise match vs pure-MLX softmax reference, fp32 and fp16, with
  tolerances matching the existing attention test.
- Boundary cases: `T < BLOCK_T`, `T` not a multiple of `BLOCK_T`, `T = 1`.

Extend `scripts/bench_metal_kernels.py`: compare old kernel, split-K,
naive softmax@V, and `mx.fast.scaled_dot_product_attention`. Report
losses honestly per repo convention.

## Non-goals

- Prefill/flash tiled attention (separate queued item).
- Wiring into `backends/mlx_backend.py` (separate item).
