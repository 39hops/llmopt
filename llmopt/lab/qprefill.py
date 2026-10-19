"""Chunked prefill for the fused-linear mlx-lm towers (adopted from
scratch/qwen_metal_chunked.py and scratch/qwen_metal_prefill_head.py,
OBSERVATIONS QWEN-METAL-CHUNKED-PREFILL-0 and QWEN-METAL-PREFILL-STUDY-0).

The prompt is fed to the model in consecutive chunks through the cache
objects the model creates (ArraysCache conv tail and recurrent state per
DeltaNet layer, KVCache with offset per attention layer); every chunk's
outputs and every cache array are evaluated before the next chunk is
built, so the lazy graph never spans two chunks. The vocabulary head is
applied to the last hidden position only in the generation path
(head="final"; on the fused kernels bit-identical final logits, caches
and decode against the full projection, QWEN-METAL-PREFILL-STUDY-0
section 5; under the decode-once GEMM path the caches and decode stay
bit-identical while the final logits differ at the GEMM's reduction
level, QWEN-METAL-PREFILL-GEMM-0) and to every position in the research
path (head="full", the per-position logits for
qualification and E0 comparisons). The body (embedding, layers with
their caches, final norm) runs over the whole chunk in both.

Requires the mlx-lm layout model.language_model.model (body) and
model.language_model.lm_head (untied head)."""
from __future__ import annotations


def cache_arrays(cache):
    """Every array a cache list holds (for mx.eval and for comparison)."""
    import mlx.core as mx
    out = []
    for c in cache:
        st = c.state
        if isinstance(st, (list, tuple)):
            out += [a for a in st if isinstance(a, mx.array)]
        elif isinstance(st, mx.array):
            out.append(st)
    return out


def split_body_head(model):
    lm = getattr(model, "language_model", None)
    if lm is None or not hasattr(lm, "lm_head") or not hasattr(lm, "model"):
        raise ValueError("prefill needs model.language_model.model and an untied model.language_model.lm_head")
    return lm.model, lm.lm_head


def prefill_chunked(model, ids, cache, chunk: int = 256, head: str = "final", on_chunk=None):
    """Prefill ids (a list of token ids, one unpadded prompt) into cache in
    chunks of `chunk` tokens. head="final": the head runs on the last
    hidden row of each chunk and the last chunk's [1, 1, V] logits are
    returned. head="full": the head runs on every position and the last
    chunk's [1, S, V] logits are returned; on_chunk(start, logits) is
    called with each chunk's logits after evaluation, before the next chunk
    is built, so a caller can move them to host. head="none": the body
    only (hidden states evaluated), returns None."""
    import mlx.core as mx
    if head not in ("final", "full", "none"):
        raise ValueError(head)
    if chunk < 1:
        raise ValueError("chunk must be at least 1")
    body, lm = split_body_head(model)
    last = None
    for s in range(0, len(ids), chunk):
        h = body(mx.array([ids[s:s + chunk]]), cache)
        if head == "full":
            lg = lm(h)
        elif head == "final":
            lg = lm(h[:, -1:, :])
        else:
            lg = None
        mx.eval(lg if lg is not None else h, *cache_arrays(cache))
        if on_chunk is not None and lg is not None:
            on_chunk(s, lg)
        last = lg
        del h
    return last
