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

Matrix path. prefill_chunked's default matmul="gemm" runs its body and head calls
inside qmetal.gemm_scope(True), so the fused residents take the decode-once
float32 GEMM for tiles at or above their per-codec threshold during this
call only (qualified in QWEN-METAL-PREFILL-GEMM-0 for exactly this entry
point: one unpadded prompt, chunked, caches evaluated per chunk);
matmul="fused" is the fused-only control (gemm_scope(False)); matmul=None
leaves the process default. Nothing outside the call inherits the path:
token-by-token decode, batched or speculative-verify calls dispatch by the
process default (fused) whatever their row count. The entry point refuses
batched input (a nested list) and non-integer ids.

Serving memory. ServingConfig carries the serving defaults (chunk 256,
matmul "gemm", a 2 GiB MLX allocator-cache limit) and
configure_serving_memory applies the limit explicitly in the serving
process (measured in QWEN-METAL-PREFILL-GEMM-0: the device footprint at
4,096 tokens 19.3 GB without the limit and 11.7 GB with it, at 3.6 percent
of prefill throughput and no decode cost); the library default is never
changed at import. memory_report distinguishes active memory (arrays
alive), peak allocation since the last reset, the allocator pool (cached
free blocks), their sum (the device footprint a process holds) and the
process peak RSS.

Requires the mlx-lm layout model.language_model.model (body) and
model.language_model.lm_head (untied head)."""
from __future__ import annotations

import dataclasses
import resource

MATMUL_PATHS = ("gemm", "fused", None)


@dataclasses.dataclass(frozen=True)
class ServingConfig:
    """Serving defaults for the qualified single-prompt chunked prefill."""
    chunk: int = 256
    matmul: str = "gemm"
    cache_limit_bytes: int = 2 * 1024 ** 3


def configure_serving_memory(cfg: ServingConfig = ServingConfig()) -> int:
    """Apply cfg.cache_limit_bytes to the MLX allocator cache of this
    process; returns the previous limit so a caller can restore it. Refuses
    to configure a process whose qmetal default has the GEMM path on
    (QMETAL_PREFILL_GEMM=1 is the research switch, never a serving
    setting): in a serving process the path is selected per call by
    prefill_chunked(matmul=cfg.matmul) alone."""
    import mlx.core as mx
    from llmopt.lab import qmetal
    if qmetal.PREFILL_GEMM or qmetal._Resident.prefill_gemm:
        raise RuntimeError("serving refuses a process-wide GEMM default (QMETAL_PREFILL_GEMM / _Resident.prefill_gemm); pass matmul per call")
    return int(mx.set_cache_limit(int(cfg.cache_limit_bytes)))


def memory_report() -> dict:
    """MLX's own accounting: active_bytes (arrays alive on the device),
    peak_bytes (peak ACTIVE memory since the last mx.reset_peak_memory;
    the pool is not in it), cache_bytes (allocator pool held for reuse),
    footprint_bytes = active + cache (what the process holds on the
    device by that accounting; the OS may report more), and
    rss_peak_bytes, the process peak RSS so far from ru_maxrss (bytes on
    macOS; the field is omitted elsewhere because the unit differs)."""
    import mlx.core as mx
    import sys
    active, cache = int(mx.get_active_memory()), int(mx.get_cache_memory())
    out = {"active_bytes": active, "peak_bytes": int(mx.get_peak_memory()), "cache_bytes": cache,
           "footprint_bytes": active + cache}
    if sys.platform == "darwin":
        out["rss_peak_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return out


def _check_ids(ids) -> list:
    if isinstance(ids, (str, bytes)) or not hasattr(ids, "__len__") or len(ids) == 0:
        raise ValueError("ids must be a non-empty sequence of token ids (one unpadded prompt)")
    out = []
    for t in ids:
        if isinstance(t, bool) or not isinstance(t, int) or int(t) < 0:
            try:
                import numpy as np
                ok = isinstance(t, np.integer) and int(t) >= 0
            except ImportError:
                ok = False
            if not ok:
                raise ValueError("ids must be non-negative integers; batched (nested) prompts are not supported")
        out.append(int(t))
    return out


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


def prefill_chunked(model, ids, cache, chunk: int = 256, head: str = "final", on_chunk=None, matmul="gemm"):
    """Prefill ids (a list of token ids, one unpadded prompt) into cache in
    chunks of `chunk` tokens. head="final": the head runs on the last
    hidden row of each chunk and the last chunk's [1, 1, V] logits are
    returned. head="full": the head runs on every position and the last
    chunk's [1, S, V] logits are returned; on_chunk(start, logits) is
    called with each chunk's logits after evaluation, before the next chunk
    is built, so a caller can move them to host. head="none": the body
    only (hidden states evaluated), returns None. matmul: "gemm" (the
    default: the decode-once GEMM path for this call, qualified for this
    entry point), "fused" (the fused-only control) or None (the process
    default, fused unless the research switch is set); see the module
    docstring."""
    import mlx.core as mx
    from llmopt.lab import qmetal
    if head not in ("final", "full", "none"):
        raise ValueError(head)
    if chunk < 1:
        raise ValueError("chunk must be at least 1")
    if matmul not in MATMUL_PATHS:
        raise ValueError(f"matmul must be one of {MATMUL_PATHS}, got {matmul!r}")
    ids = _check_ids(ids)
    body, lm = split_body_head(model)
    last = None
    for s in range(0, len(ids), chunk):
        # the scope covers this chunk's body and head calls only; on_chunk
        # runs outside it, so whatever a callback issues (a decode step, a
        # batched call) dispatches by the process default
        with (qmetal.gemm_scope(matmul == "gemm") if matmul is not None else _NoScope()):
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


class _NoScope:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
