"""Problem-level parallelism for CPU benches (spec:
2026-07-07-engine-optimizations-design.md, O3).

Benches are embarrassingly parallel across problems. pmap() fans a
module-level worker over items with a fork-context pool: fork (not
macOS's default spawn) so sympy state carries into workers without
re-import, and workers stay pure sympy/CPU. Per-item timeouts belong
INSIDE the worker (SIGALRM is per-process); workers return values,
never raise. Model-backed benches must keep jobs=1 — torch MPS/CUDA
contexts do not fork safely."""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def pmap(worker: Callable[[T], R], items: Sequence[T],
         jobs: int | None = None) -> list[R]:
    """Order-preserving parallel map. jobs=1 is a true serial bypass
    (identical code path to a plain loop, for determinism checks)."""
    jobs = default_jobs() if jobs is None else jobs
    if jobs <= 1 or len(items) <= 1:
        return [worker(it) for it in items]
    ctx = mp.get_context("fork")
    with ctx.Pool(min(jobs, len(items))) as pool:
        return pool.map(worker, list(items))
