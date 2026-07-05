"""Numerical-equivalence harness: assert an optimized implementation matches
its reference, bit-exact (token ids) or within tolerance (logits/logprobs).

The spine of trust for every optimization in this library: prompt-lookup and
greedy speculative decoding must be token-identical to vanilla greedy; a
quantized model must stay within its KL budget; a KV-cache-reusing path must
produce logits within fp tolerance of recompute.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EquivalenceReport:
    equal: bool
    detail: str

    def __bool__(self) -> bool:
        return self.equal


def assert_tokens_equal(reference: list[int], optimized: list[int]) -> EquivalenceReport:
    """Bit-exact token comparison with first-divergence diagnostics."""
    if reference == optimized:
        return EquivalenceReport(True, f"identical ({len(reference)} tokens)")
    n = min(len(reference), len(optimized))
    for i in range(n):
        if reference[i] != optimized[i]:
            return EquivalenceReport(
                False,
                f"diverge at position {i}: ref={reference[i]} opt={optimized[i]} "
                f"(context ref[{max(0, i - 3)}:{i + 1}]={reference[max(0, i - 3):i + 1]})",
            )
    return EquivalenceReport(
        False, f"length mismatch: ref={len(reference)} opt={len(optimized)}, common prefix ok"
    )


def assert_logits_close(
    reference, optimized, *, atol: float = 1e-4, rtol: float = 1e-3
) -> EquivalenceReport:
    """Tolerance comparison for logits/logprobs tensors or arrays.

    Reports max abs error and worst position instead of a bare boolean.
    """
    import numpy as np

    ref = np.asarray(_to_numpy(reference), dtype=np.float64)
    opt = np.asarray(_to_numpy(optimized), dtype=np.float64)
    if ref.shape != opt.shape:
        return EquivalenceReport(False, f"shape mismatch: {ref.shape} vs {opt.shape}")
    err = np.abs(ref - opt)
    bound = atol + rtol * np.abs(ref)
    ok = bool((err <= bound).all())
    worst = np.unravel_index(int((err - bound).argmax()), err.shape)
    return EquivalenceReport(
        ok,
        f"max_abs_err={err.max():.3e} at {worst} "
        f"(atol={atol}, rtol={rtol}, {'PASS' if ok else 'FAIL'})",
    )


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return x
