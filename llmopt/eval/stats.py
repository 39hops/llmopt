"""Statistics for trustworthy evals: pass@k, bootstrap CIs, paired tests,
Wilson intervals, eval-noise estimation. Pure Python + numpy, no torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021, HumanEval).

    n samples drawn, c correct, probability >=1 correct in a random size-k
    subset: 1 - C(n-c, k) / C(n, k), computed stably in log space.
    """
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    if c < 0 or c > n:
        raise ValueError(f"c={c} out of range [0, {n}]")
    if n - c < k:
        return 1.0
    return 1.0 - math.exp(_log_comb(n - c, k) - _log_comb(n, k))


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


@dataclass(frozen=True)
class CI:
    point: float
    lo: float
    hi: float
    level: float


def bootstrap_ci(
    values,
    statistic=np.mean,
    *,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> CI:
    """Percentile bootstrap CI for statistic(values)."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = np.apply_along_axis(statistic, 1, values[idx])
    alpha = (1 - level) / 2
    return CI(
        point=float(statistic(values)),
        lo=float(np.quantile(boots, alpha)),
        hi=float(np.quantile(boots, 1 - alpha)),
        level=level,
    )


def paired_bootstrap_pvalue(
    a, b, *, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Paired bootstrap test for mean(a) > mean(b) on per-item scores.

    Returns (mean_diff, p_value) where p_value approximates
    P(diff <= 0) under the bootstrap distribution. Standard for comparing
    two systems on the same eval items.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired test needs equal-length aligned scores")
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_means = diff[idx].mean(axis=1)
    return float(diff.mean()), float((boot_means <= 0).mean())


def wilson_interval(successes: int, n: int, *, level: float = 0.95) -> CI:
    """Wilson score interval for a binomial proportion (better than normal
    approx at small n / extreme p)."""
    if n == 0:
        raise ValueError("n must be > 0")
    z = _z_for(level)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return CI(point=p, lo=max(0.0, center - half), hi=min(1.0, center + half), level=level)


def _z_for(level: float) -> float:
    # inverse normal CDF via Acklam-lite: only common levels needed
    table = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
    if level in table:
        return table[level]
    # fallback: bisection on erf
    lo, hi = 0.0, 10.0
    target = (1 + level) / 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def eval_noise(run_scores) -> dict:
    """Cross-run noise summary: run_scores is [n_runs][n_items] of scores
    from repeated evals (different seeds/sampling). Returns per-run means,
    std of means, and the min detectable effect (~2 * std) -- if your
    improvement is below MDE, it's noise.
    """
    arr = np.asarray(run_scores, dtype=np.float64)
    means = arr.mean(axis=1)
    return {
        "run_means": means.tolist(),
        "mean": float(means.mean()),
        "std_of_means": float(means.std(ddof=1)) if len(means) > 1 else float("nan"),
        "min_detectable_effect": float(2 * means.std(ddof=1)) if len(means) > 1 else float("nan"),
    }
