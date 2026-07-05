"""Tests for eval/stats.py and eval/equivalence.py (numpy only, no torch)."""

import math

import numpy as np
import pytest

from llmopt.eval.equivalence import assert_logits_close, assert_tokens_equal
from llmopt.eval.stats import (
    bootstrap_ci,
    eval_noise,
    paired_bootstrap_pvalue,
    pass_at_k,
    wilson_interval,
)


# ---- pass@k ----

def test_pass_at_1_is_success_rate():
    assert pass_at_k(n=10, c=3, k=1) == pytest.approx(0.3)


def test_pass_at_k_all_correct():
    assert pass_at_k(n=5, c=5, k=3) == 1.0


def test_pass_at_k_none_correct():
    assert pass_at_k(n=5, c=0, k=3) == pytest.approx(0.0)


def test_pass_at_k_matches_bruteforce():
    # n=6, c=2, k=3: 1 - C(4,3)/C(6,3) = 1 - 4/20
    assert pass_at_k(6, 2, 3) == pytest.approx(1 - 4 / 20)


def test_pass_at_k_k_greater_than_n_raises():
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 5)


# ---- intervals / tests ----

def test_bootstrap_ci_contains_mean():
    vals = [0, 1, 1, 0, 1, 1, 1, 0, 1, 0]
    ci = bootstrap_ci(vals, n_boot=2000, seed=1)
    assert ci.lo <= ci.point <= ci.hi
    assert ci.point == pytest.approx(0.6)


def test_paired_bootstrap_detects_clear_win():
    rng = np.random.default_rng(0)
    b = rng.normal(0, 1, 200)
    a = b + 0.5  # a clearly better
    diff, p = paired_bootstrap_pvalue(a, b, n_boot=2000, seed=2)
    assert diff == pytest.approx(0.5)
    assert p < 0.01


def test_paired_bootstrap_null_not_significant():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 200)
    diff, p = paired_bootstrap_pvalue(a, a.copy(), n_boot=500, seed=4)
    assert diff == 0.0
    assert p > 0.4  # ties -> ~1.0 under (boot_means <= 0)


def test_wilson_narrower_than_naive_at_extremes():
    ci = wilson_interval(0, 20)
    assert ci.lo == 0.0
    assert ci.hi > 0.0  # never collapses to [0,0] like normal approx
    assert ci.hi < 0.25


def test_eval_noise_mde():
    runs = [[1, 0, 1, 1], [1, 1, 1, 0], [0, 1, 1, 1]]
    out = eval_noise(runs)
    assert out["mean"] == pytest.approx(0.75)
    assert not math.isnan(out["min_detectable_effect"])


# ---- equivalence ----

def test_tokens_equal_pass():
    r = assert_tokens_equal([1, 2, 3], [1, 2, 3])
    assert r


def test_tokens_equal_reports_divergence_position():
    r = assert_tokens_equal([1, 2, 3, 4], [1, 2, 9, 4])
    assert not r
    assert "position 2" in r.detail


def test_tokens_equal_length_mismatch():
    r = assert_tokens_equal([1, 2, 3], [1, 2])
    assert not r
    assert "length mismatch" in r.detail


def test_logits_close_within_tol():
    ref = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert assert_logits_close(ref, ref + 1e-6)


def test_logits_close_fails_beyond_tol():
    ref = np.zeros((2, 2))
    opt = ref.copy()
    opt[1, 1] = 0.1
    r = assert_logits_close(ref, opt, atol=1e-4, rtol=0)
    assert not r
    assert "(1, 1)" in r.detail.replace("(np.int64(1), np.int64(1))", "(1, 1)")


def test_logits_close_shape_mismatch():
    assert not assert_logits_close(np.zeros((2, 2)), np.zeros((2, 3)))
