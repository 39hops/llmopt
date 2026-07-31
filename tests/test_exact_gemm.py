import platform
import random

import pytest

mx = pytest.importorskip("mlx.core")
if platform.system() != "Darwin":
    pytest.skip("Metal kernels are Apple-silicon only",
                allow_module_level=True)

from llmopt.kernels.metal import exact_gemm  # noqa: E402


def _oracle(A, B):
    """Python big-int reference — exact by definition."""
    return [[sum(ar[k] * br[k] for k in range(len(ar)))
             for br in B] for ar in A]


def _rand_mat(r, m, k, bound):
    return [[r.randint(-bound, bound) for _ in range(k)]
            for _ in range(m)]


def test_exact_vs_bigint_random():
    r = random.Random("exact-gemm-1")
    for m, n, k in ((3, 5, 7), (16, 64, 256), (1, 1, 1000)):
        A = _rand_mat(r, m, k, (1 << 15) - 1)
        B = _rand_mat(r, n, k, (1 << 15) - 1)
        got = exact_gemm(mx.array(A, dtype=mx.int32),
                         mx.array(B, dtype=mx.int32))
        want = _oracle(A, B)
        assert got.tolist() == want            # EXACT, no tolerance


def test_extremes_and_signs():
    v = (1 << 15) - 1
    A = mx.full((2, 512), v, dtype=mx.int32)
    B = mx.full((2, 512), -v, dtype=mx.int32)
    got = exact_gemm(A, B)
    assert got.tolist() == [[-v * v * 512] * 2] * 2


def test_run_to_run_determinism():
    r = random.Random("exact-gemm-2")
    A = mx.array(_rand_mat(r, 8, 128, 30000), dtype=mx.int32)
    B = mx.array(_rand_mat(r, 8, 128, 30000), dtype=mx.int32)
    first = exact_gemm(A, B).tolist()
    for _ in range(3):
        assert exact_gemm(A, B).tolist() == first


def test_bounds_rejected():
    big = mx.full((1, 4), 1 << 15, dtype=mx.int32)
    ok = mx.ones((1, 4), dtype=mx.int32)
    with pytest.raises(AssertionError):
        exact_gemm(big, ok)
