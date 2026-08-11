"""Parity battery for llmopt/backends/intbirth_native.py.

Native-vs-python FUNCTION-space bit-identity (never weight distance —
RESULTS 6163 joint-perm closure): same int64 inputs, byte-equal
outputs, across shapes including the degenerate edges. Same-device,
same-process by construction (cross-device gate comparisons stay
forbidden; sigma never transports)."""
import numpy as np
import pytest

from llmopt.backends import intbirth_native as ib

needs_native = pytest.mark.skipif(
    not ib.HAVE_NATIVE, reason="intbirth .so unavailable")

SHAPES = [  # (r, K, N) incl. 1xK, Kx1, and empty edges
    (4, 8, 6), (1, 8, 6), (4, 1, 6), (4, 8, 1), (1, 1, 1),
    (0, 8, 6), (4, 0, 6), (4, 8, 0), (17, 33, 9),
]


def _rand(shape, rng, lo=-(1 << 20), hi=1 << 20):
    return rng.integers(lo, hi, size=shape, dtype=np.int64)


@needs_native
@pytest.mark.parametrize("r,k,n", SHAPES)
def test_gemm_parity(r, k, n):
    rng = np.random.default_rng(20260811 + r * 1000 + k * 10 + n)
    a = _rand((r, k), rng)
    w_nk = _rand((n, k), rng)
    w_kn = _rand((k, n), rng)
    y = _rand((r, n), rng)
    for native, py, args in [
        (ib.int_gemm, ib.py_int_gemm, (a, w_nk)),
        (ib.int_gemm_nt, ib.py_int_gemm_nt, (a, w_kn)),
        (ib.int_gemm_xty, ib.py_int_gemm_xty, (a, y)),
    ]:
        got, want = native(*args), py(*args)
        assert got.dtype == np.int64 and got.shape == want.shape
        assert np.array_equal(got, want)


@needs_native
@pytest.mark.parametrize("d", [1, 2, 3, 7, 512, 1000, 1 << 20])
def test_rdiv_parity(d):
    rng = np.random.default_rng(d)
    x = _rand((257,), rng, -(1 << 40), 1 << 40)
    assert np.array_equal(ib.rdiv(x, d), ib.py_rdiv(x, d))


@needs_native
def test_rdiv_half_away_negatives():
    # round-half-away contract, both signs, exact half cases
    x = np.array([5, -5, 4, -4, 3, -3, 0, 1, -1], dtype=np.int64)
    want = np.array([3, -3, 2, -2, 2, -2, 0, 1, -1], dtype=np.int64)
    assert np.array_equal(ib.rdiv(x, 2), want)
    assert np.array_equal(ib.py_rdiv(x, 2), want)


def test_python_fallback_matches_intmath_rdiv():
    """Fallback rdiv == intmath.rdiv (the certified torch form)."""
    torch = pytest.importorskip("torch")
    from llmopt import intmath
    rng = np.random.default_rng(7)
    x = _rand((513,), rng, -(1 << 32), 1 << 32)
    for d in (2, 3, 512, 999):
        want = intmath.rdiv(torch.from_numpy(x), d).numpy()
        assert np.array_equal(ib.py_rdiv(x, d), want)


def test_python_fallback_gemm_matches_intmath_int_mm():
    torch = pytest.importorskip("torch")
    from llmopt import intmath
    rng = np.random.default_rng(11)
    a = _rand((5, 9), rng)
    w = _rand((4, 9), rng)
    want = intmath.int_mm(torch.from_numpy(a),
                          torch.from_numpy(w)).numpy()
    assert np.array_equal(ib.py_int_gemm(a, w), want)
