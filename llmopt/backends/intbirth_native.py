"""Native intbirth shim — axiom's C++ integer kernels, llmopt calling
conventions.

Adopts the axiom-lab `intbirth` extension (pybind11, built at
$AXIOM_BUILD_DIR, default /Users/artin/code/axiom/build-rel) when the
.so imports; otherwise falls back to pure-numpy implementations with
identical semantics. The contract is llmopt/intmath.py's (the certified
integer core, RESULTS 2026-07-31/08-01 R1a/R1b/R2/R3a):

- ``rdiv``: round-half-away integer division, ONE function
  program-wide (P3-form equality proven on all integers, axiom relay
  2026-07-31-4). Rounding PLACEMENT stays the caller's spec.
- ``int_gemm(a, w)``: a[r,K] @ w[N,K]^T -> [r,N] — the torch
  ``int_mm`` convention from intmath.py, exact int64.
- ``int_gemm_nt(a, w)``: a[r,K] @ w[K,N] -> [r,N].
- ``int_gemm_xty(x, y)``: x[r,K]^T @ y[r,N] -> [K,N] (the dW outer
  form).

All four take/return numpy int64 (the native module's boundary type);
torch callers go through ``.numpy()`` at the edge. Never score the
adoption by weight distance — parity here is FUNCTION-space
bit-identity on the outputs (tests/test_intbirth_native.py), per the
2026-07-06 joint-perm closure.
"""
import os
import sys

import numpy as np

AXIOM_BUILD_DIR = os.environ.get(
    "AXIOM_BUILD_DIR", "/Users/artin/code/axiom/build-rel")

_native = None
try:
    if os.path.isdir(AXIOM_BUILD_DIR):
        if AXIOM_BUILD_DIR not in sys.path:
            sys.path.append(AXIOM_BUILD_DIR)
        import intbirth as _native  # noqa: F401
except Exception:
    _native = None

HAVE_NATIVE = _native is not None
available = HAVE_NATIVE  # legacy-style flag; same truth


# ------------------------------------------------ pure-python fallback
def _as_i64(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.int64))


def _py_rdiv(x, d):
    """Round-half-away division, elementwise; d a python int scalar.

    Mirrors intmath.rdiv (sign-symmetric): sign(x)*((|x|+d//2)//d).
    """
    x = _as_i64(x)
    d = int(d)
    return np.sign(x) * ((np.abs(x) + d // 2) // d)


def _py_int_gemm(a, w):
    """a[r,K] @ w[N,K]^T -> [r,N], exact int64."""
    return _as_i64(a) @ _as_i64(w).T


def _py_int_gemm_nt(a, w):
    """a[r,K] @ w[K,N] -> [r,N], exact int64."""
    return _as_i64(a) @ _as_i64(w)


def _py_int_gemm_xty(x, y):
    """x[r,K]^T @ y[r,N] -> [K,N] (the dW outer form), exact int64."""
    return _as_i64(x).T @ _as_i64(y)


# ------------------------------------------------------- public bind
if HAVE_NATIVE:
    def rdiv(x, d):
        """Native round-half-away division (intbirth.rdiv)."""
        return _native.rdiv(_as_i64(x), int(d))

    def int_gemm(a, w):
        """Native a[r,K] @ w[N,K]^T -> [r,N] (intbirth.int_gemm)."""
        return _native.int_gemm(_as_i64(a), _as_i64(w))

    def int_gemm_nt(a, w):
        """Native a[r,K] @ w[K,N] -> [r,N] (intbirth.int_gemm_nt)."""
        return _native.int_gemm_nt(_as_i64(a), _as_i64(w))

    def int_gemm_xty(x, y):
        """Native x[r,K]^T @ y[r,N] -> [K,N] (intbirth.int_gemm_xty)."""
        return _native.int_gemm_xty(_as_i64(x), _as_i64(y))
else:
    rdiv = _py_rdiv
    int_gemm = _py_int_gemm
    int_gemm_nt = _py_int_gemm_nt
    int_gemm_xty = _py_int_gemm_xty

# The fallbacks stay importable under their own names so the parity
# battery can run native-vs-python on the same process.
py_rdiv = _py_rdiv
py_int_gemm = _py_int_gemm
py_int_gemm_nt = _py_int_gemm_nt
py_int_gemm_xty = _py_int_gemm_xty
