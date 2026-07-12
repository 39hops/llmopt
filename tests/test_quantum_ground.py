"""Variational ground-state engine vs exact diagonalization."""
import numpy as np
import pytest

from llmopt.quantum.ground import (ansatz_state, build_tfim, energy,
                                   exact_ground, param_shift_grad)


def test_tfim_classical_limit():
    # h=0: classical Ising chain, E0 = -(n-1)
    for n in (3, 5):
        assert abs(exact_ground(build_tfim(n, 0.0)) + (n - 1)) < 1e-9


def test_tfim_strong_field_limit():
    # h >> 1: E0 -> -n*h (all spins along X)
    n, h = 4, 50.0
    assert abs(exact_ground(build_tfim(n, h)) + n * h) / (n * h) < 0.01


def test_ansatz_normalized():
    rng = np.random.default_rng(0)
    psi = ansatz_state(rng.normal(size=12), 4, 3)
    assert abs(np.linalg.norm(psi) - 1) < 1e-9


def test_param_shift_matches_finite_difference():
    n, layers = 3, 2
    H = build_tfim(n, 1.0)
    rng = np.random.default_rng(1)
    p = rng.normal(size=layers * n)
    g = param_shift_grad(p, H, n, layers)
    for k in range(len(p)):
        eps = 1e-5
        pp, pm = p.copy(), p.copy()
        pp[k] += eps
        pm[k] -= eps
        fd = (energy(pp, H, n, layers) - energy(pm, H, n, layers)) / (2 * eps)
        assert abs(g[k] - fd) < 1e-5


def test_variational_never_below_truth():
    # the referee built into physics: E(psi) >= E0 for ANY psi
    n = 4
    H = build_tfim(n, 1.0)
    e0 = exact_ground(H)
    rng = np.random.default_rng(2)
    for _ in range(20):
        e = energy(rng.normal(size=2 * n), H, n, 2)
        assert e >= e0 - 1e-9
