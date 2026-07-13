"""Variational ground-state engine, rung 1 (spec:
2026-07-12-variational-ground-engine-design.md; methods not
molecules — model Hamiltonians only).

The variational principle IS the verifier: <psi|H|psi> >= E0 for any
normalized psi, so energy can never score below truth — propose/
verify with the referee built into physics. Exact diagonalization at
small n gives the perfect oracle to measure error against.

Pieces: TFIM Hamiltonian (dense, n <= 12), exact ground energy, a
minimal numpy statevector simulator (RY + CZ-ring hardware-efficient
ansatz), energy expectation, parameter-shift gradients (exact for RY:
dE/dtheta = (E(theta + pi/2) - E(theta - pi/2)) / 2).
"""
from __future__ import annotations

import numpy as np

_X = np.array([[0.0, 1.0], [1.0, 0.0]])
_Z = np.array([[1.0, 0.0], [0.0, -1.0]])
_I = np.eye(2)


def _kron_at(op: np.ndarray, i: int, n: int) -> np.ndarray:
    m = np.array([[1.0]])
    for k in range(n):
        m = np.kron(m, op if k == i else _I)
    return m


def build_tfim(n: int, h: float) -> np.ndarray:
    """H = -sum_i Z_i Z_{i+1} - h * sum_i X_i (open chain)."""
    H = np.zeros((2**n, 2**n))
    for i in range(n - 1):
        H -= _kron_at(_Z, i, n) @ _kron_at(_Z, i + 1, n)
    for i in range(n):
        H -= h * _kron_at(_X, i, n)
    return H


def exact_ground(H: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(H)[0])


def _apply_ry(state: np.ndarray, theta: float, i: int, n: int) -> np.ndarray:
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    st = state.reshape(2**i, 2, 2 ** (n - i - 1))
    out = np.empty_like(st)
    out[:, 0, :] = c * st[:, 0, :] - s * st[:, 1, :]
    out[:, 1, :] = s * st[:, 0, :] + c * st[:, 1, :]
    return out.reshape(-1)


def _apply_cz(state: np.ndarray, i: int, j: int, n: int) -> np.ndarray:
    idx = np.arange(2**n)
    bi = (idx >> (n - 1 - i)) & 1
    bj = (idx >> (n - 1 - j)) & 1
    signs = np.where((bi & bj) == 1, -1.0, 1.0)
    return state * signs


def ansatz_state(params: np.ndarray, n: int, layers: int) -> np.ndarray:
    """Hardware-efficient: per layer, RY on every qubit then a CZ
    ring. layers=0 with one RY row = product (mean-field) state.
    params shape: ((layers or 1) * n,)."""
    state = np.zeros(2**n)
    state[0] = 1.0
    rows = max(layers, 1)
    th = params.reshape(rows, n)
    for L in range(rows):
        for i in range(n):
            state = _apply_ry(state, th[L, i], i, n)
        if layers > 0:  # entangle between RY rows (none for product)
            for i in range(n):
                state = _apply_cz(state, i, (i + 1) % n, n)
    return state


def energy(params: np.ndarray, H: np.ndarray, n: int,
           layers: int) -> float:
    psi = ansatz_state(params, n, layers)
    return float(psi @ H @ psi)


def param_shift_grad(params: np.ndarray, H: np.ndarray, n: int,
                     layers: int) -> np.ndarray:
    """Exact gradient for RY generators (parameter-shift rule)."""
    g = np.zeros_like(params)
    for k in range(len(params)):
        p = params.copy()
        p[k] += np.pi / 2
        ep = energy(p, H, n, layers)
        p[k] -= np.pi
        em = energy(p, H, n, layers)
        g[k] = 0.5 * (ep - em)
    return g


def hva_state(params: np.ndarray, n: int) -> np.ndarray:
    """Hamiltonian-variational ansatz (the plateau-breaker measured
    2026-07-12: hardware-efficient RY+CZ stalls ~1% at TFIM
    criticality regardless of depth or restarts — the ansatz must
    carry the HAMILTONIAN'S structure). Layers alternate
    exp(-i a * H_zz) (diagonal phases) and exp(-i b * H_x) (RX on
    every qubit), starting from |+>^n. params: (layers, 2)."""
    th = params.reshape(-1, 2)
    state = np.full(2**n, 2.0 ** (-n / 2), dtype=np.complex128)
    idx = np.arange(2**n)
    bits = (idx[:, None] >> (n - 1 - np.arange(n))) & 1
    z = 1.0 - 2.0 * bits            # (+1/-1) per qubit
    zz = (z[:, :-1] * z[:, 1:]).sum(axis=1)   # sum_i z_i z_{i+1}
    for a, b in th:
        state = state * np.exp(1j * a * zz)   # exp(-i a * (-ZZ sum))
        c, s = np.cos(b), np.sin(b)
        for i in range(n):
            st = state.reshape(2**i, 2, 2 ** (n - i - 1))
            out = np.empty_like(st)
            out[:, 0, :] = c * st[:, 0, :] + 1j * s * st[:, 1, :]
            out[:, 1, :] = 1j * s * st[:, 0, :] + c * st[:, 1, :]
            state = out.reshape(-1)
    return state


def hva_energy(params: np.ndarray, H: np.ndarray, n: int) -> float:
    psi = hva_state(params, n)
    return float(np.real(np.conj(psi) @ H @ psi))


def hva_optimize(H: np.ndarray, n: int, layers: int, iters: int = 300,
                 lr: float = 0.05, seed: int = 0) -> float:
    """Adam on central finite differences (generators are Pauli SUMS,
    so the 2-point shift rule doesn't apply)."""
    rng = np.random.default_rng(seed)
    params = rng.normal(0, 0.1, layers * 2)
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    best = np.inf
    for t in range(1, iters + 1):
        g = np.zeros_like(params)
        for k in range(len(params)):
            p = params.copy()
            p[k] += 1e-4
            ep = hva_energy(p, H, n)
            p[k] -= 2e-4
            em = hva_energy(p, H, n)
            g[k] = (ep - em) / 2e-4
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        params -= lr * (m / (1 - 0.9**t)) / (
            np.sqrt(v / (1 - 0.999**t)) + 1e-8)
        best = min(best, hva_energy(params, H, n))
    return best


# ------------------------------------------------ structure search
# Layer vocabulary (rung 2): both families' building blocks. An
# ansatz STRUCTURE is a token list, e.g. ["xm","zz","xm"] ~ HVA,
# ["ry","cz","ry","cz"] ~ hardware-efficient. Params per token:
# ry/rx: n, zz/xm: 1, cz: 0.
_TOK_PARAMS = {"ry": None, "rx": None, "zz": 1, "xm": 1, "cz": 0}


def struct_nparams(tokens: list[str], n: int) -> int:
    return sum(n if _TOK_PARAMS[t] is None else _TOK_PARAMS[t]
               for t in tokens)


def struct_state(tokens: list[str], params: np.ndarray,
                 n: int) -> np.ndarray:
    """Execute a token-list ansatz from |+>^n (complex statevector)."""
    state = np.full(2**n, 2.0 ** (-n / 2), dtype=np.complex128)
    idx = np.arange(2**n)
    bits = (idx[:, None] >> (n - 1 - np.arange(n))) & 1
    z = 1.0 - 2.0 * bits
    zz = (z[:, :-1] * z[:, 1:]).sum(axis=1)
    k = 0
    for t in tokens:
        if t == "zz":
            state = state * np.exp(1j * params[k] * zz)
            k += 1
        elif t == "xm":
            b = params[k]
            k += 1
            c, s = np.cos(b), np.sin(b)
            for i in range(n):
                st = state.reshape(2**i, 2, 2 ** (n - i - 1))
                out = np.empty_like(st)
                out[:, 0, :] = c * st[:, 0, :] + 1j * s * st[:, 1, :]
                out[:, 1, :] = 1j * s * st[:, 0, :] + c * st[:, 1, :]
                state = out.reshape(-1)
        elif t in ("ry", "rx"):
            for i in range(n):
                th = params[k]
                k += 1
                c, s = np.cos(th / 2), np.sin(th / 2)
                st = state.reshape(2**i, 2, 2 ** (n - i - 1))
                out = np.empty_like(st)
                if t == "ry":
                    out[:, 0, :] = c * st[:, 0, :] - s * st[:, 1, :]
                    out[:, 1, :] = s * st[:, 0, :] + c * st[:, 1, :]
                else:
                    out[:, 0, :] = c * st[:, 0, :] - 1j * s * st[:, 1, :]
                    out[:, 1, :] = -1j * s * st[:, 0, :] + c * st[:, 1, :]
                state = out.reshape(-1)
        elif t == "cz":
            for i in range(n):
                state = _apply_cz(state, i, (i + 1) % n, n)
    return state


def struct_energy(tokens: list[str], params: np.ndarray, H: np.ndarray,
                  n: int) -> float:
    psi = struct_state(tokens, params, n)
    return float(np.real(np.conj(psi) @ H @ psi))


def struct_optimize(H: np.ndarray, tokens: list[str], n: int,
                    iters: int = 120, lr: float = 0.08,
                    seed: int = 0) -> float:
    """Adam on central finite differences over the token structure."""
    npar = struct_nparams(tokens, n)
    if npar == 0:
        return struct_energy(tokens, np.zeros(0), H, n)
    rng = np.random.default_rng(seed)
    params = rng.normal(0, 0.1, npar)
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    best = np.inf
    for t in range(1, iters + 1):
        g = np.zeros_like(params)
        for k in range(npar):
            p = params.copy()
            p[k] += 1e-4
            ep = struct_energy(tokens, p, H, n)
            p[k] -= 2e-4
            em = struct_energy(tokens, p, H, n)
            g[k] = (ep - em) / 2e-4
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        params -= lr * (m / (1 - 0.9**t)) / (
            np.sqrt(v / (1 - 0.999**t)) + 1e-8)
        best = min(best, struct_energy(tokens, params, H, n))
    return best


def optimize(H: np.ndarray, n: int, layers: int, iters: int = 300,
             lr: float = 0.1, seed: int = 0) -> tuple[float, np.ndarray]:
    """Adam on parameter-shift gradients. Returns (best E, params)."""
    rng = np.random.default_rng(seed)
    rows = max(layers, 1)
    params = rng.normal(0, 0.1, rows * n)
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    best_e, best_p = np.inf, params.copy()
    for t in range(1, iters + 1):
        g = param_shift_grad(params, H, n, layers)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        params -= lr * (m / (1 - 0.9**t)) / (
            np.sqrt(v / (1 - 0.999**t)) + 1e-8)
        e = energy(params, H, n, layers)
        if e < best_e:
            best_e, best_p = e, params.copy()
    return best_e, best_p
