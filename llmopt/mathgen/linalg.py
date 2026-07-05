"""Verifiable linear algebra problems (sympy-grounded).

Same reverse-construction discipline as problems.py: answers exist by
design, checks verify rather than compare where possible.

- determinant / rank: direct integer answers.
- eigenvalues: built as P D P^-1 with chosen integer eigenvalues and a
  unimodular integer P, so the matrix is integer and the spectrum is
  exactly what we planted. Checked as a multiset.
- inverse: A is a product of integer elementary matrices (det ±1), so
  A^-1 is integer. check() *verifies*: A @ prediction == I — any correct
  inverse passes regardless of formatting.
"""

from __future__ import annotations

import random

import sympy as sp

from llmopt.mathgen.problems import Problem


def _unimodular(rng: random.Random, n: int) -> sp.Matrix:
    """Random integer matrix with det ±1: product of elementary row ops."""
    m = sp.eye(n)
    for _ in range(2 * n + rng.randint(1, 3)):
        i, j = rng.sample(range(n), 2)
        m[i, :] = m[i, :] + rng.randint(-2, 2) * m[j, :]
    return m


def _fmt(m: sp.Matrix) -> str:
    return str(m.tolist())


def make_determinant(level: int, seed: int) -> Problem:
    rng = random.Random(f"det-{level}-{seed}")
    n = 2 if level == 1 else 3
    a = sp.Matrix(n, n, lambda i, j: rng.randint(-5, 5))
    ans = a.det()
    return Problem(
        prompt=f"Compute the determinant of the matrix {_fmt(a)}",
        answer=sp.sstr(ans), kind="determinant", level=level, _expr=ans,
    )


def make_eigenvalues(level: int, seed: int) -> Problem:
    rng = random.Random(f"eig-{level}-{seed}")
    n = 2 if level <= 2 else 3
    eigs = sorted(rng.sample(range(-5, 7), n))
    p = _unimodular(rng, n)
    a = p * sp.diag(*eigs) * p.inv()
    return Problem(
        prompt=(f"Find all eigenvalues of the matrix {_fmt(a)}. "
                "Answer with the eigenvalues separated by commas."),
        answer=", ".join(str(e) for e in eigs),
        kind="eigenvalues", level=level,
        _expr=[sp.Integer(e) for e in eigs],
    )


def make_inverse(level: int, seed: int) -> Problem:
    rng = random.Random(f"inv-{level}-{seed}")
    n = 2 if level <= 2 else 3
    a = _unimodular(rng, n)
    if a == sp.eye(n):
        return make_inverse(level, seed + 1_000_003)
    return Problem(
        prompt=(f"Compute the inverse of the matrix {_fmt(a)}. "
                "Answer as a list of rows like [[a, b], [c, d]]."),
        answer=_fmt(a.inv()), kind="matrix_inverse", level=level, _expr=a,
    )


def make_rank(level: int, seed: int) -> Problem:
    """rank planted: r independent rows + dependent integer combinations."""
    rng = random.Random(f"rank-{level}-{seed}")
    n = 3 if level <= 2 else 4
    r = rng.randint(1, n - 1)
    rows = [[rng.randint(-4, 4) for _ in range(n)] for _ in range(r)]
    base = sp.Matrix(rows)
    if base.rank() != r:  # unlucky draw: dependent "independent" rows
        return make_rank(level, seed + 1_000_003)
    while len(rows) < n:
        coeffs = [rng.randint(-2, 2) for _ in range(r)]
        rows.append([
            sum(c * rows[k][j] for k, c in enumerate(coeffs)) for j in range(n)
        ])
    order = list(range(n))
    rng.shuffle(order)
    a = sp.Matrix([rows[i] for i in order])
    ans = sp.Integer(a.rank())
    return Problem(
        prompt=f"Compute the rank of the matrix {_fmt(a)}",
        answer=str(ans), kind="rank", level=level, _expr=ans,
    )


MAKERS = {
    "determinant": make_determinant,
    "eigenvalues": make_eigenvalues,
    "matrix_inverse": make_inverse,
    "rank": make_rank,
}
