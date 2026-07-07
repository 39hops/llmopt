"""Verifiable number-theory problems. Oracle = integer arithmetic:
check() recomputes, never string-matches. Answers are integers (or
small tuples), which also makes these ideal short-answer training
targets.

Families:
- modpow: a^k mod n (level scales k and n; sympy pow with mod).
- lincong: solve a*x = b (mod n) for the least non-negative x;
  built solvable by construction (draw x first, derive b).
- gcdlin: gcd(a, b) plus Bezout check via extended Euclid.
- order: multiplicative order of a mod p (level 3+; a coprime).
"""

from __future__ import annotations

import math
import random

import sympy as sp

from llmopt.mathgen.problems import Problem


def _int_problem(prompt: str, value: int, kind: str, level: int) -> Problem:
    return Problem(prompt=prompt, answer=str(value), kind=kind,
                   level=level, _expr=sp.Integer(value))


def make_modpow(level: int, seed: int) -> Problem:
    rng = random.Random(f"ntheory-modpow-{level}-{seed}")
    n = rng.randint(5, 30 if level == 1 else 200)
    a = rng.randint(2, n - 1)
    k = rng.randint(2, 10) if level == 1 else rng.randint(10, 500)
    if level >= 3:
        k = rng.randint(10**3, 10**6)
    v = pow(a, k, n)
    return _int_problem(
        f"Compute {a}^{k} mod {n}. Answer with a single integer.",
        v, "modpow", level)


def make_lincong(level: int, seed: int) -> Problem:
    rng = random.Random(f"ntheory-lincong-{level}-{seed}")
    n = rng.randint(5, 30 if level <= 2 else 500)
    a = rng.randint(2, n - 1)
    x = rng.randint(0, n - 1)
    b = (a * x) % n  # solvable by construction
    # canonical answer: least non-negative solution
    least = min(t for t in range(n) if (a * t - b) % n == 0)
    return _int_problem(
        f"Find the least non-negative integer x with "
        f"{a}*x = {b} (mod {n}). Answer with a single integer.",
        least, "lincong", level)


def make_gcdlin(level: int, seed: int) -> Problem:
    rng = random.Random(f"ntheory-gcd-{level}-{seed}")
    hi = 50 if level == 1 else 10**3 if level == 2 else 10**6
    a, b = rng.randint(2, hi), rng.randint(2, hi)
    return _int_problem(
        f"Compute gcd({a}, {b}). Answer with a single integer.",
        math.gcd(a, b), "gcdlin", level)


def make_order(level: int, seed: int) -> Problem:
    rng = random.Random(f"ntheory-order-{level}-{seed}")
    p = int(rng.choice(list(sp.primerange(5, 50 if level <= 2 else 500))))
    a = rng.randint(2, p - 1)
    v = sp.n_order(a, p)
    return _int_problem(
        f"Find the multiplicative order of {a} modulo {p} (the least "
        f"k >= 1 with {a}^k = 1 mod {p}). Answer with a single integer.",
        int(v), "order", level)


MAKERS = {
    "modpow": make_modpow,
    "lincong": make_lincong,
    "gcdlin": make_gcdlin,
    "order": make_order,
}
