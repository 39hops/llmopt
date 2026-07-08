"""Induction proofs v0 (spec: 2026-07-07-proofs-rung-design.md).

The reframe that makes this cheap: an induction proof of
sum_{k=1}^{n} a_k = F(n) is exactly two sympy-checkable obligations:
  BASE: a_1 = F(1)
  STEP: F(n) + a_{n+1} simplifies to F(n+1)
Statements are true by construction (draw F first, a_k = F(k)-F(k-1)),
and the generator asserts both obligations at build time — every
emitted problem carries its own proof.

Answer format (trained/parsed; two lines):
  BASE: <value>
  STEP: <expression in n equal to F(n+1)>
check() verifies the base value by substitution and the step by
simplify-difference — any algebraically equivalent step form passes.
v0 scope honestly: the checker verifies the OBLIGATIONS are
discharged; it does not grade proof PROSE (unverifiable, against
house rules) and plain restatement of F(n+1) passes (circularity
guard is v0.5 — requires tracking derivation provenance, which is
the engine tie-in's job).
"""

from __future__ import annotations

import random

import sympy as sp

from llmopt.mathgen.problems import Problem

N = sp.Symbol("n")
K = sp.Symbol("k")


def _closed_form(rng: random.Random, level: int) -> sp.Expr:
    if level == 1:  # polynomial sums (k, k^2, arithmetic-ish)
        c = rng.randint(1, 4)
        return sp.expand(c * rng.choice(
            [N * (N + 1) / 2, N * (N + 1) * (2 * N + 1) / 6, N**2, N * (N + 2)]))
    if level == 2:  # geometric
        r = rng.choice([2, 3])
        c = rng.randint(1, 3)
        return c * (r**N - 1)
    # level 3: mixed poly + geometric
    r = rng.choice([2, 3])
    return sp.expand(rng.randint(1, 3) * (r**N - 1)
                     + rng.randint(1, 3) * N * (N + 1) / 2)


def make_prove_ind(level: int, seed: int) -> Problem:
    rng = random.Random(f"proofs-ind-{level}-{seed}")
    f = _closed_form(rng, level)
    a_k = sp.simplify(f.subs(N, K) - f.subs(N, K - 1))
    base = sp.simplify(f.subs(N, 1))
    step_target = sp.simplify(f.subs(N, N + 1))
    # build-time proof: both obligations must hold by construction
    assert sp.simplify(a_k.subs(K, 1) - base) == 0
    assert sp.simplify(f + a_k.subs(K, N + 1) - step_target) == 0
    answer = f"BASE: {sp.sstr(base)}\nSTEP: {sp.sstr(step_target)}"
    return Problem(
        prompt=(f"Prove by induction that sum_{{k=1}}^{{n}} of "
                f"a_k = {sp.sstr(a_k)} equals {sp.sstr(f)} for all "
                "n >= 1. Answer with two lines: 'BASE: <value of both "
                "sides at n=1>' and 'STEP: <expression that "
                f"F(n) + a_(n+1) simplifies to, in n>'."),
        answer=answer, kind="prove_ind", level=level,
        _expr=(base, step_target))


def check_induction(payload, prediction: str) -> bool:
    """The two-obligation checker (called from Problem.check)."""
    from llmopt.mathgen.problems import parse_answer

    base_truth, step_truth = payload
    base_expr = step_expr = None
    for line in prediction.splitlines():
        s = line.strip()
        if s.upper().startswith("BASE:"):
            base_expr = parse_answer(s[5:])
        elif s.upper().startswith("STEP:"):
            step_expr = parse_answer(s[5:])
    if base_expr is None or step_expr is None:
        return False
    try:
        return (sp.simplify(base_expr - base_truth) == 0
                and sp.simplify(step_expr - step_truth) == 0)
    except Exception:
        return False


def make_prove_discover(level: int, seed: int) -> Problem:
    """Discovery + proof: the closed form is NOT given — the model
    must conjecture F(n) and then discharge both obligations against
    its OWN claim. The proof half is mechanical; the conjecture half
    is where a model can outrun a grammar (Artin's re-assessment of
    where the LLM shines, 2026-07-08). check() is self-consistent:
    the claimed CLOSED must itself satisfy BASE and STEP, and BASE
    must equal the sum's actual value at n=1 (which pins F to the
    true closed form, not just any self-consistent triple)."""
    rng = random.Random(f"proofs-disc-{level}-{seed}")
    f = _closed_form(rng, level)
    a_k = sp.simplify(f.subs(N, K) - f.subs(N, K - 1))
    base = sp.simplify(f.subs(N, 1))
    return Problem(
        prompt=(f"Find a closed form F(n) for sum_{{k=1}}^{{n}} of "
                f"a_k = {sp.sstr(a_k)}, then prove it by induction. "
                "Answer with three lines: 'CLOSED: <F(n)>', "
                "'BASE: <value of F(1)>', 'STEP: <expression that "
                "F(n) + a_(n+1) simplifies to>'."),
        answer=(f"CLOSED: {sp.sstr(f)}\nBASE: {sp.sstr(base)}\n"
                f"STEP: {sp.sstr(sp.simplify(f.subs(N, N + 1)))}"),
        kind="prove_discover", level=level, _expr=(a_k, base))


def check_discovery(payload, prediction: str) -> bool:
    """Self-consistency checker: parse CLOSED/BASE/STEP; the claimed
    F must satisfy its own obligations AND match the true base value
    (a_1), which anchors the conjecture to the actual sum."""
    from llmopt.mathgen.problems import parse_answer

    a_k, base_truth = payload
    closed = base_expr = step_expr = None
    for line in prediction.splitlines():
        s = line.strip()
        if s.upper().startswith("CLOSED:"):
            closed = parse_answer(s[7:])
        elif s.upper().startswith("BASE:"):
            base_expr = parse_answer(s[5:])
        elif s.upper().startswith("STEP:"):
            step_expr = parse_answer(s[5:])
    if closed is None or base_expr is None or step_expr is None:
        return False
    try:
        if sp.simplify(base_expr - base_truth) != 0:
            return False  # base must equal the sum's real value a_1
        if sp.simplify(closed.subs(N, 1) - base_truth) != 0:
            return False  # claimed F anchored at n=1
        # step: F(n) + a_(n+1) must equal the claimed STEP, and the
        # claimed STEP must equal F(n+1) — the induction closes
        lhs = closed + a_k.subs(K, N + 1)
        return (sp.simplify(lhs - step_expr) == 0
                and sp.simplify(step_expr - closed.subs(N, N + 1)) == 0)
    except Exception:
        return False


MAKERS = {"prove_ind": make_prove_ind,
          "prove_discover": make_prove_discover}
