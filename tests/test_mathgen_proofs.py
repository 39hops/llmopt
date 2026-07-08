"""Induction proofs v0: two sympy obligations (base + step), true by
construction, checker accepts equivalent step forms and rejects wrong
base, wrong step, and missing lines (spec test plan)."""

import sympy as sp

from llmopt.mathgen.problems import _resolve_maker
from llmopt.mathgen.proofs import make_prove_ind

n = sp.Symbol("n")


def test_reference_proofs_pass():
    for level in (1, 2, 3):
        for seed in range(10):
            p = make_prove_ind(level, seed)
            assert p.check(p.answer), p.prompt


def test_equivalent_step_forms_pass():
    p = make_prove_ind(1, 0)
    base_line, step_line = p.answer.splitlines()
    step = sp.sympify(step_line.split(":", 1)[1], locals={"n": n})
    for form in (sp.expand(step), sp.factor(step)):
        assert p.check(f"{base_line}\nSTEP: {sp.sstr(form)}")


def test_wrong_obligations_rejected():
    p = make_prove_ind(1, 0)
    base_line, step_line = p.answer.splitlines()
    assert not p.check(f"BASE: 999\n{step_line}")          # wrong base
    assert not p.check(f"{base_line}\nSTEP: n**7")          # wrong step
    assert not p.check(base_line)                           # missing step
    assert not p.check("prose about induction, no lines")   # no format


def test_discover_reference_and_wrong_closed_rejected():
    from llmopt.mathgen.proofs import make_prove_discover

    for level in (1, 2, 3):
        for seed in range(8):
            p = make_prove_discover(level, seed)
            assert p.check(p.answer), p.prompt
    p = make_prove_discover(1, 0)
    lines = p.answer.splitlines()
    # a self-consistent but WRONG closed form must fail (base anchor)
    fake = "CLOSED: n**5\nBASE: 1\nSTEP: (n + 1)**5"
    assert not p.check(fake)
    # missing CLOSED line fails
    assert not p.check("\n".join(lines[1:]))


def test_registered_and_deterministic():
    mk = _resolve_maker("prove_ind")
    assert [mk(2, s).prompt for s in range(10)] == \
           [mk(2, s).prompt for s in range(10)]
