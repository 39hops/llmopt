"""Linear algebra + ODE problem kinds (pure sympy)."""

import pytest

sp = pytest.importorskip("sympy")

from llmopt.mathgen.linalg import MAKERS as LINALG_MAKERS
from llmopt.mathgen.odes import MAKERS as ODE_MAKERS
from llmopt.mathgen.problems import make_dataset

KINDS = tuple(LINALG_MAKERS) + tuple(ODE_MAKERS)


def test_all_new_kinds_self_verify():
    for p in make_dataset(70, kinds=KINDS, seed=3):
        assert p.check(p.answer), (p.kind, p.prompt, p.answer)


def test_wrong_answers_rejected():
    for p in make_dataset(35, kinds=KINDS, seed=4):
        assert not p.check("x + 12345"), p.kind
        assert not p.check("[[9, 9], [9, 9]]"), p.kind


def test_eigenvalues_order_free():
    p = LINALG_MAKERS["eigenvalues"](2, 11)
    vals = [v.strip() for v in p.answer.split(",")]
    assert p.check(", ".join(reversed(vals)))  # any order accepted


def test_inverse_verified_not_compared():
    p = LINALG_MAKERS["matrix_inverse"](1, 7)
    # equivalent formatting passes (Matrix(...) wrapper)
    assert p.check(f"Matrix({p.answer})")


def test_planted_eigenvalues_match_sympy():
    p = LINALG_MAKERS["eigenvalues"](2, 9)
    mat = sp.Matrix(eval(p.prompt.split("matrix ")[1].split(". Answer")[0]))
    planted = sorted(sp.Integer(int(v)) for v in p.answer.split(","))
    assert sorted(mat.eigenvals(multiple=True)) == planted


def test_ode_any_equivalent_form_passes():
    p = ODE_MAKERS["ode_linear1"](1, 5)
    assert p.check(sp.sstr(sp.expand(sp.sympify(p.answer))))


def test_ode_solution_actually_solves():
    for s in range(4):
        p = ODE_MAKERS["ode_cc2"](1, s)
        eq, x0, y0, yp0 = p._expr
        y = sp.Function("y")
        sol = sp.sympify(p.answer, locals={"x": sp.Symbol("x")})
        assert bool(sp.checkodesol(eq, sp.Eq(y(sp.Symbol("x")), sol))[0])


def test_ode_wrong_constant_split_rejected():
    # satisfies the ODE and y(0) but not y'(0) — must fail
    p = ODE_MAKERS["ode_cc2"](1, 2)
    eq, x0, y0, yp0 = p._expr
    x = sp.Symbol("x")
    sol = sp.sympify(p.answer, locals={"x": x})
    exps = sorted(sol.atoms(sp.exp), key=sp.sstr)
    if len(exps) == 2:  # swap the two exponentials' coefficients
        c1 = sol.coeff(exps[0])
        c2 = sol.coeff(exps[1])
        if c1 != c2:
            wrong = c2 * exps[0] + c1 * exps[1]
            assert not p.check(sp.sstr(wrong))
