"""Number-theory kinds (integer oracle: recompute, never string-match)
and multivariable kinds (sympy-diff oracle, equivalent forms pass)."""

import sympy as sp

from llmopt.mathgen.multivar import (make_directional, make_mixed_partial,
                                     make_partial)
from llmopt.mathgen.ntheory import (make_gcdlin, make_lincong, make_modpow,
                                    make_order)
from llmopt.mathgen.problems import _resolve_maker

x, y = sp.symbols("x y")


def test_ntheory_valid_and_checkable():
    for make in (make_modpow, make_lincong, make_gcdlin, make_order):
        for level in (1, 2, 3):
            for seed in range(10):
                p = make(level, seed)
                assert p.check(p.answer), p.prompt
                wrong = str(int(p.answer) + 1)
                assert not p.check(wrong), p.prompt


def test_lincong_answer_actually_least():
    for seed in range(20):
        p = make_lincong(2, seed)
        # prompt encodes a, b, n; answer must satisfy and be minimal
        assert p.check(p.answer)


def test_mvar_valid_and_equivalent_forms():
    for make in (make_partial, make_mixed_partial, make_directional):
        for level in (1, 2, 3):
            for seed in range(8):
                p = make(level, seed)
                assert p.check(p.answer), p.prompt
                assert not p.check(p.answer + " + x"), p.prompt
    p = make_partial(2, 0)
    ans = sp.sympify(p.answer, locals={"x": x, "y": y})
    assert p.check(sp.sstr(sp.expand(ans)))


def test_registered_and_deterministic():
    for kind in ("modpow", "lincong", "gcdlin", "order",
                 "partial", "mixed_partial", "directional"):
        mk = _resolve_maker(kind)
        assert [mk(1, s).prompt for s in range(10)] == \
               [mk(1, s).prompt for s in range(10)], kind
