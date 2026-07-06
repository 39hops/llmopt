import sympy as sp

from llmopt.search.derivation import State
from llmopt.search.proposer import build_prompt, make_proposer

x = sp.Symbol("x")


def test_prompt_contains_state_and_numbered_moves():
    p = build_prompt("Derivative(x**2, x)", ["d_power@...", "expand"])
    assert "Derivative(x**2, x)" in p
    assert "1. d_power@..." in p and "2. expand" in p
    assert p.rstrip().endswith("Best move:")


def test_make_proposer_reranks_by_score():
    def score_fn(state_str, labels):
        return [0.1 if "expand" in l else 0.9 for l in labels]

    prop = make_proposer(score_fn)
    s = State(sp.Derivative(x**2, x))
    kids = [("expand", State(x)), ("d_power@D", State(2 * x))]
    ranked = prop(s, kids)
    assert [n for n, _ in ranked] == ["d_power@D", "expand"]
    assert prop(s, []) == []
