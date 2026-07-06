import sympy as sp

from llmopt.search.derivation import State
from llmopt.search.proposer import (
    build_prompt,
    entropy_k,
    make_proposer,
    make_scoring_proposer,
)

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


def test_scoring_proposer_returns_scores():
    def score_fn(state_str, labels):
        return [float(len(l)) for l in labels]

    prop = make_scoring_proposer(score_fn)
    s = State(sp.Derivative(x**2, x))
    kids = [("aa", State(x)), ("dddd", State(2 * x)), ("c", State(x + 1))]
    ranked, scores = prop(s, kids)
    assert [n for n, _ in ranked] == ["dddd", "aa", "c"]
    assert scores == sorted(scores, reverse=True)
    assert len(scores) == 3


def test_entropy_k_peaked_vs_flat():
    policy = entropy_k(k_min=1, k_max=6)
    peaked = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    flat = [1.0] * 6
    assert policy(None, [None] * 6, peaked) == 1
    assert policy(None, [None] * 6, flat) == 6
    mid = policy(None, [None] * 6, [3.0, 2.0, 1.0, 0.0, -1.0, -2.0])
    assert 1 <= mid <= 6


def test_entropy_k_single_child():
    policy = entropy_k()
    assert policy(None, [None], [5.0]) == 1
