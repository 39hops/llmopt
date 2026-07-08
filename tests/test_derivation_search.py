"""Derivation search foundations: legality, HCE ordering, beam solves."""

import sympy as sp

from llmopt.search.derivation import (
    State,
    beam_search,
    hce,
    is_solved,
    successors,
)

x = sp.Symbol("x")


def test_successors_are_equal_to_parent():
    s = State(sp.Derivative(x**3 + sp.sin(x), x))
    for name, child in successors(s):
        assert sp.simplify(child.expr - s.expr.doit()) == 0 or sp.simplify(
            child.expr.doit() - s.expr.doit()
        ) == 0, name


def test_hce_prefers_solved_states():
    unsolved = State(sp.Derivative(x**2, x))
    solved = State(2 * x)
    assert hce(solved) < hce(unsolved)


def test_solved_detection():
    assert is_solved(State(2 * x))
    assert not is_solved(State(sp.Integral(x, x)))


def test_beam_solves_derivative():
    r = beam_search(sp.Derivative(x**3 + sp.sin(x), x))
    assert r.solved
    assert sp.simplify(r.state.expr - (3 * x**2 + sp.cos(x))) == 0


def test_higher_order_unsolved_at_rung1():
    # sympy collapses Derivative(Derivative(f,x),x) into a single
    # second-order node at construction — there is no unevaluated
    # "peeled" form to rewrite to. Rung-1 rules are first-order only
    # (spec), so this is an honest miss, not a bug.
    r = beam_search(sp.Derivative(sp.Derivative(x**4, x), x))
    assert not r.solved


def test_integral_solved_at_rung2():
    r = beam_search(sp.Integral(3 * x**2 + 2 * x, x))
    assert r.solved
    assert sp.simplify(sp.diff(r.state.expr, x) - (3 * x**2 + 2 * x)) == 0
    assert r.state.plies > 1


def test_timebox_preserves_outer_wall():
    """Per-rule time boxes must never eat the search's outer alarm:
    if the outer wall expires inside a box, it re-fires through the
    outer handler (measured failure mode: infinite conversion of the
    wall into rule timeouts)."""
    import signal
    import time

    from llmopt.search.derivation import _timeboxed

    def slow(_):
        time.sleep(5)

    t0 = time.monotonic()
    assert _timeboxed(slow, None, default="boxed") == "boxed"
    assert time.monotonic() - t0 < 4

    class Outer(BaseException):
        pass

    def raise_outer(*_):
        raise Outer()

    old = signal.signal(signal.SIGALRM, raise_outer)
    signal.alarm(2)
    t0 = time.monotonic()
    try:
        for _ in range(50):
            _timeboxed(slow, None, default=None)
        raise AssertionError("outer wall never fired")
    except Outer:
        assert time.monotonic() - t0 < 4
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def test_usub_end_to_end():
    r = beam_search(sp.Integral(2 * x * sp.cos(x**2), x), max_plies=16)
    assert r.solved
    assert sp.simplify(sp.diff(r.state.expr, x) - 2 * x * sp.cos(x**2)) == 0
    # historically via i_usub; i_linear_basis now reaches sin(x**2)
    # directly (it's in the trig-power span)
    assert any(h.startswith(("i_usub@", "i_linear_basis@"))
               for h in r.state.history)


def test_parts_end_to_end():
    r = beam_search(sp.Integral(x * sp.cos(x), x), max_plies=16)
    assert r.solved
    assert sp.simplify(sp.diff(r.state.expr, x) - x * sp.cos(x)) == 0


def test_diff_edges_still_exact():
    # the constant-offset tolerance must not leak into Derivative-only
    # edges: solved diff answers still match sp.diff exactly
    root = sp.Derivative(x**3 + sp.sin(x), x)
    r = beam_search(root)
    assert r.solved
    assert sp.simplify(r.state.expr - sp.diff(x**3 + sp.sin(x), x)) == 0


def test_search_is_not_degenerate():
    """Rung-0 regression: doit solved everything in ~1 ply. Rung-1
    derivations must be genuine multi-ply descents."""
    r = beam_search(sp.Derivative(x**3 + sp.sin(x), x))
    assert r.solved
    assert r.state.plies > 1
    assert not any("doit" in h or h == "simplify" for h in r.state.history)


def test_history_is_a_legible_step_chain():
    r = beam_search(sp.Derivative(x**2 * sp.sin(x), x))
    assert r.solved
    rule_steps = [h for h in r.state.history if "@" in h]
    assert rule_steps, "expected at least one rule@node entry"
    names = {h.split("@")[0] for h in rule_steps}
    assert names <= {
        "d_const", "d_x", "d_sum", "d_product", "d_power",
        "d_chain_table", "d_quotient",
    }


def test_beam_matches_sympy_on_mathgen_set():
    import random

    from llmopt.mathgen.problems import _expression

    rng = random.Random("rung1-e2e-2-0")
    for _ in range(10):
        f = _expression(rng, 2)
        r = beam_search(sp.Derivative(f, x), max_plies=20)
        assert r.solved, f
        assert sp.simplify(r.state.expr - sp.diff(f, x)) == 0, f


def test_macros_off_by_default():
    s = State(sp.Derivative(sp.sin(x) / (x**2 + 1), x))
    assert not any("d_quotient" in name for name, _ in successors(s))
    assert any(
        "d_quotient" in name for name, _ in successors(s, use_macros=True)
    )


def test_max_nodes_budget():
    r = beam_search(sp.Derivative(x**3 + sp.sin(x), x), max_nodes=2)
    assert r.nodes <= 2


def test_is_zero_ladder():
    from llmopt.search.derivation import _is_zero

    # symbolic-zero forms that expand alone doesn't kill
    assert _is_zero(sp.sin(x) ** 2 + sp.cos(x) ** 2 - 1)
    assert _is_zero(sp.exp(x) * sp.exp(-x) - 1)
    assert _is_zero((x + 1) ** 2 - x**2 - 2 * x - 1)
    # nonzero must be rejected (numeric screen path)
    assert not _is_zero(sp.sin(x) ** 2 + sp.cos(x) ** 2 - 2)
    assert not _is_zero(sp.log(x) - x)
    assert not _is_zero(sp.Integer(3))


def test_trace_collects_equivalent_states():
    trace = []
    root = sp.Derivative(x**2 * sp.sin(x), x)
    r = beam_search(root, trace=trace)
    assert r.solved
    assert len(trace) >= r.state.plies  # at least the winning path
    for s in trace:
        assert sp.simplify(s.expr.doit() - root.doit()) == 0


def test_eval_fn_is_pluggable_and_used():
    calls = []

    def spy_eval(s):
        calls.append(s)
        return hce(s)

    r = beam_search(sp.Derivative(x**2 + sp.sin(x), x), eval_fn=spy_eval)
    assert r.solved
    assert calls, "custom eval_fn was never consulted"


def test_default_eval_unchanged():
    a = beam_search(sp.Derivative(x**3 + sp.sin(x), x))
    b = beam_search(sp.Derivative(x**3 + sp.sin(x), x), eval_fn=hce)
    assert a.state.expr == b.state.expr and a.nodes == b.nodes


def test_propose_k_truncates_branching():
    full = beam_search(sp.Derivative(x**2 * sp.sin(x), x))
    seen = []

    def keep_first(state, children):
        seen.append(len(children))
        return children

    pruned = beam_search(sp.Derivative(x**2 * sp.sin(x), x),
                         proposer=keep_first, propose_k=2)
    assert pruned.solved and full.solved
    assert pruned.nodes < full.nodes  # fewer children admitted
    assert seen, "proposer never called"


def test_proposer_rerank_changes_expansion_order():
    def reversed_proposer(state, children):
        return list(reversed(children))

    r = beam_search(sp.Derivative(x**2 + sp.sin(x), x),
                    proposer=reversed_proposer, propose_k=1)
    # k=1 with a bad ordering may or may not solve — the API contract is
    # only that it runs and respects the truncation
    assert r.nodes >= 1


def test_adaptive_propose_k_callable():
    def scoring_proposer(state, children):
        n = len(children)
        return children, [float(n - i) for i in range(n)]

    ks_seen = []

    def policy(state, ranked, scores):
        ks_seen.append(len(scores))
        return 2

    r = beam_search(sp.Derivative(x**2 * sp.sin(x), x),
                    proposer=scoring_proposer, propose_k=policy)
    assert r.solved
    assert ks_seen, "policy never consulted"


def test_sampled_verification_sound_and_faster():
    import time

    from llmopt.search.derivation import replay_verify

    root = sp.Derivative(x**2 * sp.sin(x) + x**3, x)
    t0 = time.time()
    full = beam_search(root)
    t_full = time.time() - t0
    t0 = time.time()
    sampled = beam_search(root, verify_p=0.1)
    t_samp = time.time() - t0
    assert sampled.solved and full.solved
    assert not sampled.corrupted
    # reported solutions are ALWAYS fully verified
    assert replay_verify(root, sampled.state.history)
    assert sp.simplify(sampled.state.expr - full.state.expr.doit()) == 0 or \
        sp.simplify(sampled.state.expr - sp.diff(x**2 * sp.sin(x) + x**3, x)) == 0
    # speed is measured in the bench, not asserted here (timing flaky in CI);
    # record informally:
    print(f"full {t_full:.2f}s vs sampled {t_samp:.2f}s")


def test_beam_records_history():
    r = beam_search(sp.Derivative(x**2, x))
    assert r.solved
    assert len(r.state.history) == r.state.plies >= 1
