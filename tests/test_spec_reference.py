"""Independent reference tests for greedy speculative decoding and the
quantities the E0 census will report. Pure Python/numpy; no model.

Three registers are kept apart in the module under test:
  mathematical identities (checked exhaustively on small cases),
  numerical estimates (float tolerances named per test),
  empirical replay (an estimator over recorded argmax streams, with
  the geometric closed form carried only as a labelled model).
"""
import math
import random

import numpy as np
import pytest

from llmopt.lab import spec_reference as sr


# ---------- 1. greedy speculative decoding is exact for any proposal ----------

def _teacher_from_table(table):
    """Deterministic next-token function over a finite alphabet: a hash of
    the context picks a row of the table."""
    def T(ctx):
        return table[hash(tuple(ctx)) % len(table)]
    return T


@pytest.mark.parametrize("seed", range(20))
def test_greedy_spec_equals_plain_greedy_for_arbitrary_proposals(seed):
    rng = random.Random(seed)
    V = 5
    table = [rng.randrange(V) for _ in range(97)]
    T = _teacher_from_table(table)
    prompt = [rng.randrange(V) for _ in range(3)]
    n = 40
    plain = sr.greedy(T, prompt, n)
    for proposer in (
        lambda ctx, k: [rng.randrange(V) for _ in range(k)],       # random
        lambda ctx, k: [0] * k,                                     # constant
        lambda ctx, k: sr.greedy(T, ctx, k),                        # oracle
    ):
        for k in (1, 2, 3, 7):
            out, stats = sr.greedy_speculative(T, proposer, prompt, n, k)
            assert out == plain
            assert stats["blocks"] >= 1
            assert sum(stats["emitted_per_block"]) == n


def test_greedy_spec_stops_at_eos_like_plain_greedy():
    eos = 3
    table = [1, 2, 3, 0, 1, 2, 0, 3, 1]
    T = _teacher_from_table(table)
    plain = sr.greedy(T, [0], 50, eos=eos)
    out, _ = sr.greedy_speculative(T, lambda c, k: [1] * k, [0], 50, 4, eos=eos)
    assert out == plain
    assert out[-1] == eos or len(out) == 50


def test_oracle_proposer_accepts_every_draft():
    table = list(range(7))
    T = _teacher_from_table(table)
    out, stats = sr.greedy_speculative(T, lambda c, k: sr.greedy(T, c, k), [0], 30, 5)
    assert all(a == 5 for a in stats["accepted_per_block"][:-1])


# ---------- 2. emitted-per-block identities and the break-even inequality ----------

def test_tail_sum_identity_holds_for_any_acceptance_distribution():
    k = 4
    for _ in range(50):
        p = np.random.dirichlet(np.ones(k + 1))     # P(A = j), j = 0..k
        tail = [p[j:].sum() for j in range(k + 1)]  # P(A >= j)
        expected_emitted = sum(j * p[j] for j in range(k + 1)) + 1
        assert math.isclose(sr.emitted_from_tail(tail), expected_emitted, rel_tol=1e-12)


@pytest.mark.parametrize("alpha,k", [(0.0, 3), (0.5, 1), (0.829, 4), (0.999, 8)])
def test_geometric_model_matches_direct_sum(alpha, k):
    direct = sum(alpha ** j for j in range(k + 1))
    assert math.isclose(sr.geometric_emitted(alpha, k), direct, rel_tol=1e-12)
    assert sr.GEOMETRIC_MODEL_LABEL.startswith("MODEL ASSUMPTION")


def test_logged_accept_ratio_underestimates_alpha():
    alpha, k = 0.8, 6
    accepted = sr.geometric_emitted(alpha, k) - 1
    assert accepted / k < alpha


def test_break_even_inequality_and_monotonicity():
    # costs in units of one plain teacher token
    assert sr.speedup(alpha=0.0, k=4, c_draft=0.0, c_verify=1.0) == pytest.approx(1.0)
    s1 = sr.speedup(alpha=0.6, k=4, c_draft=0.05, c_verify=1.0)
    s2 = sr.speedup(alpha=0.8, k=4, c_draft=0.05, c_verify=1.0)
    assert s2 > s1 > 1.0
    assert sr.speedup(alpha=0.8, k=4, c_draft=0.5, c_verify=1.0) < s2
    # when verify is as expensive as k+1 plain tokens there is no gain
    assert sr.speedup(alpha=0.99, k=4, c_draft=0.0, c_verify=5.0) < 1.0 + 1e-9


# ---------- empirical accepted-prefix replay ----------

def test_replay_counts_accepted_prefix_and_charges_every_cost():
    # teacher argmax stream and draft argmax stream at the SAME contexts
    teacher = [1, 1, 1, 1, 1, 1, 1, 1]
    draft = [1, 1, 0, 1, 1, 1, 1, 1]   # one disagreement at position 2
    r = sr.replay_blocks(teacher, draft, k=3)
    # block 1: positions 0..2 -> accept 2, emit 3 (two drafts + teacher token at pos 2)
    # block 2: positions 3..5 -> accept 3, emit 4 (the bonus teacher token is position 6)
    # block 3: position 7 only (stream truncates the draft to 1) -> accept 1, emit 1
    assert r["emitted_per_block"] == [3, 4, 1]
    assert r["accepted_per_block"] == [2, 3, 1]
    assert sum(r["emitted_per_block"]) == len(teacher)
    cost = sr.replay_cost(r, c_draft=0.1, c_verify=lambda k: 1.0, c_commit=0.01, c_rollback=0.02)
    # drafted tokens: 3 + 3 + 1 = 7 ; verifies: 3 ; commits: 3 ; rollbacks: blocks with a rejection = 1
    assert math.isclose(cost["draft"], 0.7) and cost["verify"] == 3.0
    assert math.isclose(cost["commit"], 0.03) and math.isclose(cost["rollback"], 0.02)
    assert math.isclose(cost["tokens_per_unit"], len(teacher) / cost["total"])


def test_replay_is_not_the_geometric_model():
    rng = random.Random(0)
    teacher = [rng.randrange(3) for _ in range(500)]
    draft = [t if rng.random() < 0.8 else (t + 1) % 3 for t in teacher]
    r = sr.replay_blocks(teacher, draft, k=4)
    assert r["model_label"] == "EMPIRICAL REPLAY"
    assert abs(r["mean_emitted"] - sr.geometric_emitted(0.8, 4)) < 0.6  # same ballpark only


# ---------- 3. margins, shift-aligned error, agreement ----------

def test_shift_aligned_sup_error_is_shift_invariant_and_minimal():
    z = np.array([3.0, 1.0, 0.0, -2.0])
    d = np.array([0.5, -0.5, 0.2, 0.1])
    for c in (0.0, 7.0, -3.5):
        e = sr.shift_aligned_sup_error(z, z + d + c)
        assert math.isclose(e, (d.max() - d.min()) / 2, rel_tol=1e-12)
    assert sr.shift_aligned_sup_error(z, z + d) <= np.abs(d).max()


def test_margin_theorem_holds_and_is_tight():
    rng = np.random.default_rng(1)
    for _ in range(200):
        z = rng.normal(size=6)
        i, m = sr.top1_and_margin(z)
        eps = rng.uniform(0, m / 2 * 0.999)
        zp = z + rng.uniform(-eps, eps, size=6)
        assert sr.top1_and_margin(zp)[0] == i
    z = np.array([2.0, 1.0, 0.0])
    i, m = sr.top1_and_margin(z)
    zp = sr.worst_case_perturbation(z, m / 2 * 1.001)
    assert sr.top1_and_margin(zp)[0] != i
    assert sr.shift_aligned_sup_error(z, zp) <= m / 2 * 1.001 + 1e-12


def test_logit_margin_versus_probability_margin():
    z = np.array([2.0, 1.0, 0.0, -1.0])
    p = sr.softmax(z)
    lm = sr.top1_and_margin(z)[1]
    pm = sr.probability_margin(p)
    # p_j <= p_i * exp(-m)  =>  p_i - p_j >= p_i (1 - exp(-m))
    assert pm >= p.max() * (1 - math.exp(-lm)) - 1e-12
    assert sr.top1_and_margin(np.log(p))[1] == pytest.approx(lm)


def test_greedy_acceptance_is_top1_equality_and_sampled_acceptance_is_one_minus_tv():
    rng = np.random.default_rng(2)
    for _ in range(100):
        p = rng.dirichlet(np.ones(5)); q = rng.dirichlet(np.ones(5))
        assert sr.greedy_accept(p, q) == (int(p.argmax()) == int(q.argmax()))
        assert math.isclose(sr.sampled_accept_prob(p, q), 1.0 - sr.total_variation(p, q), rel_tol=1e-12)
        assert math.isclose(sr.sampled_accept_prob(p, q), np.minimum(p, q).sum(), rel_tol=1e-12)


def test_flip_forces_tv_and_kl_lower_bounds():
    rng = np.random.default_rng(3)
    n_flip = 0
    for _ in range(500):
        p = rng.dirichlet(np.ones(4)); q = rng.dirichlet(np.ones(4))
        if int(p.argmax()) != int(q.argmax()):
            n_flip += 1
            mu = sr.probability_margin(p)
            assert sr.total_variation(p, q) >= mu / 2 - 1e-12
            # Pinsker (numerical check of a theorem outside the pinned Mathlib)
            assert sr.kl(p, q) >= 2 * sr.total_variation(p, q) ** 2 - 1e-12
            assert sr.kl(p, q) >= mu ** 2 / 2 - 1e-12
    assert n_flip > 50


# ---------- per-position census statistics used by the E0 driver ----------

def test_position_stats_fields_and_identities():
    rng = np.random.default_rng(4)
    V = 50
    z_t = rng.normal(size=V) * 3
    z_d = z_t + rng.normal(size=V) * 0.3 + 5.0     # shifted copy, small noise
    tok = int(z_t.argmax())
    s = sr.position_stats(z_t, z_d, tok, topk=8)
    assert s["teacher_token"] == tok and s["teacher_argmax"] == tok
    assert s["agree"] == (s["draft_argmax"] == tok)
    assert s["sup_err_shift"] <= s["sup_err_raw"] + 1e-12
    assert s["sup_err_shift"] == pytest.approx(sr.shift_aligned_sup_error(z_t, z_d))
    assert 0.0 <= s["tv"] <= 1.0 and s["sampled_accept"] == pytest.approx(1.0 - s["tv"])
    assert s["kl_teacher_draft"] >= 2 * s["tv"] ** 2 - 1e-9
    assert s["teacher_token_rank_in_draft"] >= 0
    assert len(s["draft_topk_ids"]) == 8 and len(s["draft_topk_logits"]) == 8
    assert s["margin_certified"] == (s["sup_err_shift"] < s["teacher_margin_logit"] / 2)
    if s["margin_certified"]:
        assert s["agree"]


def test_position_stats_margin_certificate_implies_agreement_on_random_cases():
    rng = np.random.default_rng(5)
    n_cert = 0
    for _ in range(300):
        z_t = rng.normal(size=12) * 2
        z_d = z_t + rng.normal(size=12) * rng.uniform(0.01, 1.0)
        s = sr.position_stats(z_t, z_d, int(z_t.argmax()), topk=4)
        if s["margin_certified"]:
            n_cert += 1
            assert s["agree"]
    assert n_cert > 20
