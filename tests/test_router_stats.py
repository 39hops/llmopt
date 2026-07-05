"""Router-stats aggregation and keep-set criteria (pure python)."""

import pytest

from llmopt.moe.router_stats import RouterStats, overlap, prune_summary


def _stats(n_experts=8):
    s = RouterStats(n_experts=n_experts)
    # 3 tokens, top-2 routing: expert 0 dominant, 1 and 2 marginal, rest cold
    s.update(
        0,
        [[0, 1], [0, 2], [0, 1]],
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
    )
    return s


def test_ever_keeps_any_selected():
    assert _stats().keep_set(0, "ever") == {0, 1, 2}


def test_mass_drops_marginal_experts():
    # expert 0 carries 2.4 of 3.0 total mass = 80%; a 0.75 threshold
    # should keep expert 0 alone
    assert _stats().keep_set(0, "mass", threshold=0.75) == {0}


def test_mass_high_threshold_keeps_more():
    kept = _stats().keep_set(0, "mass", threshold=0.99)
    assert 0 in kept and len(kept) >= 2


def test_topq_is_a_count_quantile():
    # top 25% of 8 experts = 2 experts, by count: 0 (3 picks), then 1 (2)
    assert _stats().keep_set(0, "topq", threshold=0.25) == {0, 1}


def test_topq_never_keeps_cold_experts():
    # asking for half of 8 experts = 4, but only 3 were ever selected
    assert _stats().keep_set(0, "topq", threshold=0.5) == {0, 1, 2}


def test_unknown_criterion_raises():
    with pytest.raises(ValueError):
        _stats().keep_set(0, "vibes")


def test_weights_default_to_one():
    s = RouterStats(n_experts=4)
    s.update(0, [[0, 1]])
    assert s.mass[0][0] == 1.0


def test_overlap_jaccard():
    assert overlap({0, 1}, {1, 2}) == pytest.approx(1 / 3)
    assert overlap(set(), set()) == 1.0


def test_prune_summary_shape():
    d, g = _stats(), RouterStats(n_experts=8)
    g.update(0, [[3, 4], [3, 5]])
    out = prune_summary(d, g, "ever")
    assert out[0]["domain_kept"] == 3
    assert out[0]["general_kept"] == 3
    assert out[0]["jaccard"] == 0.0  # disjoint keep-sets

def test_utilization_sums_to_one():
    assert sum(_stats().utilization(0)) == pytest.approx(1.0)
