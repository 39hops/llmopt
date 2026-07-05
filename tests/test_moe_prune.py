"""Pruning helpers: stats JSON roundtrip, keep-set derivation."""

import json

from llmopt.moe.prune import keep_fraction, keep_sets, stats_from_json
from llmopt.moe.router_stats import RouterStats


def _saved_stats(tmp_path):
    s = RouterStats(n_experts=4)
    s.update(0, [[0, 1], [0, 2]], [[0.9, 0.1], [0.8, 0.2]])
    s.update(1, [[3, 0]], [[0.6, 0.4]])
    p = tmp_path / "stats.json"
    p.write_text(json.dumps({
        "model": "m", "n_experts": 4,
        "math": {"counts": s.counts, "mass": s.mass},
        "general": {"counts": {}, "mass": {}},
    }))
    return s, p


def test_json_roundtrip_restores_int_layer_keys(tmp_path):
    s, p = _saved_stats(tmp_path)
    math, general, n = stats_from_json(p)
    assert n == 4
    assert math.counts == s.counts          # keys back to int
    assert math.mass == s.mass
    assert general.counts == {}


def test_keep_sets_per_layer(tmp_path):
    s, p = _saved_stats(tmp_path)
    math, _, n = stats_from_json(p)
    keep = keep_sets(math, "ever")
    assert keep == {0: {0, 1, 2}, 1: {0, 3}}


def test_keep_fraction():
    assert keep_fraction({0: {0, 1}, 1: {0}}, n_experts=4) == (2 + 1) / 8
    assert keep_fraction({}, n_experts=4) == 1.0
