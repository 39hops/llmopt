"""Optional numerical resolution fence on a registered bar (phase 2D).

Law under test (documented in llmopt/lab/metrics.py):
  - no fence, or fence width 0: behaviour identical to the pre-2D
    adjudicator;
  - |observed - bar| < width  -> UNRESOLVED (strictly inside the band);
  - |observed - bar| == width -> resolved by the comparison law
    (below: observed <= bar fires; above: observed >= bar fires);
  - the fence, the observed value and the bar value must be finite
    real numbers (bool refused); otherwise MetricContractError with
    reason metric_value_invalid, never FIRE and never NO-FIRE.
UNRESOLVED from any component (bar, conjunct, refutation predicate)
dominates the bar / refutation outcome.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from llmopt.lab.metrics import (INVALID_VALUE, Metric, MetricContractError,
                                adjudicate)
from llmopt.lab.prereg import (PreregSchemaError, adjudicate_prereg,
                               adjudicate_refutation, validate)

ROOT = Path(__file__).resolve().parents[1]


def M(v):
    return Metric(v, "m", "p", "a")


# ---------------------------------------------------------------- metrics

@pytest.mark.parametrize("direction", ["below", "above"])
@pytest.mark.parametrize("v", [0.0, 4.9, 5.0, 5.1, 100.0, -3.0])
def test_absent_or_zero_fence_matches_legacy(direction, v):
    legacy = adjudicate(M(v), 5.0, direction)
    assert adjudicate(M(v), 5.0, direction, fence=None) == legacy
    assert adjudicate(M(v), 5.0, direction, fence=0.0) == legacy
    assert adjudicate(M(v), 5.0, direction, fence=0) == legacy
    assert legacy in ("FIRE", "NO-FIRE")


@pytest.mark.parametrize("direction", ["below", "above"])
def test_strictly_inside_band_is_unresolved(direction):
    assert adjudicate(M(5.0), 5.0, direction, fence=2.0) == "UNRESOLVED"
    assert adjudicate(M(6.0), 5.0, direction, fence=2.0) == "UNRESOLVED"
    assert adjudicate(M(4.0), 5.0, direction, fence=2.0) == "UNRESOLVED"
    inside_hi = math.nextafter(7.0, 0.0)
    inside_lo = math.nextafter(3.0, 10.0)
    assert adjudicate(M(inside_hi), 5.0, direction, fence=2.0) == "UNRESOLVED"
    assert adjudicate(M(inside_lo), 5.0, direction, fence=2.0) == "UNRESOLVED"


def test_boundary_equality_follows_comparison_law():
    # below: observed <= bar fires. bar - width is below the bar: FIRE.
    assert adjudicate(M(3.0), 5.0, "below", fence=2.0) == "FIRE"
    assert adjudicate(M(7.0), 5.0, "below", fence=2.0) == "NO-FIRE"
    # above: observed >= bar fires. bar + width is above the bar: FIRE.
    assert adjudicate(M(7.0), 5.0, "above", fence=2.0) == "FIRE"
    assert adjudicate(M(3.0), 5.0, "above", fence=2.0) == "NO-FIRE"


def test_immediately_outside_band_resolves():
    out_hi = math.nextafter(7.0, 10.0)
    out_lo = math.nextafter(3.0, 0.0)
    assert adjudicate(M(out_hi), 5.0, "below", fence=2.0) == "NO-FIRE"
    assert adjudicate(M(out_lo), 5.0, "below", fence=2.0) == "FIRE"
    assert adjudicate(M(out_hi), 5.0, "above", fence=2.0) == "FIRE"
    assert adjudicate(M(out_lo), 5.0, "above", fence=2.0) == "NO-FIRE"


def test_fence_on_an_integer_bar_with_integer_observations():
    # the house law: deltas under 5 solves on the 120 gate are unresolved
    assert adjudicate(M(62), 60, "above", fence=5) == "UNRESOLVED"
    assert adjudicate(M(65), 60, "above", fence=5) == "FIRE"
    assert adjudicate(M(55), 60, "above", fence=5) == "NO-FIRE"
    assert adjudicate(M(56), 60, "above", fence=5) == "UNRESOLVED"


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf, -math.inf,
                                 True, "2", [2.0]])
def test_invalid_fence_is_a_contract_error(bad):
    with pytest.raises(MetricContractError) as e:
        adjudicate(M(50.0), 5.0, "above", fence=bad)
    assert e.value.reason == INVALID_VALUE


@pytest.mark.parametrize("direction", ["below", "above"])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True,
                                 "5", None])
def test_invalid_observed_value_never_fires(direction, bad):
    with pytest.raises(MetricContractError) as e:
        adjudicate(M(bad), 5.0, direction)
    assert e.value.reason == INVALID_VALUE
    with pytest.raises(MetricContractError):
        adjudicate(M(bad), 5.0, direction, fence=1.0)


@pytest.mark.parametrize("direction", ["below", "above"])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True,
                                 "5", None])
def test_invalid_bar_value_never_fires(direction, bad):
    with pytest.raises(MetricContractError) as e:
        adjudicate(M(1.0), bad, direction)
    assert e.value.reason == INVALID_VALUE


def test_validation_precedes_comparison_even_for_descriptive_input():
    # the structural refusals still come first; a non-Metric never
    # reaches the numeric path
    with pytest.raises(MetricContractError):
        adjudicate(5.0, 5.0, "above", fence=1.0)


# ----------------------------------------------------------------- prereg

def _doc(fence=None, conj_fence=None, rp_fence=None, prec=False):
    bar = {"id": 1, "name": "B1", "metric": "m", "population": "p",
           "aggregation": "a", "direction": "above", "value": 60,
           "arms": ["x"]}
    if fence is not None:
        bar["fence"] = fence
    if conj_fence is not None:
        bar["conjuncts"] = [{"measurement": "1:twin", "direction": "above",
                             "value": 60, "fence": conj_fence}]
    doc = {"name": "FENCE-T", "results_id": "x", "registered": "2026-10-16",
           "machine": "mac", "arms": {"x": {"description": "arm"}},
           "bars": [bar], "receipts": ["logs/t/r.jsonl"],
           "refuted_if": "prose", "registered_prior": "prose",
           "refuted_if_predicate": {"measurement": "R", "direction": "below",
                                    "value": 10}}
    if rp_fence is not None:
        doc["refuted_if_predicate"]["fence"] = rp_fence
    if prec:
        bar["gate_class"] = "sanity"
        doc["refutation_precedence"] = {"suppressed_unless_bars_fire": [1]}
    return doc


def _obs(v, twin=None, r=None):
    ms = {"1": {"value": v, "metric": "m", "population": "p",
                "aggregation": "a"}}
    if twin is not None:
        ms["1:twin"] = {"value": twin, "metric": "m", "population": "p",
                        "aggregation": "a"}
    if r is not None:
        ms["R"] = {"value": r, "metric": "m", "population": "p",
                   "aggregation": "a"}
    return {"measurement_valid": True, "arms": {"x": {"admissible": True}},
            "measurements": ms}


F5 = {"width": 5, "unit": "solves"}


def test_schema_accepts_fence_object_and_refuses_other_shapes():
    validate(_doc(fence=F5))
    validate(_doc(fence={"width": 0, "unit": "solves"}))
    validate(_doc(conj_fence=F5))
    validate(_doc(rp_fence=F5))
    for bad in (5, {"width": 5}, {"unit": "solves"},
                {"width": -1, "unit": "solves"},
                {"width": math.nan, "unit": "solves"},
                {"width": math.inf, "unit": "solves"},
                {"width": True, "unit": "solves"},
                {"width": "5", "unit": "solves"},
                {"width": 5, "unit": ""}, {"width": 5, "unit": 3},
                {"width": 5, "unit": "solves", "extra": 1}):
        with pytest.raises(PreregSchemaError):
            validate(_doc(fence=bad))
        with pytest.raises(PreregSchemaError):
            validate(_doc(conj_fence=bad))
        with pytest.raises(PreregSchemaError):
            validate(_doc(rp_fence=bad))


def test_bar_inside_fence_is_unresolved_with_units_in_reason():
    out = adjudicate_prereg(_doc(fence=F5), _obs(62))[0]
    assert out.outcome == "UNRESOLVED"
    assert len(out.reasons) == 1
    assert out.reasons[0].startswith("fence:")
    assert "solves" in out.reasons[0] and "62" in out.reasons[0]
    assert "60" in out.reasons[0] and "5" in out.reasons[0]


def test_bar_outside_fence_resolves_both_ways():
    assert adjudicate_prereg(_doc(fence=F5), _obs(65))[0].outcome == "FIRE"
    assert adjudicate_prereg(_doc(fence=F5), _obs(55))[0].outcome == "NO-FIRE"
    assert adjudicate_prereg(_doc(fence=F5), _obs(65))[0].reasons == ()


def test_fenceless_document_unchanged():
    assert adjudicate_prereg(_doc(), _obs(62))[0].outcome == "FIRE"
    assert adjudicate_prereg(_doc(), _obs(60))[0].outcome == "FIRE"
    assert adjudicate_prereg(_doc(), _obs(59))[0].outcome == "NO-FIRE"


def test_conjunct_inside_fence_makes_bar_unresolved():
    # main bar clearly fires (90 >= 60), twin lands inside its band
    out = adjudicate_prereg(_doc(conj_fence=F5), _obs(90, twin=61))[0]
    assert out.outcome == "UNRESOLVED"
    assert any(r.startswith("conjunct 1:twin: fence:") for r in out.reasons)
    # twin clearly fires: bar fires
    assert adjudicate_prereg(_doc(conj_fence=F5),
                             _obs(90, twin=70))[0].outcome == "FIRE"
    # twin clearly misses: bar does not fire
    assert adjudicate_prereg(_doc(conj_fence=F5),
                             _obs(90, twin=40))[0].outcome == "NO-FIRE"


def test_unresolved_dominates_a_missed_conjunct():
    # main bar inside its band, conjunct misses: the bar was not fully
    # adjudicated, so UNRESOLVED (same precedence as a not-run conjunct)
    out = adjudicate_prereg(_doc(fence=F5, conj_fence=F5),
                            _obs(61, twin=40))[0]
    assert out.outcome == "UNRESOLVED"


def test_refutation_predicate_fence():
    doc = _doc(rp_fence={"width": 2, "unit": "points"})
    obs = _obs(90, r=11)
    outs = adjudicate_prereg(doc, obs)
    r = adjudicate_refutation(doc, obs, bar_outcomes=outs)
    assert r.startswith("UNRESOLVED (fence:")
    assert "points" in r
    obs = _obs(90, r=8)
    assert adjudicate_refutation(doc, obs, bar_outcomes=outs) == "REFUTED"
    obs = _obs(90, r=12)
    assert adjudicate_refutation(doc, obs, bar_outcomes=outs) == "NOT-REFUTED"


def test_unresolved_sanity_bar_blocks_refutation():
    doc = _doc(fence=F5, prec=True)
    obs = _obs(62, r=1)   # bar inside band, predicate would REFUTE
    outs = adjudicate_prereg(doc, obs)
    assert outs[0].outcome == "UNRESOLVED"
    r = adjudicate_refutation(doc, obs, bar_outcomes=outs)
    assert r.startswith("UNADJUDICATED (precedence: bar 1 UNRESOLVED")


def test_unresolved_condition_bar_does_not_read_not_refuted():
    doc = _doc(fence=F5, prec=True)
    doc["bars"].append({"id": 2, "name": "C", "metric": "m",
                        "population": "p", "aggregation": "a",
                        "direction": "above", "value": 5, "arms": ["x"],
                        "gate_class": "range",
                        "fence": {"width": 1, "unit": "u"}})
    doc["refutation_precedence"]["applies_only_if_bars_fire"] = [2]
    validate(doc)
    obs = _obs(90, r=1)
    obs["measurements"]["2"] = {"value": 5.5, "metric": "m",
                                "population": "p", "aggregation": "a"}
    outs = adjudicate_prereg(doc, obs)
    assert {o.bar_id: o.outcome for o in outs} == {1: "FIRE", 2: "UNRESOLVED"}
    r = adjudicate_refutation(doc, obs, bar_outcomes=outs)
    assert r.startswith("UNRESOLVED (condition: bar 2 UNRESOLVED")
    obs["measurements"]["2"]["value"] = 3   # condition bar misses
    outs = adjudicate_prereg(doc, obs)
    assert adjudicate_refutation(doc, obs, bar_outcomes=outs).startswith(
        "NOT-REFUTED (condition: bar 2 NO-FIRE")


@pytest.mark.parametrize("bad", [math.nan, math.inf, True, "62", None])
def test_invalid_measurement_value_raises_in_prereg(bad):
    with pytest.raises(MetricContractError) as e:
        adjudicate_prereg(_doc(), _obs(bad))
    assert e.value.reason == INVALID_VALUE
    with pytest.raises(MetricContractError):
        adjudicate_prereg(_doc(fence=F5), _obs(bad))


def test_cli_exits_2_on_fence_unresolved(tmp_path):
    p = tmp_path / "p.json"
    o = tmp_path / "o.json"
    p.write_text(json.dumps(_doc(fence=F5)))
    o.write_text(json.dumps(_obs(62)))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/adjudicate.py"), str(p), str(o)],
        capture_output=True, text=True, cwd=ROOT, check=False)
    assert proc.returncode == 2
    assert "UNRESOLVED [fence:" in proc.stdout
    o.write_text(json.dumps(_obs(90, r=11)))
    p.write_text(json.dumps(_doc(rp_fence={"width": 2, "unit": "pt"})))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/adjudicate.py"), str(p), str(o)],
        capture_output=True, text=True, cwd=ROOT, check=False)
    assert proc.returncode == 2
    assert "REFUTED-IF: UNRESOLVED (fence:" in proc.stdout
