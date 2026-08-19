"""The machine-readable pre-reg schema and its deterministic adjudicator.

The regression half is the REAL STREAM-WDISTILL-0 audit-repair
verdict, re-derived by the adjudicator from the booked numbers:
BAR 1 NO-FIRE (0.3674% v 10%), BAR 2 UNRESOLVED (scalar A serialized
19 bytes over B1 — inadmissible arm, so no admissible contrast
exists), BAR 3 FIRE (+4.54% > 0). If the adjudicator disagrees with
the booked verdict, one of them is wrong and the suite says so.
"""
import copy
import json
import pathlib

import pytest

from llmopt.lab.metrics import MetricContractError
from llmopt.lab.prereg import (PreregSchemaError, adjudicate_prereg,
                               load, validate)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PREREG = ROOT / "docs" / "preregs" / "stream-wdistill-0.json"

# The booked observations, exactly as the audit-repair measured them.
BOOKED_OBS = {
    "measurement_valid": True,
    "arms": {
        "A": {"admissible": False,
              "reason": "serialized 19 bytes over B1"},
        "B": {"admissible": True},
        "C": {"admissible": True},
        "D": {"admissible": True},
        "E": {"admissible": True},
    },
    "measurements": {
        "1": {"value": 0.003674, "metric": "frobenius_rel_gain",
              "population": "experts:0:256",
              "aggregation": "pooled_ratio",
              "provenance": "(D 0.813590 - C 0.810601)/D"},
        "3": {"value": 0.0454, "metric": "frobenius_rel_penalty",
              "population": "experts:0:256",
              "aggregation": "pooled_ratio",
              "provenance": "(E 0.847409 - C 0.810601)/C"},
    },
}


def test_the_checked_in_prereg_validates():
    doc = load(PREREG)
    assert doc["name"] == "STREAM-WDISTILL-0"
    assert "RETROSPECTIVE" in doc["note"]


def test_adjudicator_reproduces_the_booked_verdict():
    """BAR 1 NO-FIRE, BAR 2 UNRESOLVED (arm A), BAR 3 FIRE."""
    out = {o.bar_id: o for o in
           adjudicate_prereg(load(PREREG), BOOKED_OBS)}
    assert out[1].outcome == "NO-FIRE" and not out[1].reasons
    assert out[2].outcome == "UNRESOLVED"
    assert any("arm:A" in r and "19 bytes" in r for r in out[2].reasons)
    assert out[3].outcome == "FIRE" and not out[3].reasons


def test_invalid_measurement_unresolves_every_bar():
    obs = copy.deepcopy(BOOKED_OBS)
    obs["measurement_valid"] = False
    obs["measurement_reason"] = "unasserted revision off /resolve/main"
    for o in adjudicate_prereg(load(PREREG), obs):
        assert o.outcome == "UNRESOLVED"
        assert any("measurement_invalid" in r for r in o.reasons)


def test_missing_measurement_books_not_run():
    obs = copy.deepcopy(BOOKED_OBS)
    del obs["measurements"]["1"]
    out = {o.bar_id: o for o in adjudicate_prereg(load(PREREG), obs)}
    assert out[1].outcome == "UNRESOLVED"
    assert out[1].reasons == ("not-run",)


def test_wrong_population_is_a_contract_error_not_an_outcome():
    """The original incident: a layer bar handed an expert:0 number."""
    obs = copy.deepcopy(BOOKED_OBS)
    obs["measurements"]["1"]["population"] = "expert:0"
    with pytest.raises(MetricContractError):
        adjudicate_prereg(load(PREREG), obs)


def test_schema_refuses_unknown_keys_and_missing_falsifiability():
    doc = json.loads(PREREG.read_text())
    typo = copy.deepcopy(doc)
    typo["bars"][0]["poplation"] = "x"       # typoed fence must be loud
    with pytest.raises(PreregSchemaError):
        validate(typo)
    unfalsifiable = copy.deepcopy(doc)
    unfalsifiable["refuted_if"] = ""
    with pytest.raises(PreregSchemaError):
        validate(unfalsifiable)


def test_bar_naming_undeclared_arm_is_refused():
    doc = json.loads(PREREG.read_text())
    doc["bars"][0]["arms"] = ["C", "Z"]
    with pytest.raises(PreregSchemaError):
        validate(doc)


# ---- adopted 2026-08-16 (external review): the fourth object and
# ---- compound predicates. Both were missing while Amendment 8
# ---- claimed the full law was executable.

def test_contrast_inadmissibility_unresolves_without_blaming_arms():
    """Both arms individually admissible, the RELATION defective."""
    obs = copy.deepcopy(BOOKED_OBS)
    obs["arms"]["A"] = {"admissible": True}
    obs["measurements"]["2"] = {
        "value": 0.05, "metric": "frobenius_rel_gain",
        "population": "experts:0:256", "aggregation": "pooled_ratio"}
    obs["contrasts"] = {"2": {
        "admissible": False,
        "reason": "byte-accounting conventions differ between arms"}}
    out = {o.bar_id: o for o in adjudicate_prereg(load(PREREG), obs)}
    assert out[2].outcome == "UNRESOLVED"
    assert any("contrast: inadmissible" in r for r in out[2].reasons)
    assert not any("arm:" in r for r in out[2].reasons)


def _prereg_0s():
    return load(ROOT / "docs" / "preregs" / "stream-wdistill-0s.json")


def _obs_0s(mean_gain, twins):
    m = {"metric": "operator_rel_gain", "population": "experts:0:256",
         "aggregation": "pooled_ratio"}
    return {
        "measurement_valid": True,
        "arms": {a: {"admissible": True} for a in _prereg_0s()["arms"]},
        "measurements": {
            "1": dict(m, value=0.2), "2": dict(m, value=0.2),
            "3": dict(m, value=mean_gain),
            **{f"3:twin2026081{i+6}": dict(m, value=t)
               for i, t in enumerate(twins)}},
    }


def test_bar3_conjunct_all_three_must_hold():
    """Mean gain clears 5% but one twin individually beats natural:
    the bar must NOT fire (this is the registered AND)."""
    out = {o.bar_id: o for o in adjudicate_prereg(
        _prereg_0s(), _obs_0s(0.06, [0.01, 0.02, -0.001]))}
    assert out[3].outcome == "NO-FIRE"


def test_bar3_fires_when_mean_and_all_twins_hold():
    out = {o.bar_id: o for o in adjudicate_prereg(
        _prereg_0s(), _obs_0s(0.06, [0.01, 0.02, 0.03]))}
    assert out[3].outcome == "FIRE"


def test_bar3_missing_twin_measurement_is_unresolved():
    obs = _obs_0s(0.06, [0.01, 0.02, 0.03])
    del obs["measurements"]["3:twin20260818"]
    out = {o.bar_id: o for o in adjudicate_prereg(_prereg_0s(), obs)}
    assert out[3].outcome == "UNRESOLVED"
    assert any("not-run" in r for r in out[3].reasons)


# ---- the receipt adapter (blind path), adopted 2026-08-16 and
# ---- committed BEFORE the 0S receipt existed.

def _adapter():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "obs0s", ROOT / "scripts" / "obs_from_receipt_0s.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _row(**over):
    op = {"S1-T": 0.797, "S1-U4": 0.657, "S2": 0.377, "W4": 0.328,
          "W8": 0.356, "W32": 0.3735,
          "W32-shuf20260816": 0.3735, "W32-shuf20260817": 0.3734,
          "W32-shuf20260818": 0.3731}
    row = {"smoke": False, "n_experts": 256,
           "arm_within_budget": {a: True for a in list(op) + ["W4-shuf20260816", "W8-shuf20260816"]},
           "codebook_walled_arms": [], "operator_layer": op,
           "code_commit": "abc1234", "revision": "r", "wall_s": 1.0}
    row.update(over)
    return row


def test_adapter_refuses_smoke_receipt():
    with pytest.raises(SystemExit):
        _adapter().observations(_row(smoke=True))


def test_adapter_partial_population_invalidates_measurement():
    obs = _adapter().observations(_row(n_experts=200))
    assert obs["measurement_valid"] is False


def test_adapter_walled_arm_is_inadmissible_whatever_it_printed():
    """The wall hazard: a partially trained stack still produces an
    ordinary-looking number; the adapter must refuse it structurally."""
    obs = _adapter().observations(
        _row(codebook_walled_arms=["W32-shuf20260817/w2"]))
    assert obs["arms"]["W32-shuf"]["admissible"] is False
    out = {o.bar_id: o for o in adjudicate_prereg(_prereg_0s(), obs)}
    assert out[3].outcome == "UNRESOLVED"


def test_adapter_derivations_feed_a_clean_adjudication():
    obs = _adapter().observations(_row())
    out = {o.bar_id: o for o in adjudicate_prereg(_prereg_0s(), obs)}
    # smoke-shaped numbers: scalar S2 within 1% of W32 -> bars miss
    assert out[1].outcome == "NO-FIRE"
    assert out[2].outcome == "FIRE"          # W4 beats S2 by 13% here
    assert out[3].outcome == "NO-FIRE"
    assert "S_best=S2" in obs["measurements"]["1"]["provenance"]


def test_refutation_clause_is_machine_scored():
    """Added 2026-08-17: REFUTED-IF was the one consequential
    sentence still hand-computed. The 0S booked refutation
    (0.035% v 5%) must now reproduce mechanically."""
    from llmopt.lab.prereg import adjudicate_refutation
    obs = _adapter().observations(_row())
    assert adjudicate_refutation(_prereg_0s(), obs) == "REFUTED"
    # counterfactual: a 10% gap must not refute
    obs["measurements"]["refuted:scalar_within"]["value"] = 0.10
    assert adjudicate_refutation(_prereg_0s(), obs) == "NOT-REFUTED"


# --- refutation_precedence (added 2026-08-18, forward-only after
# QWEN-LBAND-1: the alarm->refutation precedence rule must live in
# the registered JSON, not only in adjudicator code) ---

def _prec_doc():
    doc = json.loads(json.dumps(load(
        ROOT / "docs" / "preregs" / "qwen-rk-census-0.json")))
    doc["refutation_precedence"] = {"suppressed_unless_bars_fire": [1]}
    return doc


def test_precedence_validates_and_requires_predicate():
    validate(_prec_doc())
    bad = _prec_doc()
    del bad["refuted_if_predicate"]
    with pytest.raises(PreregSchemaError):
        validate(bad)


def test_precedence_refuses_unknown_bar_and_bad_shape():
    bad = _prec_doc()
    bad["refutation_precedence"]["suppressed_unless_bars_fire"] = [99]
    with pytest.raises(PreregSchemaError):
        validate(bad)
    bad2 = _prec_doc()
    bad2["refutation_precedence"] = {"typo_key": [1]}
    with pytest.raises(PreregSchemaError):
        validate(bad2)


def _prec_obs(rk):
    m = {"metric": "r_k_min_over_layers",
         "population": "positions:corpus+prefixes",
         "aggregation": "mean", "value": rk}
    return {"measurement_valid": True,
            "arms": {"A": {"admissible": True}},
            "measurements": {"1": dict(m),
                             "refuted:r_k_min_over_layers": dict(m)}}


def test_precedence_suppresses_refutation_when_bar_misses():
    from llmopt.lab.prereg import adjudicate_refutation
    doc = _prec_doc()
    obs = _prec_obs(0.11)          # bar 1 (>=0.7) misses; predicate <0.5 hits
    outs = adjudicate_prereg(doc, obs)
    r = adjudicate_refutation(doc, obs, bar_outcomes=outs)
    assert r.startswith("UNADJUDICATED (precedence: bar 1")


def test_precedence_passes_through_when_bar_fires():
    from llmopt.lab.prereg import adjudicate_refutation
    doc = _prec_doc()
    obs = _prec_obs(0.75)          # bar fires; predicate (<0.5) misses
    outs = adjudicate_prereg(doc, obs)
    assert adjudicate_refutation(doc, obs, bar_outcomes=outs) == \
        "NOT-REFUTED"


def test_precedence_without_outcomes_raises():
    from llmopt.lab.prereg import adjudicate_refutation
    doc = _prec_doc()
    with pytest.raises(PreregSchemaError):
        adjudicate_refutation(doc, _prec_obs(0.11))
