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
