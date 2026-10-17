"""Negative controls: invalid, mismatched or tampered evidence can never
read FIRE (rehabilitation phase 2D, step 3).

Each control starts from a booked pre-registration and its booked
observations (the historical pins) or a small synthetic document,
applies one corruption, and asserts the named outcome: a typed
contract refusal (PreregSchemaError / ObservationsContractError /
RegistrationMismatchError / MetricContractError) or a scientific
UNRESOLVED. The two classes are kept apart on purpose: a refusal means
the pipeline is broken and nothing may be booked; UNRESOLVED is an
honest, bookable outcome.

On registration tampering. Schema validation cannot detect an edited
threshold: an edited document is still a valid document. Detection
needs an identity frozen BEFORE the measurements existed. Two such
identities exist here: (1) git history (every booked pre-reg's last
edit precedes its observations' first commit, measured 2026-10-16),
and (2) registration_digest(), which an observations document may
carry as "registration_sha256" when it is generated, and which the
adjudicator then checks. Historical observations carry no digest and
are adjudicated exactly as before; the control below shows that
without a digest the edit is INVISIBLE, and with one it is refused.
"""
from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from llmopt.lab.metrics import (AGGREGATION_MISMATCH, INVALID_VALUE,
                                METRIC_MISMATCH, POPULATION_MISMATCH,
                                UNIT_MISMATCH, MetricContractError)
from llmopt.lab.prereg import (ObservationsContractError,
                               PreregSchemaError,
                               RegistrationMismatchError,
                               adjudicate_prereg, adjudicate_refutation,
                               load, load_observations,
                               registration_digest, unlocked_receipts,
                               validate)

ROOT = Path(__file__).resolve().parents[1]


def _load_or_skip(path):
    """load() refuses a pre-reg whose operand provider is not on disk;
    some providers are local-only Mac artifacts, so on another host the
    pair is skipped with the reason named rather than failed."""
    from llmopt.lab.prereg import PreregSchemaError, load
    try:
        return load(path)
    except PreregSchemaError as e:
        if "provider" in str(e):
            pytest.skip(f"operand provider absent on this host: {e}")
        raise
PINS = json.loads((ROOT / "tests/fixtures/prereg_adjudication_pins.json")
                  .read_text())["pairs"]
PIN_IDS = [f"{p['prereg'].split('/')[-1]}:{p['observations'].split('/')[-1]}"
           for p in PINS]
# one-bar booked document for the synthetic controls
RK = ROOT / "docs/preregs/qwen-rk-census-0.json"
RK_OBS = ROOT / "logs/qwenrouter/rk_observations.json"


def _pair(pin):
    return (_load_or_skip(ROOT / pin["prereg"]),
            json.loads((ROOT / pin["observations"]).read_text()))


def _fired(doc, obs):
    return [o.bar_id for o in adjudicate_prereg(doc, obs)
            if o.outcome == "FIRE"]


def _outcomes(doc, obs):
    return {o.bar_id: o.outcome for o in adjudicate_prereg(doc, obs)}


# ------------------------------------------------- registration tampering

def test_edited_threshold_is_invisible_to_schema_validation_alone():
    doc, obs = load(RK), json.loads(RK_OBS.read_text())
    assert _outcomes(doc, obs) == {1: "NO-FIRE"}
    tampered = copy.deepcopy(doc)
    tampered["bars"][0]["value"] = -1e9 if doc["bars"][0]["direction"] == "above" else 1e9
    validate(tampered)                          # still a valid document
    assert _outcomes(tampered, obs) == {1: "FIRE"}   # the edit flips it


def test_edited_threshold_is_refused_when_observations_carry_the_digest():
    doc, obs = load(RK), json.loads(RK_OBS.read_text())
    obs = dict(obs, registration_sha256=registration_digest(doc))
    assert _outcomes(doc, obs) == {1: "NO-FIRE"}     # digest matches
    tampered = copy.deepcopy(doc)
    tampered["bars"][0]["value"] = -1e9
    with pytest.raises(RegistrationMismatchError):
        adjudicate_prereg(tampered, obs)
    with pytest.raises(RegistrationMismatchError):
        adjudicate_refutation(tampered, obs, bar_outcomes=[])


@pytest.mark.parametrize("field", ["value", "direction", "metric",
                                   "population", "aggregation", "arms"])
def test_digest_covers_every_adjudicable_bar_field(field):
    doc = load(RK)
    d0 = registration_digest(doc)
    t = copy.deepcopy(doc)
    b = t["bars"][0]
    if field == "value":
        b["value"] = b["value"] + 1
    elif field == "direction":
        b["direction"] = "below" if b["direction"] == "above" else "above"
    elif field == "arms":
        t["arms"]["ghost"] = {"description": "g"}
        b["arms"] = list(b["arms"]) + ["ghost"]
    else:
        b[field] = b[field] + "x"
    assert registration_digest(t) != d0


def test_digest_ignores_prose_only_fields():
    doc = load(RK)
    t = copy.deepcopy(doc)
    t["note"] = "retrospective note"
    t["refuted_if"] = doc["refuted_if"] + " (clarified)"
    assert registration_digest(t) == registration_digest(doc)


def test_digest_is_stable_across_key_order_and_whitespace():
    doc = load(RK)
    reordered = json.loads(json.dumps(doc, sort_keys=True, indent=4))
    assert registration_digest(reordered) == registration_digest(doc)


def test_digest_field_must_be_a_sha256_string():
    doc, obs = load(RK), json.loads(RK_OBS.read_text())
    for bad in (123, "", "abc", None):
        with pytest.raises(ObservationsContractError):
            adjudicate_prereg(doc, dict(obs, registration_sha256=bad))


def test_every_booked_prereg_last_edit_precedes_its_observations():
    """Git history as the frozen identity for the 20 historical pairs:
    the pre-reg's last commit is no later than the observations' first
    commit. Measured 2026-10-16; two pairs share one commit."""
    def dates(path, first):
        out = subprocess.run(
            ["git", "log", "--format=%cI", "--follow", "--", path],
            capture_output=True, text=True, cwd=ROOT, check=False
        ).stdout.split()
        if not out:
            pytest.skip("no git history available")
        return out[-1] if first else out[0]
    same_commit = 0
    for pin in PINS:
        last_edit = dates(pin["prereg"], first=False)
        obs_first = dates(pin["observations"], first=True)
        assert last_edit <= obs_first, pin["prereg"]
        same_commit += last_edit == obs_first
    assert same_commit == 2


# ------------------------------------------------ wrong bar / wrong triple

def test_measurement_keyed_to_an_unregistered_bar_is_refused():
    doc, obs = load(RK), json.loads(RK_OBS.read_text())
    obs = copy.deepcopy(obs)
    obs["measurements"]["7"] = obs["measurements"]["1"]
    with pytest.raises(ObservationsContractError):
        adjudicate_prereg(doc, obs)


def test_measurements_swapped_between_bars_with_different_triples_refuse():
    # ex6-depth-1 has six bars with distinct metric names
    pin = next(p for p in PINS if p["prereg"].endswith("ex6-depth-1.json"))
    doc, obs = _pair(pin)
    ids = [str(b["id"]) for b in doc["bars"]]
    swapped = copy.deepcopy(obs)
    a, b = ids[0], ids[1]
    swapped["measurements"][a], swapped["measurements"][b] = (
        obs["measurements"][b], obs["measurements"][a])
    with pytest.raises(MetricContractError) as e:
        adjudicate_prereg(doc, swapped)
    assert e.value.reason in (METRIC_MISMATCH, POPULATION_MISMATCH,
                              AGGREGATION_MISMATCH)


def test_same_triple_bar_swap_is_a_documented_blind_spot():
    """Two bars registering the same metric triple cannot be told apart
    by the contract check; only the registration digest plus locked
    observations pin WHICH number went to WHICH bar. Recorded, not
    hidden."""
    doc = {"name": "T", "results_id": "x", "registered": "2026-10-16",
           "machine": "mac", "arms": {"x": {"description": "x"}},
           "bars": [{"id": 1, "name": "A", "metric": "m", "population": "p",
                     "aggregation": "a", "direction": "above", "value": 10,
                     "arms": ["x"]},
                    {"id": 2, "name": "B", "metric": "m", "population": "p",
                     "aggregation": "a", "direction": "above", "value": 50,
                     "arms": ["x"]}],
           "receipts": ["logs/t/r"], "refuted_if": "p",
           "registered_prior": "p"}
    obs = {"measurement_valid": True, "arms": {"x": {"admissible": True}},
           "measurements": {"1": {"value": 60, "metric": "m",
                                  "population": "p", "aggregation": "a"},
                            "2": {"value": 20, "metric": "m",
                                  "population": "p", "aggregation": "a"}}}
    assert _outcomes(doc, obs) == {1: "FIRE", 2: "NO-FIRE"}
    swapped = copy.deepcopy(obs)
    swapped["measurements"]["1"], swapped["measurements"]["2"] = (
        obs["measurements"]["2"], obs["measurements"]["1"])
    assert _outcomes(doc, swapped) == {1: "FIRE", 2: "FIRE"}  # undetected


@pytest.mark.parametrize("key,reason", [("metric", METRIC_MISMATCH),
                                        ("population", POPULATION_MISMATCH),
                                        ("aggregation", AGGREGATION_MISMATCH)])
def test_wrong_metric_population_aggregation_is_a_contract_error(key, reason):
    doc, obs = load(RK), json.loads(RK_OBS.read_text())
    obs = copy.deepcopy(obs)
    obs["measurements"]["1"][key] = "other"
    with pytest.raises(MetricContractError) as e:
        adjudicate_prereg(doc, obs)
    assert e.value.reason == reason


def test_registered_unit_must_match_the_measurement_unit():
    doc, obs = load(RK), json.loads(RK_OBS.read_text())
    doc = copy.deepcopy(doc)
    obs = copy.deepcopy(obs)
    doc["bars"][0]["unit"] = "ratio"
    validate(doc)
    with pytest.raises(MetricContractError) as e:   # unit absent
        adjudicate_prereg(doc, obs)
    assert e.value.reason == UNIT_MISMATCH
    obs["measurements"]["1"]["unit"] = "percent"
    with pytest.raises(MetricContractError) as e:
        adjudicate_prereg(doc, obs)
    assert e.value.reason == UNIT_MISMATCH
    obs["measurements"]["1"]["unit"] = "ratio"
    assert _outcomes(doc, obs) == {1: "NO-FIRE"}


def test_fence_unit_must_equal_the_bar_unit_when_both_are_declared():
    doc = copy.deepcopy(load(RK))
    doc["bars"][0]["unit"] = "ratio"
    doc["bars"][0]["fence"] = {"width": 0.01, "unit": "percent"}
    with pytest.raises(PreregSchemaError):
        validate(doc)
    doc["bars"][0]["fence"]["unit"] = "ratio"
    validate(doc)


# --------------------------------------------------------- arms and ids

def test_bar_naming_an_undeclared_arm_is_refused_by_the_schema():
    doc = copy.deepcopy(load(RK))
    doc["bars"][0]["arms"] = list(doc["bars"][0]["arms"]) + ["phantom"]
    with pytest.raises(PreregSchemaError):
        validate(doc)


def test_observation_arms_no_bar_names_cannot_change_an_outcome():
    """Documented behaviour, not a refusal: a driver may record every
    arm it ran. Only arms a bar names are read, so an extra observed
    arm, admissible or not, leaves every outcome unchanged."""
    doc, obs = load(RK), json.loads(RK_OBS.read_text())
    base = _outcomes(doc, obs)
    for adm in (True, False):
        o2 = copy.deepcopy(obs)
        o2["arms"]["phantom"] = {"admissible": adm, "reason": "x"}
        assert _outcomes(doc, o2) == base


def test_duplicate_keys_in_an_observations_file_are_refused(tmp_path):
    text = RK_OBS.read_text()
    # duplicate the measurements object: json.loads would keep the last
    dup = text.rstrip().rstrip("}") + ', "measurements": {}}'
    f = tmp_path / "o.json"
    f.write_text(dup)
    assert "measurements" in json.loads(dup)      # stdlib accepts it
    with pytest.raises(ObservationsContractError):
        load_observations(f)
    assert load_observations(RK_OBS) == json.loads(text)


def test_duplicate_arm_declared_twice_in_a_prereg_file_is_refused(tmp_path):
    text = RK.read_text()
    doc = json.loads(text)
    arm = next(iter(doc["arms"]))
    # inject a second copy of the first arm key
    dup = text.replace('"arms": {', '"arms": {"%s": {"description": "d"}, '
                       % arm, 1)
    f = tmp_path / "p.json"
    f.write_text(dup)
    with pytest.raises(PreregSchemaError):
        load(f)


@pytest.mark.parametrize("rid", ["", None, 7, "   "])
def test_missing_or_invalid_results_id_is_refused(rid):
    doc = copy.deepcopy(load(RK))
    doc["results_id"] = rid
    with pytest.raises(PreregSchemaError):
        validate(doc)
    del doc["results_id"]
    with pytest.raises(PreregSchemaError):
        validate(doc)


def test_duplicate_bar_ids_and_bool_ids_are_refused():
    doc = copy.deepcopy(load(RK))
    doc["bars"].append(copy.deepcopy(doc["bars"][0]))
    with pytest.raises(PreregSchemaError):
        validate(doc)
    doc = copy.deepcopy(load(RK))
    doc["bars"][0]["id"] = True
    with pytest.raises(PreregSchemaError):
        validate(doc)


# ---------------------------------------------------- missing measurement

def test_missing_measurement_is_unresolved_not_run_never_fire():
    doc, obs = load(RK), json.loads(RK_OBS.read_text())
    obs = copy.deepcopy(obs)
    del obs["measurements"]["1"]
    out = adjudicate_prereg(doc, obs)[0]
    assert out.outcome == "UNRESOLVED" and out.reasons == ("not-run",)
    obs["measurements"] = {}
    assert adjudicate_prereg(doc, obs)[0].outcome == "UNRESOLVED"
    del obs["measurements"]
    assert adjudicate_prereg(doc, obs)[0].outcome == "UNRESOLVED"


# ---------------------------------------------------------- receipts

def test_receipt_path_the_lock_has_never_seen_is_reported():
    lock = json.loads((ROOT / "docs/receipts.lock.json").read_text())
    doc = copy.deepcopy(load(RK))
    assert unlocked_receipts(doc, lock) == []       # the booked set is locked
    doc["receipts"].append("logs/nowhere/never_written.jsonl")
    assert unlocked_receipts(doc, lock) == ["logs/nowhere/never_written.jsonl"]
    # a path present in the lock but recorded as absent is also unlocked
    name = doc["receipts"][0]
    lock2 = copy.deepcopy(lock)
    lock2["receipts"][name]["exists"] = False
    assert name in unlocked_receipts(doc, lock2)


def test_absolute_or_empty_receipt_paths_are_refused():
    for bad in ("/etc/passwd", "", 3):
        doc = copy.deepcopy(load(RK))
        doc["receipts"] = [bad]
        with pytest.raises(PreregSchemaError):
            validate(doc)


# ------------------------------------------------------ malformed numbers

@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, False,
                                 "0.5", None, [0.5], {"v": 0.5}])
def test_malformed_measurement_value_is_refused_not_scored(bad):
    doc, obs = load(RK), json.loads(RK_OBS.read_text())
    obs = copy.deepcopy(obs)
    obs["measurements"]["1"]["value"] = bad
    with pytest.raises(MetricContractError) as e:
        adjudicate_prereg(doc, obs)
    assert e.value.reason == INVALID_VALUE


def test_nan_and_infinity_literals_in_a_json_file_cannot_fire(tmp_path):
    """Python's json accepts NaN / Infinity tokens; the adjudicator
    must refuse them at the CLI, never print FIRE, and exit non-zero."""
    base = json.loads(RK_OBS.read_text())
    for literal in ("NaN", "Infinity", "-Infinity"):
        text = json.dumps(base).replace(
            json.dumps(base["measurements"]["1"]["value"]), literal, 1)
        assert literal in text
        f = tmp_path / f"{literal}.json"
        f.write_text(text)
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/adjudicate.py"), str(RK),
             str(f)], capture_output=True, text=True, cwd=ROOT, check=False)
        assert proc.returncode != 0
        assert "FIRE" not in proc.stdout
        assert "metric_value_invalid" in proc.stderr


@pytest.mark.parametrize("bad", [math.nan, "10", True, None])
def test_malformed_bar_value_is_refused_by_the_schema(bad):
    doc = copy.deepcopy(load(RK))
    doc["bars"][0]["value"] = bad
    with pytest.raises(PreregSchemaError):
        validate(doc)


# --------------------------------------------- population / pairing

def test_unpaired_contrast_is_unresolved_without_blaming_arms():
    doc, obs = load(RK), json.loads(RK_OBS.read_text())
    obs = copy.deepcopy(obs)
    obs["contrasts"] = {"1": {"admissible": False,
                              "reason": "unpaired seeds"}}
    out = adjudicate_prereg(doc, obs)[0]
    assert out.outcome == "UNRESOLVED"
    assert out.reasons == ("contrast: inadmissible: unpaired seeds",)


def test_population_change_is_a_contract_error_not_a_verdict():
    doc, obs = load(RK), json.loads(RK_OBS.read_text())
    obs = copy.deepcopy(obs)
    obs["measurements"]["1"]["population"] = "layer:0"
    with pytest.raises(MetricContractError) as e:
        adjudicate_prereg(doc, obs)
    assert e.value.reason == POPULATION_MISMATCH


def test_invalid_measurement_unresolves_every_bar():
    for pin in PINS[:6]:
        doc, obs = _pair(pin)
        obs = dict(obs, measurement_valid=False,
                   measurement_reason="partial population")
        assert set(_outcomes(doc, obs).values()) == {"UNRESOLVED"}


# -------------------------------------------------- mutation sweep

def _mutations(obs, key):
    m = obs["measurements"][key]
    yield "nan", {**m, "value": math.nan}
    yield "inf", {**m, "value": math.inf}
    yield "str", {**m, "value": str(m["value"])}
    yield "bool", {**m, "value": True}
    yield "metric", {**m, "metric": m["metric"] + "_x"}
    yield "population", {**m, "population": m["population"] + "_x"}
    yield "aggregation", {**m, "aggregation": m["aggregation"] + "_x"}
    yield "missing", None


@pytest.mark.parametrize("pin", PINS, ids=PIN_IDS)
def test_mutating_any_fired_measurement_never_keeps_fire(pin):
    """For every historical bar that FIREd, each single corruption of
    the measurement it fired on either raises a typed error or turns
    the bar UNRESOLVED. No corruption leaves FIRE standing."""
    doc, obs = _pair(pin)
    fired = _fired(doc, obs)
    for bar in doc["bars"]:
        if bar["id"] not in fired:
            continue
        key = str(bar["id"])
        for label, mutated in _mutations(obs, key):
            o2 = copy.deepcopy(obs)
            if mutated is None:
                del o2["measurements"][key]
            else:
                o2["measurements"][key] = mutated
            try:
                out = {o.bar_id: o.outcome for o in adjudicate_prereg(doc, o2)}
            except MetricContractError:
                continue
            assert out[bar["id"]] != "FIRE", (pin["prereg"], key, label)
            assert label == "missing" and out[bar["id"]] == "UNRESOLVED"


@pytest.mark.parametrize("pin", PINS, ids=PIN_IDS)
def test_wrong_arm_or_invalid_measurement_never_fires(pin):
    doc, obs = _pair(pin)
    bad_arm = dict(obs, arms={a: {"admissible": False, "reason": "walled"}
                              for a in doc["arms"]})
    assert "FIRE" not in _outcomes(doc, bad_arm).values()
    no_arms = dict(obs, arms={})
    assert "FIRE" not in _outcomes(doc, no_arms).values()
    invalid = dict(obs, measurement_valid=False)
    assert "FIRE" not in _outcomes(doc, invalid).values()


# ----------------------------------------- precedence with unsatisfied bars

def test_refutation_precedence_blocks_when_required_bar_is_not_fire():
    pins = [p for p in PINS if _load_or_skip(ROOT / p["prereg"]).get(
        "refutation_precedence")]
    assert pins, "no booked document registers refutation_precedence"
    for pin in pins:
        doc, obs = _pair(pin)
        req = doc["refutation_precedence"]["suppressed_unless_bars_fire"]
        for bar_id in req:
            o2 = copy.deepcopy(obs)
            o2["arms"] = {a: {"admissible": a not in
                              next(b for b in doc["bars"]
                                   if b["id"] == bar_id)["arms"],
                              "reason": "walled"} for a in doc["arms"]}
            outs = adjudicate_prereg(doc, o2)
            r = adjudicate_refutation(doc, o2, bar_outcomes=outs)
            assert r.startswith("UNADJUDICATED (precedence: bar"), \
                (pin["prereg"], bar_id, r)
            assert "REFUTED" not in r.split("(")[0]
