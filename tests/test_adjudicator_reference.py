"""Production adjudicator versus the test-only reference evaluator.

The reference (tests/reference_adjudicator.py) is an independent
implementation in exact rational arithmetic. Disagreements are
investigated, never resolved by editing expectations: every case the
two must agree on is either a fixed table or a seeded random draw
that can be replayed.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from llmopt.lab.metrics import Metric, MetricContractError, adjudicate
from llmopt.lab.prereg import (adjudicate_prereg, adjudicate_refutation,
                               load)
from tests import reference_adjudicator as ref

ROOT = Path(__file__).resolve().parents[1]


def prod_predicate(observed, bar, direction, fence=None):
    try:
        return adjudicate(Metric(observed, "m", "p", "a"), bar, direction,
                          fence=fence)
    except MetricContractError:
        return ref.REFUSE


# ------------------------------------------------------------ fixed table

TABLE = [
    # observed, bar, direction, fence, expected
    (5.0, 5.0, "below", None, "FIRE"),
    (5.0, 5.0, "above", None, "FIRE"),
    (4.999, 5.0, "below", None, "FIRE"),
    (4.999, 5.0, "above", None, "NO-FIRE"),
    (5.001, 5.0, "below", None, "NO-FIRE"),
    (5.001, 5.0, "above", None, "FIRE"),
    (5.0, 5.0, "below", 0, "FIRE"),
    (5.0, 5.0, "above", 1, "UNRESOLVED"),
    (6.0, 5.0, "above", 1, "FIRE"),
    (4.0, 5.0, "above", 1, "NO-FIRE"),
    (4.0, 5.0, "below", 1, "FIRE"),
    (6.0, 5.0, "below", 1, "NO-FIRE"),
    (math.nextafter(6.0, 0), 5.0, "above", 1, "UNRESOLVED"),
    (math.nextafter(6.0, 10), 5.0, "above", 1, "FIRE"),
    (-1, -1, "below", None, "FIRE"),
    (0, 0, "above", 0.0, "FIRE"),
    (1e308, 1e308, "above", None, "FIRE"),
    (-1e308, 1e308, "below", None, "FIRE"),
    (math.nan, 5.0, "above", None, "REFUSE"),
    (math.inf, 5.0, "above", None, "REFUSE"),
    (-math.inf, 5.0, "below", None, "REFUSE"),
    (5.0, math.nan, "below", None, "REFUSE"),
    (True, 1, "above", None, "REFUSE"),
    (1, True, "above", None, "REFUSE"),
    ("5", 5, "above", None, "REFUSE"),
    (None, 5, "above", None, "REFUSE"),
    (5, 5, "sideways", None, "REFUSE"),
    (5, 5, "above", -1, "REFUSE"),
    (5, 5, "above", math.nan, "REFUSE"),
    (5, 5, "above", math.inf, "REFUSE"),
    (5, 5, "above", True, "REFUSE"),
]


@pytest.mark.parametrize("observed,bar,direction,fence,expected", TABLE)
def test_fixed_table(observed, bar, direction, fence, expected):
    assert ref.predicate(observed, bar, direction, fence) == expected
    assert prod_predicate(observed, bar, direction, fence) == expected


# --------------------------------------------------------- randomized

def _draw(rng):
    kind = rng.random()
    if kind < 0.5:
        return rng.uniform(-100, 100)
    if kind < 0.8:
        return rng.randint(-50, 50)
    if kind < 0.9:
        return rng.choice([0.0, -0.0, 1e-300, -1e-300, 1e300, 0.1 + 0.2])
    return rng.choice([math.nan, math.inf, -math.inf, True, None, "7"])


def test_randomized_agreement():
    rng = random.Random(20261016)
    n = agree = 0
    for _ in range(4000):
        o, b = _draw(rng), _draw(rng)
        d = rng.choice(["above", "below", "above", "below", "upward"])
        f = (None if rng.random() < 0.4
             else rng.choice([0, 0.5, 1, 2.5, 10, -1, math.nan, math.inf,
                              True, rng.uniform(0, 5)]))
        r, p = ref.predicate(o, b, d, f), prod_predicate(o, b, d, f)
        n += 1
        agree += r == p
        assert r == p, (o, b, d, f, r, p)
    assert n == agree == 4000


def test_adversarial_boundaries_both_directions():
    rng = random.Random(7)
    for _ in range(500):
        bar = rng.uniform(-1e3, 1e3)
        width = rng.choice([0.0, 1e-9, 0.5, 3.0, 100.0])
        for d in ("above", "below"):
            for o in (bar, bar + width, bar - width,
                      math.nextafter(bar + width, math.inf),
                      math.nextafter(bar + width, -math.inf),
                      math.nextafter(bar - width, math.inf),
                      math.nextafter(bar - width, -math.inf)):
                assert (ref.predicate(o, bar, d, width)
                        == prod_predicate(o, bar, d, width)), (o, bar, d, width)


def test_float_boundary_rounding_is_visible_and_matched():
    # 0.1 + 0.2 is not 0.3 in binary; both implementations must see the
    # same side of the bar for the same float inputs
    o, b = 0.1 + 0.2, 0.3
    assert ref.predicate(o, b, "below") == "NO-FIRE"
    assert prod_predicate(o, b, "below") == "NO-FIRE"
    assert prod_predicate(o, b, "below", fence=1e-15) == "UNRESOLVED"


# ------------------------------------------------- whole-document level

def _bar(i, direction="above", value=10, fence=None, conjuncts=None,
         gate_class=None):
    b = {"id": i, "name": f"B{i}", "metric": "m", "population": "p",
         "aggregation": "a", "direction": direction, "value": value,
         "arms": ["x", "y"]}
    if fence is not None:
        b["fence"] = {"width": fence, "unit": "u"}
    if conjuncts:
        b["conjuncts"] = conjuncts
    if gate_class:
        b["gate_class"] = gate_class
    return b


def _doc(bars, rp=None, prec=None):
    doc = {"name": "REF-T", "results_id": "x", "registered": "2026-10-16",
           "machine": "mac",
           "arms": {"x": {"description": "x"}, "y": {"description": "y"}},
           "bars": bars, "receipts": ["logs/t/r.jsonl"],
           "refuted_if": "prose", "registered_prior": "prose"}
    if rp:
        doc["refuted_if_predicate"] = rp
    if prec:
        doc["refutation_precedence"] = prec
    return doc


def _m(v, metric="m", population="p", aggregation="a"):
    return {"value": v, "metric": metric, "population": population,
            "aggregation": aggregation}


def _compare(doc, obs):
    """Run both; production raising a contract error maps to REFUSE."""
    try:
        outs = adjudicate_prereg(doc, obs)
        prod = {o.bar_id: o.outcome for o in outs}
    except MetricContractError:
        prod, prod_ref = "REFUSE", "REFUSE"
    else:
        try:
            prod_ref = adjudicate_refutation(doc, obs, bar_outcomes=outs)
        except MetricContractError:
            prod_ref = "REFUSE"
    want = {}
    refused = False
    for b in doc["bars"]:
        out, kind = ref.bar_outcome(b, obs)
        if kind == "refuse":
            refused = True
        want[b["id"]] = out
    if refused:
        assert prod == "REFUSE", (prod, want)
        return
    assert prod == want, (prod, want)
    want_ref = ref.refutation(doc, obs, want)
    if want_ref is None:
        assert prod_ref is None
    elif want_ref == "REFUSE":
        assert prod_ref == "REFUSE"
    else:
        assert prod_ref is not None and prod_ref.startswith(want_ref), \
            (prod_ref, want_ref)


def test_randomized_documents_agree():
    rng = random.Random(99)
    for _ in range(600):
        bars = []
        for i in (1, 2):
            cjs = None
            if rng.random() < 0.5:
                cjs = [{"measurement": f"{i}:c", "direction":
                        rng.choice(["above", "below"]), "value":
                        rng.randint(0, 20)}]
                if rng.random() < 0.5:
                    cjs[0]["fence"] = {"width": rng.choice([0, 1, 3]),
                                       "unit": "u"}
            bars.append(_bar(i, rng.choice(["above", "below"]),
                             rng.randint(0, 20),
                             rng.choice([None, 0, 1, 4]), cjs,
                             "sanity" if i == 1 else "range"))
        rp = {"measurement": "R", "direction": rng.choice(["above",
                                                            "below"]),
              "value": rng.randint(0, 20)}
        if rng.random() < 0.5:
            rp["fence"] = {"width": rng.choice([1, 2]), "unit": "u"}
        prec = None
        if rng.random() < 0.6:
            prec = {"suppressed_unless_bars_fire": [1]}
            if rng.random() < 0.5:
                prec["applies_only_if_bars_fire"] = [2]
        doc = _doc(bars, rp, prec)
        ms = {}
        declared = {"1", "2", "R"} | {cj["measurement"] for b in bars
                                      for cj in b.get("conjuncts", [])}
        for key in ("1", "2", "1:c", "2:c", "R"):
            if key not in declared:
                continue      # a stray key is refused by contract
            if rng.random() < 0.85:
                v = rng.choice([rng.randint(0, 20), rng.uniform(0, 20),
                                math.nan, True, "3"]) \
                    if rng.random() < 0.15 else rng.randint(0, 20)
                ms[key] = _m(v)
                if rng.random() < 0.05:
                    ms[key]["population"] = "q"
        obs = {"measurement_valid": rng.random() < 0.9,
               "measurement_reason": "r",
               "arms": {"x": {"admissible": rng.random() < 0.9},
                        "y": {"admissible": rng.random() < 0.9}},
               "measurements": ms}
        if rng.random() < 0.2:
            obs["contrasts"] = {"1": {"admissible": False, "reason": "d"}}
        _compare(doc, obs)


def test_reference_reproduces_every_historical_pin():
    pins = json.loads((ROOT / "tests/fixtures/prereg_adjudication_pins.json")
                      .read_text())
    for pin in pins["pairs"]:
        doc = load(ROOT / pin["prereg"])
        obs = json.loads((ROOT / pin["observations"]).read_text())
        want = {o["bar_id"]: o["outcome"] for o in pin["outcomes"]}
        got = {}
        for b in doc["bars"]:
            out, kind = ref.bar_outcome(b, obs)
            assert kind == "outcome", (pin["prereg"], b["id"])
            got[b["id"]] = out
        assert got == want, pin["prereg"]
        r = ref.refutation(doc, obs, got)
        if pin["refutation"] is None:
            assert r is None, pin["prereg"]
        else:
            assert pin["refutation"].startswith(r), (pin["prereg"], r)
