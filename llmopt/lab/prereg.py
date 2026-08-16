"""Machine-readable pre-registration: bars a program can adjudicate.

WHY. Every booked bar so far lived as prose. Prose bars produced two
booked incident classes in one thread (STREAM-WDISTILL-0): a bar
registered for a LAYER scored against ONE EXPERT, and an arm whose
serialization was over budget entering a "matched-bytes" contrast
anyway. The adjudication law that repaired it is PROMOTED house text:

    bar_adjudicable = measurement_valid
                      AND every named arm admissible
                      AND the contrast admissible

This module makes that law executable. A pre-reg is a JSON document
(docs/preregs/<name>.json) validated here; adjudication takes the
pre-reg plus an observations document and returns, per bar, exactly
one of FIRE / NO-FIRE / UNRESOLVED — deterministically, with the
reason chain attached. The numeric comparison itself is delegated to
llmopt.lab.metrics.adjudicate, so the population/aggregation
contract (the wrong_metric_population incident) is enforced by the
same code path the incident suite pins.

TWO DIFFERENT FAILURE SHAPES, deliberately kept apart:
  UNRESOLVED   a scientific outcome — the run happened but an arm or
               the measurement lost admissibility (arm A serialized
               19 bytes over B1). Booked, honest, final.
  raise        a contract violation — the measurement handed to a
               bar is not the quantity the bar registered (wrong
               metric name, population, or aggregation). Never
               booked; fix the pipeline.

STRUCTURED RECEIPT REFERENCES. The pre-reg's "receipts" list names
the exact repo-relative paths its run will write. This closes the
known gap from the receipt lock (a receipt cited as a bare filename
in prose is invisible to the citation scraper): a path declared here
is machine-readable by construction and feeds the same lock.

Schema (all unknown top-level or bar keys REFUSED — a typoed fence
must fail loudly, not vanish):
  name            rung name, e.g. "STREAM-WDISTILL-0"
  results_id      stable entry id of the prose pre-reg in
                  docs/results-index.jsonl (the anchor transition)
  registered      YYYY-MM-DD
  machine         "mac" | "wsl" | "w11"
  arms            {arm_name: {"description": str}}
  bars            [{"id", "name", "metric", "population",
                    "aggregation", "direction", "value",
                    "arms": [names], "description"?}]
  receipts        [repo-relative path str] — paths the run WILL write
  refuted_if      prose, required (falsifiability is not optional)
  registered_prior prose, required

Observations document:
  measurement_valid  bool, plus "measurement_reason" when False
  arms               {name: {"admissible": bool, "reason"?: str}}
  measurements       {str(bar_id): {"value", "metric", "population",
                                    "aggregation", "provenance"?}}
                     a bar with no measurement books UNRESOLVED
                     with reason "not-run".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from llmopt.lab.metrics import (Metric, MetricContractError, adjudicate)

DIRECTIONS = ("below", "above")
MACHINES = ("mac", "wsl", "w11")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

PREREG_KEYS = {"name", "results_id", "registered", "machine", "arms",
               "bars", "receipts", "refuted_if", "registered_prior",
               "note"}
# note is OPTIONAL prose; every other top-level key is required. A
# retrospective encoding of an already-booked verdict MUST say so in
# note — a pre-reg written after receipts exist is not a pre-reg.
REQUIRED_KEYS = PREREG_KEYS - {"note"}
BAR_KEYS = {"id", "name", "metric", "population", "aggregation",
            "direction", "value", "arms", "description"}


class PreregSchemaError(ValueError):
    """The pre-reg document itself is malformed. Fix the document."""


@dataclass(frozen=True)
class BarOutcome:
    bar_id: int
    bar_name: str
    outcome: str            # "FIRE" | "NO-FIRE" | "UNRESOLVED"
    reasons: tuple[str, ...]  # empty for FIRE/NO-FIRE


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise PreregSchemaError(msg)


def validate(doc: dict) -> dict:
    """Validate a pre-reg document; returns it unchanged on success."""
    _require(isinstance(doc, dict), "pre-reg must be an object")
    unknown = set(doc) - PREREG_KEYS
    _require(not unknown, f"unknown top-level keys {sorted(unknown)}")
    missing = REQUIRED_KEYS - set(doc)
    _require(not missing, f"missing keys {sorted(missing)}")
    _require(bool(doc["name"]), "empty name")
    _require(bool(_DATE.match(doc["registered"] or "")),
             f"registered {doc['registered']!r} is not YYYY-MM-DD")
    _require(doc["machine"] in MACHINES,
             f"machine {doc['machine']!r} not in {MACHINES}")
    _require(bool(doc["refuted_if"]),
             "refuted_if is empty — an unfalsifiable rung is not a rung")
    _require(bool(doc["registered_prior"]), "registered_prior is empty")
    _require(bool(isinstance(doc["arms"], dict) and doc["arms"]),
             "arms must be a non-empty object")
    _require(isinstance(doc["receipts"], list),
             "receipts must be a list of repo-relative paths")
    for p in doc["receipts"]:
        _require(bool(isinstance(p, str) and p and not p.startswith("/")),
                 f"receipt path {p!r} must be repo-relative")
    _require(bool(isinstance(doc["bars"], list) and doc["bars"]),
             "bars must be a non-empty list")
    seen_ids = set()
    for bar in doc["bars"]:
        unknown = set(bar) - BAR_KEYS
        _require(not unknown,
                 f"bar {bar.get('id')}: unknown keys {sorted(unknown)}")
        missing = (BAR_KEYS - {"description"}) - set(bar)
        _require(not missing,
                 f"bar {bar.get('id')}: missing keys {sorted(missing)}")
        _require(bar["id"] not in seen_ids, f"duplicate bar id {bar['id']}")
        seen_ids.add(bar["id"])
        _require(bar["direction"] in DIRECTIONS,
                 f"bar {bar['id']}: direction {bar['direction']!r}")
        _require(isinstance(bar["value"], (int, float)),
                 f"bar {bar['id']}: value must be a number already on "
                 "the page")
        _require(bool(isinstance(bar["arms"], list) and bar["arms"]),
                 f"bar {bar['id']}: names no arms")
        for a in bar["arms"]:
            _require(a in doc["arms"],
                     f"bar {bar['id']}: arm {a!r} not declared in arms")
    return doc


def load(path: str | Path) -> dict:
    return validate(json.loads(Path(path).read_text()))


def adjudicate_prereg(prereg: dict, obs: dict) -> list[BarOutcome]:
    """Deterministic adjudication of every bar against observations.

    The law, executed in order per bar:
      1. measurement_valid False        -> UNRESOLVED (its reason)
      2. any named arm inadmissible     -> UNRESOLVED (arm reasons)
      3. measurement absent             -> UNRESOLVED "not-run"
      4. measurement contract mismatch  -> raise MetricContractError
      5. metrics.adjudicate             -> FIRE | NO-FIRE
    """
    validate(prereg)
    outcomes = []
    arm_obs = obs.get("arms", {})
    for bar in prereg["bars"]:
        reasons: list[str] = []
        if not obs.get("measurement_valid", False):
            reasons.append("measurement_invalid: "
                           + obs.get("measurement_reason", "unstated"))
        for a in bar["arms"]:
            st = arm_obs.get(a)
            if st is None:
                reasons.append(f"arm:{a}: no admissibility observation")
            elif not st.get("admissible", False):
                reasons.append(f"arm:{a}: inadmissible: "
                               + st.get("reason", "unstated"))
        m = obs.get("measurements", {}).get(str(bar["id"]))
        if m is None and not reasons:
            reasons.append("not-run")
        if reasons:
            outcomes.append(BarOutcome(bar["id"], bar["name"],
                                       "UNRESOLVED", tuple(reasons)))
            continue
        # Contract check: the measurement must BE the registered
        # quantity. A mismatch is a pipeline bug, never an outcome.
        for key in ("metric", "population", "aggregation"):
            if m[key] != bar[key]:
                raise MetricContractError(
                    "metric_identity_mismatch" if key == "metric"
                    else f"metric_{key}_mismatch",
                    f"bar {bar['id']} registered {key}={bar[key]!r}, "
                    f"measurement carries {m[key]!r}")
        metric = Metric(value=float(m["value"]), metric=m["metric"],
                        population=m["population"],
                        aggregation=m["aggregation"],
                        provenance=m.get("provenance", ""))
        verdict = adjudicate(metric, bar_value=float(bar["value"]),
                             direction=bar["direction"],
                             required_population=bar["population"])
        outcomes.append(BarOutcome(bar["id"], bar["name"], verdict, ()))
    return outcomes
