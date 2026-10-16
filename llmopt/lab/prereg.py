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
                    "arms": [names], "description"?, "fence"?}]
                  fence (optional, forward-only, phase 2D 2026-10-16):
                  {"width": <finite nonnegative number>, "unit": str}
                  — a resolution half-width in the bar's own units.
                  |observed - value| < width books UNRESOLVED with a
                  "fence:" reason; equality resolves by the comparison
                  law; width 0 or no fence is the pre-2D comparison.
                  Conjuncts and refuted_if_predicate accept the same
                  object. Never retrofitted onto booked pre-regs.
  receipts        [repo-relative path str] — paths the run WILL write
  operands        OPTIONAL [{name, provider}] — what the measurement
                  algebra consumes and the exact frozen artifact or
                  committed producer for each; load()/verify_operands
                  REFUSE a provider absent from disk (existence, not
                  sufficiency; forward-only — OPERAND-PROVIDER law)
  refuted_if      prose, required (falsifiability is not optional).
                  MAY additionally be structured: when the document
                  carries "refuted_if_predicate" ({"measurement",
                  "direction", "value"} reading an observations key,
                  same shape as a bar conjunct), the adjudicator
                  scores the refutation clause too — added
                  2026-08-17 (external review: REFUTED-IF was the
                  one consequential sentence still hand-computed).
  registered_prior prose, required

Observations document:
  registration_sha256 OPTIONAL — registration_digest() of the pre-reg
                     the measurements were generated against. When
                     present it must equal the digest of the document
                     being adjudicated (RegistrationMismatchError
                     otherwise). Booked observations predate the field
                     and carry none; schema validity alone never
                     proves a registration was not edited.
  measurement_valid  bool, plus "measurement_reason" when False
  arms               {name: {"admissible": bool, "reason"?: str}}
  measurements       {str(bar_id): {"value", "metric", "population",
                                    "aggregation", "provenance"?}}
                     value must be a finite int or float; bool, str,
                     null, NaN and inf raise MetricContractError
                     (metric_value_invalid) — a pipeline fault, not
                     an outcome.
                     a bar with no measurement books UNRESOLVED
                     with reason "not-run". Conjunct predicates read
                     their own keys (e.g. "3:twin20260817"). Every
                     key must be a registered bar id, conjunct key or
                     the refutation key; arms must all be declared;
                     a duplicated JSON key is refused at load
                     (ObservationsContractError). Other top-level keys
                     are free-form provenance and are ignored.
  contrasts          OPTIONAL {str(bar_id): {"admissible": bool,
                     "reason"?: str}} — the RELATIONAL admissibility
                     the arm gates cannot see (dose mismatch, unpaired
                     seeds, byte-convention mismatch). Absent entry =
                     no registered relational defect. Added 2026-08-16
                     (external review): both arms can be individually
                     admissible while the CONTRAST is not.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from llmopt.lab.metrics import (INVALID_VALUE, UNIT_MISMATCH, Metric,
                                MetricContractError, adjudicate)

DIRECTIONS = ("below", "above")
MACHINES = ("mac", "wsl", "w11")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

PREREG_KEYS = {"name", "results_id", "registered", "machine", "arms",
               "bars", "receipts", "refuted_if", "registered_prior",
               "note", "refuted_if_predicate", "refutation_precedence",
               "operands"}
# note is OPTIONAL prose; every other top-level key is required. A
# retrospective encoding of an already-booked verdict MUST say so in
# note — a pre-reg written after receipts exist is not a pre-reg.
REQUIRED_KEYS = PREREG_KEYS - {"note", "refuted_if_predicate",
                               "refutation_precedence", "operands"}
# operands (optional, forward-only — OPERAND-PROVIDER law, banked
# 2026-08-19): [{"name": <what the measurement algebra consumes>,
#                "provider": <repo-relative frozen artifact path or
#                             committed producer script>}].
# validate() checks shape only; verify_operands()/load() refuse a
# registered operand whose provider path does not exist on disk.
# The field proves PROVIDER EXISTENCE, not semantic sufficiency —
# a present provider can still fail to determine the operand.
# Never retrofitted onto booked pre-regs.
# refutation_precedence (optional): machine-encoded alarm ->
# refutation precedence, adopted forward-only after QWEN-LBAND-1
# (2026-08-18) where the rule lived in adjudicator code and had to
# be disclosed at booking instead of read from the registration.
# Shape: {"suppressed_unless_bars_fire": [<bar id>, ...]} — the
# refutation predicate is adjudicated ONLY when every named bar
# reads FIRE; otherwise the caller receives UNADJUDICATED with the
# blocking bar named. Requires refuted_if_predicate.
BAR_KEYS = {"id", "name", "metric", "population", "aggregation",
            "direction", "value", "arms", "description", "conjuncts",
            "gate_class", "fence", "unit"}
BAR_OPTIONAL = {"description", "conjuncts", "gate_class", "fence", "unit"}
# unit (optional, forward-only, phase 2D): the bar's measurement unit as
# a string. When declared, the measurement must carry an equal "unit"
# and a fence on the bar must use the same unit. Absent, no check.
FENCE_KEYS = {"width", "unit"}
CONJUNCT_KEYS = {"measurement", "direction", "value", "metric",
                 "population", "aggregation", "fence", "unit"}
OBS_RESERVED = {"registration_sha256", "measurement_valid",
                "measurement_reason", "arms", "measurements", "contrasts"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# gate_class (optional): "sanity" (identity/provenance/traversal/
# finiteness/treatment-contract checks — MAY suppress refutation via
# refutation_precedence) or "range" (a registered scientific
# quantity landing outside its predicted band — NEVER suppresses
# refutation; an excursion is a result). Two-gate law, banked at
# MODEL-1 and made executable 2026-08-18: behavioral X/K brackets
# must not be sanity gates, or a headline success can read as an
# instrument alarm (the LBAND lesson, forward-only).
GATE_CLASSES = ("sanity", "range")
# conjuncts (optional): additional predicates that must ALL hold for
# the bar to FIRE, each {"measurement": <obs key>, "direction",
# "value"} sharing the bar's metric contract. Added 2026-08-16
# (external review): BAR 3-class bars ("beats the mean AND all three
# twins") were not expressible as one scalar comparison, so "per bar
# exactly one deterministic outcome" was not yet true of all
# registered house bars.


class PreregSchemaError(ValueError):
    """The pre-reg document itself is malformed. Fix the document."""


class RegistrationMismatchError(PreregSchemaError):
    """The observations were generated against a different registration
    (registration_sha256 does not match). Nothing may be booked."""


class ObservationsContractError(ValueError):
    """The observations document is malformed: duplicate JSON keys, an
    arm the registration never declared, a measurement keyed to no
    registered bar, or a bad registration_sha256 field."""


def _no_duplicate_keys(exc):
    def hook(pairs):
        seen = set()
        for k, _ in pairs:
            if k in seen:
                raise exc(f"duplicate JSON key {k!r}")
            seen.add(k)
        return dict(pairs)
    return hook


def registration_digest(doc: dict) -> str:
    """sha256 over the adjudicable content of a pre-reg: name,
    results_id, registered, machine, declared arm names, every bar as
    written, refuted_if_predicate and refutation_precedence. Prose
    (refuted_if, registered_prior, note, descriptions of arms),
    receipts and operands are outside the digest: they are not read by
    the adjudicator. Canonical JSON (sorted keys, fixed separators), so
    key order and whitespace do not matter.

    An observations document generated against a registration may
    record this value as "registration_sha256"; adjudication then
    refuses any document whose digest differs. The digest proves
    nothing by itself: it is evidence only when the observations file
    carrying it is itself frozen (sha-locked receipt, git history)."""
    core = {"name": doc.get("name"), "results_id": doc.get("results_id"),
            "registered": doc.get("registered"),
            "machine": doc.get("machine"),
            "arms": sorted(doc.get("arms") or {}),
            "bars": doc.get("bars"),
            "refuted_if_predicate": doc.get("refuted_if_predicate"),
            "refutation_precedence": doc.get("refutation_precedence")}
    text = json.dumps(core, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BarOutcome:
    bar_id: int
    bar_name: str
    outcome: str            # "FIRE" | "NO-FIRE" | "UNRESOLVED"
    reasons: tuple[str, ...]  # empty for FIRE/NO-FIRE


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise PreregSchemaError(msg)


def _validate_fence(fence, where: str) -> None:
    """{"width": finite nonnegative real, "unit": non-empty str}."""
    _require(bool(isinstance(fence, dict) and set(fence) == FENCE_KEYS),
             f"{where}: fence must be an object with keys "
             f"{sorted(FENCE_KEYS)}, got {fence!r}")
    w = fence["width"]
    _require(bool(not isinstance(w, bool) and isinstance(w, (int, float))
                  and math.isfinite(w) and w >= 0),
             f"{where}: fence width must be a finite nonnegative "
             f"number, got {w!r}")
    _require(bool(isinstance(fence["unit"], str) and fence["unit"]),
             f"{where}: fence unit must be a non-empty string")


def _fence_width(obj: dict) -> float | None:
    f = obj.get("fence")
    return None if f is None else float(f["width"])


def _fence_reason(observed: float, bar: float, fence: dict) -> str:
    return (f"fence: |observed {observed!r} - bar {bar!r}| = "
            f"{abs(observed - bar)!r} < {fence['width']!r} "
            f"{fence['unit']}")


def _measured(m: dict, where: str) -> float:
    """The measurement's value as a float, refusing anything that is
    not a finite real number already (bool, str, null, NaN, inf)."""
    v = m.get("value")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise MetricContractError(
            INVALID_VALUE, f"{where}: value must be a real number, got "
            f"{type(v).__name__} {v!r}")
    if not math.isfinite(v):
        raise MetricContractError(INVALID_VALUE, f"{where}: value is {v!r}")
    return float(v)


def validate(doc: dict) -> dict:
    """Validate a pre-reg document; returns it unchanged on success."""
    _require(isinstance(doc, dict), "pre-reg must be an object")
    unknown = set(doc) - PREREG_KEYS
    _require(not unknown, f"unknown top-level keys {sorted(unknown)}")
    missing = REQUIRED_KEYS - set(doc)
    _require(not missing, f"missing keys {sorted(missing)}")
    _require(bool(doc["name"]), "empty name")
    _require(bool(isinstance(doc["results_id"], str)
                  and doc["results_id"].strip()),
             "results_id must be a non-empty string naming the prose "
             "pre-registration entry")
    _require(bool(_DATE.match(doc["registered"] or "")),
             f"registered {doc['registered']!r} is not YYYY-MM-DD")
    _require(doc["machine"] in MACHINES,
             f"machine {doc['machine']!r} not in {MACHINES}")
    _require(bool(doc["refuted_if"]),
             "refuted_if is empty — an unfalsifiable rung is not a rung")
    _require(bool(doc["registered_prior"]), "registered_prior is empty")
    rp = doc.get("refuted_if_predicate")
    if rp is not None:
        _require(bool({"measurement", "direction", "value"} <= set(rp)
                      <= {"measurement", "direction", "value", "fence"}),
                 f"refuted_if_predicate keys {sorted(rp)}")
        if "fence" in rp:
            _validate_fence(rp["fence"], "refuted_if_predicate")
        _require(rp["direction"] in DIRECTIONS,
                 "refuted_if_predicate direction")
        _require(isinstance(rp["value"], (int, float)),
                 "refuted_if_predicate value must be numeric")
    prec = doc.get("refutation_precedence")
    if prec is not None:
        _require(rp is not None,
                 "refutation_precedence requires refuted_if_predicate")
        _require(bool({"suppressed_unless_bars_fire"} <= set(prec)
                      <= {"suppressed_unless_bars_fire",
                          "applies_only_if_bars_fire"}),
                 f"refutation_precedence keys {sorted(prec)}")
        cond = prec.get("applies_only_if_bars_fire")
        if cond is not None:
            # EX5 seam: prose refutations of the form "refuted if
            # bar N fires AND <predicate>" had no machine encoding
            # — the predicate alone was scored. These bars are
            # SCIENTIFIC conditions (any gate_class): when one does
            # not FIRE the refutation route reads NOT-REFUTED with
            # the condition named, distinct from sanity suppression
            # (UNADJUDICATED — instrument invalid). Forward-only.
            _require(bool(isinstance(cond, list) and cond
                          and len(set(cond)) == len(cond)),
                     "applies_only_if_bars_fire must be a non-empty "
                     "list of unique bar ids")
            declared = {b.get("id") for b in doc.get("bars", [])
                        if isinstance(b, dict)}
            for i in cond:
                _require(type(i) is int and i in declared,
                         f"applies_only_if_bars_fire names unknown "
                         f"bar {i!r}")
        ids = prec["suppressed_unless_bars_fire"]
        _require(bool(isinstance(ids, list) and ids),
                 "suppressed_unless_bars_fire must be a non-empty list")
        _require(len(set(ids)) == len(ids),
                 "suppressed_unless_bars_fire has duplicate bar ids")
        bar_ids = {b.get("id") for b in doc.get("bars", [])
                   if isinstance(b, dict)}
        by_id = {b.get("id"): b for b in doc.get("bars", [])
                 if isinstance(b, dict)}
        for i in ids:
            _require(type(i) is int,
                     f"precedence bar id {i!r} must be an int "
                     f"(bools refused: True == 1)")
            _require(i in bar_ids,
                     f"refutation_precedence names unknown bar {i!r}")
            _require(by_id[i].get("gate_class") == "sanity",
                     f"refutation_precedence bar {i} must declare "
                     f"gate_class 'sanity' — range bars never "
                     f"suppress refutation")
    _require(bool(isinstance(doc["arms"], dict) and doc["arms"]),
             "arms must be a non-empty object")
    _require(isinstance(doc["receipts"], list),
             "receipts must be a list of repo-relative paths")
    for p in doc["receipts"]:
        _require(bool(isinstance(p, str) and p and not p.startswith("/")),
                 f"receipt path {p!r} must be repo-relative")
    ops = doc.get("operands")
    if ops is not None:
        _require(bool(isinstance(ops, list) and ops),
                 "operands, when present, must be a non-empty list")
        seen_ops = set()
        for op in ops:
            _require(bool(isinstance(op, dict)
                          and set(op) == {"name", "provider"}),
                     f"operand {op!r}: keys must be exactly "
                     f"{{name, provider}}")
            _require(bool(isinstance(op["name"], str) and op["name"]),
                     "operand name must be a non-empty string")
            _require(op["name"] not in seen_ops,
                     f"duplicate operand name {op['name']!r}")
            seen_ops.add(op["name"])
            p = op["provider"]
            _require(bool(isinstance(p, str) and p
                          and not p.startswith("/")),
                     f"operand {op['name']!r}: provider {p!r} must be "
                     f"a repo-relative path")
    _require(bool(isinstance(doc["bars"], list) and doc["bars"]),
             "bars must be a non-empty list")
    seen_ids = set()
    for bar in doc["bars"]:
        unknown = set(bar) - BAR_KEYS
        _require(not unknown,
                 f"bar {bar.get('id')}: unknown keys {sorted(unknown)}")
        missing = (BAR_KEYS - BAR_OPTIONAL) - set(bar)
        _require(not missing,
                 f"bar {bar.get('id')}: missing keys {sorted(missing)}")
        _require(type(bar["id"]) is int,
                 f"bar id {bar['id']!r} must be an int (bools refused)")
        if "unit" in bar:
            _require(bool(isinstance(bar["unit"], str) and bar["unit"]),
                     f"bar {bar['id']}: unit must be a non-empty string")
        if "gate_class" in bar:
            _require(bar["gate_class"] in GATE_CLASSES,
                     f"bar {bar.get('id')}: gate_class "
                     f"{bar['gate_class']!r} not in {GATE_CLASSES}")
        _require(bar["id"] not in seen_ids, f"duplicate bar id {bar['id']}")
        seen_ids.add(bar["id"])
        _require(bar["direction"] in DIRECTIONS,
                 f"bar {bar['id']}: direction {bar['direction']!r}")
        _require(bool(not isinstance(bar["value"], bool)
                      and isinstance(bar["value"], (int, float))
                      and math.isfinite(bar["value"])),
                 f"bar {bar['id']}: value must be a finite number already "
                 "on the page")
        if "fence" in bar:
            _validate_fence(bar["fence"], f"bar {bar['id']}")
            if "unit" in bar:
                _require(bar["fence"]["unit"] == bar["unit"],
                         f"bar {bar['id']}: fence unit "
                         f"{bar['fence']['unit']!r} differs from bar unit "
                         f"{bar['unit']!r}")
        _require(bool(isinstance(bar["arms"], list) and bar["arms"]),
                 f"bar {bar['id']}: names no arms")
        for a in bar["arms"]:
            _require(a in doc["arms"],
                     f"bar {bar['id']}: arm {a!r} not declared in arms")
        for cj in bar.get("conjuncts", []):
            # EX5 seam: a conjunct measuring a DIFFERENT quantity
            # than its bar (nonoverlap margin under a median-gap
            # bar) was forced to carry the bar's metric triple —
            # knowingly false metadata. Conjuncts may now declare
            # their own metric/population/aggregation; absent, the
            # bar's triple is inherited (old docs unchanged).
            _require(bool({"measurement", "direction", "value"}
                          <= set(cj) <= CONJUNCT_KEYS),
                     f"bar {bar['id']}: conjunct keys {sorted(cj)}")
            if "fence" in cj:
                _validate_fence(cj["fence"],
                                f"bar {bar['id']} conjunct "
                                f"{cj.get('measurement')!r}")
            _require(cj["direction"] in DIRECTIONS,
                     f"bar {bar['id']}: conjunct direction")
            _require(isinstance(cj["value"], (int, float)),
                     f"bar {bar['id']}: conjunct value must be numeric")
    return doc


def verify_operands(doc: dict, root: str | Path) -> None:
    """OPERAND-PROVIDER law, v1: every registered operand's provider
    must EXIST under root (a frozen artifact path or a committed
    producer script). Design-time refusal for the class caught twice
    at review time (MODEL-2 r1 arm algebra; CHEAP-READOUT's h with no
    provider). Existence only — a present provider is not proof it
    determines the operand."""
    root = Path(root)
    for op in doc.get("operands") or []:
        _require((root / op["provider"]).exists(),
                 f"operand {op['name']!r}: provider "
                 f"{op['provider']!r} does not exist under {root} — "
                 f"a registered measurement may not name a provider "
                 f"that is not on disk")


def load(path: str | Path, operand_root: str | Path | None = None) -> dict:
    """Load + validate; when the document carries operands, also
    enforce provider existence (root defaults to the repo root
    inferred from the prereg path's docs/preregs location, else
    cwd)."""
    path = Path(path)
    doc = validate(json.loads(
        path.read_text(),
        object_pairs_hook=_no_duplicate_keys(PreregSchemaError)))
    if doc.get("operands"):
        if operand_root is None:
            parent = path.resolve().parent
            operand_root = (parent.parent.parent
                            if parent.name == "preregs"
                            and parent.parent.name == "docs"
                            else Path.cwd())
        verify_operands(doc, operand_root)
    return doc


def load_observations(path: str | Path) -> dict:
    """Read an observations JSON file, refusing duplicate keys (the
    stdlib keeps the last silently) and non-object documents."""
    obs = json.loads(Path(path).read_text(),
                     object_pairs_hook=_no_duplicate_keys(
                         ObservationsContractError))
    if not isinstance(obs, dict):
        raise ObservationsContractError("observations must be an object")
    return obs


def unlocked_receipts(doc: dict, lock: dict) -> list[str]:
    """Receipt paths the pre-reg declares that docs/receipts.lock.json
    does not hold as existing with a sha256. An empty list means every
    declared receipt is frozen evidence; anything listed is a path the
    lock has never seen (or saw as absent)."""
    rows = lock.get("receipts", {})
    out = []
    for rel in doc.get("receipts", []):
        rec = rows.get(rel) or {}
        if not (rec.get("exists") and rec.get("sha256")):
            out.append(rel)
    return out


def check_observations(prereg: dict, obs: dict) -> None:
    """Observations-side contract: shape, registration identity,
    declared arms, registered measurement keys. Raises
    ObservationsContractError or RegistrationMismatchError."""
    if not isinstance(obs, dict):
        raise ObservationsContractError("observations must be an object")
    sha = obs.get("registration_sha256")
    if "registration_sha256" in obs:
        if not (isinstance(sha, str) and _SHA256.match(sha)):
            raise ObservationsContractError(
                f"registration_sha256 must be a 64-hex sha256, got {sha!r}")
        actual = registration_digest(prereg)
        if sha != actual:
            raise RegistrationMismatchError(
                f"observations were generated against registration "
                f"{sha[:12]}, this document digests to {actual[:12]}: "
                f"the registration or the observations changed")
    arms = obs.get("arms", {})
    if not isinstance(arms, dict):
        raise ObservationsContractError("arms must be an object")
    # arms the registration never named are ignored: a driver may
    # record admissibility for every arm it ran, and only the arms a
    # bar names are read (an arm named by a bar but not declared is
    # refused by validate()).
    ms = obs.get("measurements", {})
    if not isinstance(ms, dict):
        raise ObservationsContractError("measurements must be an object")
    registered = {str(b["id"]) for b in prereg["bars"]}
    registered |= {cj["measurement"] for b in prereg["bars"]
                   for cj in b.get("conjuncts", [])}
    rp = prereg.get("refuted_if_predicate")
    if rp is not None:
        registered.add(rp["measurement"])
    stray = set(ms) - registered
    if stray:
        raise ObservationsContractError(
            f"measurements keyed to no registered bar, conjunct or "
            f"refutation predicate: {sorted(stray)}")


def adjudicate_refutation(prereg: dict, obs: dict,
                          bar_outcomes: list | None = None) -> str | None:
    """Score the structured refutation clause, if the pre-reg has one.

    Returns "REFUTED" / "NOT-REFUTED", or None when the document
    carries only prose refuted_if (hand adjudication, disclosed) or
    the named measurement is absent. Contract mismatches raise, as
    everywhere else.

    When the document registers refutation_precedence, the caller
    MUST pass bar_outcomes (the adjudicate_prereg result): the
    predicate is scored only if every named bar reads FIRE;
    otherwise the return is "UNADJUDICATED (precedence: ...)" naming
    the blocking bar. Omitting bar_outcomes on such a document
    raises — precedence registered but not consulted is the exact
    silent-rule failure the field exists to prevent."""
    rp = prereg.get("refuted_if_predicate")
    if rp is None:
        return None
    check_observations(prereg, obs)
    prec = prereg.get("refutation_precedence")
    if prec is not None:
        if bar_outcomes is None:
            raise PreregSchemaError(
                "refutation_precedence registered: adjudicate_refutation"
                " needs bar_outcomes")
        by_id = {o.bar_id: o.outcome for o in bar_outcomes}
        for i in prec["suppressed_unless_bars_fire"]:
            _require(i in by_id, f"precedence bar {i} not adjudicated")
            if by_id[i] != "FIRE":
                return (f"UNADJUDICATED (precedence: bar {i} "
                        f"{by_id[i]})")
        for i in prec.get("applies_only_if_bars_fire", []):
            _require(i in by_id, f"condition bar {i} not adjudicated")
            if by_id[i] == "UNRESOLVED":
                # an unresolved condition leaves the refutation
                # unresolved; only a definite miss reads NOT-REFUTED
                return (f"UNRESOLVED (condition: bar {i} UNRESOLVED)")
            if by_id[i] != "FIRE":
                return (f"NOT-REFUTED (condition: bar {i} "
                        f"{by_id[i]})")
    m = obs.get("measurements", {}).get(rp["measurement"])
    if m is None:
        return None
    value = _measured(m, f"refuted_if_predicate {rp['measurement']}")
    metric = Metric(value, m["metric"], m["population"],
                    m["aggregation"], provenance=m.get("provenance", ""))
    v = adjudicate(metric, bar_value=float(rp["value"]),
                   direction=rp["direction"], fence=_fence_width(rp))
    if v == "UNRESOLVED":
        return ("UNRESOLVED ("
                + _fence_reason(value, float(rp["value"]), rp["fence"])
                + ")")
    return "REFUTED" if v == "FIRE" else "NOT-REFUTED"


def adjudicate_prereg(prereg: dict, obs: dict) -> list[BarOutcome]:
    """Deterministic adjudication of every bar against observations.

    The law, executed in order per bar:
      1. measurement_valid False        -> UNRESOLVED (its reason)
      2. any named arm inadmissible     -> UNRESOLVED (arm reasons)
      3. contrast marked inadmissible   -> UNRESOLVED (its reason)
      4. measurement absent             -> UNRESOLVED "not-run"
      5. measurement contract mismatch  -> raise MetricContractError
         (metric / population / aggregation, or a value that is not
         a finite real number)
      6. metrics.adjudicate on the bar AND every conjunct
                                        -> FIRE only if all hold;
         a component inside its registered fence, or a conjunct with
         no measurement, makes the bar UNRESOLVED (its reason named)
         whatever the other components read.
    """
    validate(prereg)
    check_observations(prereg, obs)
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
        con = obs.get("contrasts", {}).get(str(bar["id"]))
        if con is not None and not con.get("admissible", False):
            reasons.append("contrast: inadmissible: "
                           + con.get("reason", "unstated"))
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
        if "unit" in bar and m.get("unit") != bar["unit"]:
            raise MetricContractError(
                UNIT_MISMATCH,
                f"bar {bar['id']} registered unit {bar['unit']!r}, "
                f"measurement carries {m.get('unit')!r}")
        value = _measured(m, f"bar {bar['id']}")
        metric = Metric(value=value, metric=m["metric"],
                        population=m["population"],
                        aggregation=m["aggregation"],
                        provenance=m.get("provenance", ""))
        verdict = adjudicate(metric, bar_value=float(bar["value"]),
                             direction=bar["direction"],
                             required_population=bar["population"],
                             fence=_fence_width(bar))
        unresolved: list[str] = []
        if verdict == "UNRESOLVED":
            unresolved.append(
                _fence_reason(value, float(bar["value"]), bar["fence"]))
        for cj in bar.get("conjuncts", []):
            cm = obs.get("measurements", {}).get(cj["measurement"])
            if cm is None:
                unresolved.append(f"conjunct {cj['measurement']}: not-run")
                break
            for key in ("metric", "population", "aggregation"):
                if cm[key] != cj.get(key, bar[key]):
                    raise MetricContractError(
                        f"metric_{key}_mismatch",
                        f"bar {bar['id']} conjunct {cj['measurement']}")
            cunit = cj.get("unit", bar.get("unit"))
            if cunit is not None and cm.get("unit") != cunit:
                raise MetricContractError(
                    UNIT_MISMATCH,
                    f"bar {bar['id']} conjunct {cj['measurement']} "
                    f"registered unit {cunit!r}, measurement carries "
                    f"{cm.get('unit')!r}")
            cvalue = _measured(cm, f"bar {bar['id']} conjunct "
                                   f"{cj['measurement']}")
            cv = adjudicate(
                Metric(cvalue, cm["metric"],
                       cm["population"], cm["aggregation"],
                       provenance=cm.get("provenance", "")),
                bar_value=float(cj["value"]),
                direction=cj["direction"],
                required_population=cj.get("population",
                                           bar["population"]),
                fence=_fence_width(cj))
            if cv == "UNRESOLVED":
                unresolved.append(
                    f"conjunct {cj['measurement']}: "
                    + _fence_reason(cvalue, float(cj["value"]),
                                    cj["fence"]))
            elif cv == "NO-FIRE":
                verdict = "NO-FIRE"
        if unresolved:
            outcomes.append(BarOutcome(bar["id"], bar["name"],
                                       "UNRESOLVED", tuple(unresolved)))
        else:
            outcomes.append(BarOutcome(bar["id"], bar["name"],
                                       verdict, ()))
    return outcomes
