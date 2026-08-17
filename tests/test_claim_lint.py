"""SCIENCE-INCIDENT FIXTURE: prose_overclaims_object.

The claim linter's regression corpus is the REAL overclaim sentences
from STREAM-WDISTILL-0 — each shipped in analysis prose and was
caught only by user-relayed review. The linter must catch every one
now, and a clean corrected sentence must pass (negative control).
"""
import pathlib

import pytest

from llmopt.lab.claimlint import lint_text
from llmopt.lab.prereg import adjudicate_prereg, load

ROOT = pathlib.Path(__file__).resolve().parents[1]
PREREG = ROOT / "docs" / "preregs" / "stream-wdistill-0.json"

OBS = {
    "measurement_valid": True,
    "arms": {"A": {"admissible": False, "reason": "19 bytes over B1"},
             "B": {"admissible": True}, "C": {"admissible": True},
             "D": {"admissible": True}, "E": {"admissible": True}},
    "measurements": {
        "1": {"value": 0.003674, "metric": "frobenius_rel_gain",
              "population": "experts:0:256", "aggregation": "pooled_ratio"},
        "3": {"value": 0.0454, "metric": "frobenius_rel_penalty",
              "population": "experts:0:256", "aggregation": "pooled_ratio"},
    },
}

# The shipped sentences, verbatim in spirit. Every one must be caught.
INCIDENT_SENTENCES = [
    "the codebooks are Lloyd-optimal after stage 8",
    "arms C and D differ only in vector width",
    "E provides independent support for the gauge reading",
    "E is statistically indistinguishable from random",
    "the matrices are near-isotropic at this width",
    "the matched-bytes scalar loses to every low-rank arm",
    "D exploits 3.58x more structure than C",
]


@pytest.mark.parametrize("sentence", INCIDENT_SENTENCES)
def test_every_shipped_overclaim_is_caught(sentence):
    """Each sentence is linted inside a thread-scoped draft, as the
    originals appeared: every shipped overclaim sat in prose that
    named STREAM-WDISTILL (scoped deny rules key on that context)."""
    draft = f"STREAM-WDISTILL analysis note.\n{sentence}"
    assert lint_text(draft), f"linter missed: {sentence!r}"


def test_corrected_wording_passes():
    """NEGATIVE CONTROL: the booked corrected reading lints clean."""
    clean = ("Individual experts are substantially anisotropic "
             "(capture proxy 4.678x the random baseline at r=296); "
             "cross-expert misalignment is the leading candidate and "
             "remains unmeasured pending the equal-rank rider.")
    assert lint_text(clean) == []


def test_prose_contradicting_the_adjudicator_errors():
    outcomes = adjudicate_prereg(load(PREREG), OBS)
    bad = "BAR 1 SHARING-PAYS fires at matched rank."
    f = lint_text(bad, outcomes)
    assert any(x.rule == "prose-contradicts-adjudicator" for x in f)


def test_verdict_sentence_on_unresolved_bar_errors():
    outcomes = adjudicate_prereg(load(PREREG), OBS)
    bad = "BAR 2 no-fire: the scalar reference was not beaten."
    f = lint_text(bad, outcomes)
    assert any(x.rule == "verdict-on-unresolved" for x in f)


def test_contest_wording_with_unresolved_bar_errors():
    outcomes = adjudicate_prereg(load(PREREG), OBS)
    bad = "arm B beats the scalar reference comfortably."
    f = lint_text(bad, outcomes)
    assert any(x.rule == "contest-word-unresolved" for x in f)


def test_correct_verdict_sentences_pass_layer_three():
    outcomes = adjudicate_prereg(load(PREREG), OBS)
    good = ("BAR 1 NO-FIRE: sharing buys 0.3674% against the 10% bar.\n"
            "BAR 3 FIRES: destroying alignment costs 4.54% pooled "
            "Frobenius.\n"
            "BAR 2 stays UNRESOLVED; the low-rank arms have higher "
            "Frobenius error than the inadmissible near-budget scalar "
            "reference (descriptive).")
    assert [x for x in lint_text(good, outcomes)
            if x.severity == "ERROR"] == []


def test_unknown_bar_number_errors():
    outcomes = adjudicate_prereg(load(PREREG), OBS)
    f = lint_text("BAR 7 fires decisively.", outcomes)
    assert any(x.rule == "unknown-bar" for x in f)


def test_deny_registry_provenance_resolves():
    """Every superseded_by must be a real results-index id (external
    review caught two rules pointing at the amendment that INTRODUCED
    the phrase rather than the one that withdrew it)."""
    import json
    ids = {json.loads(l)["id"]
           for l in open(ROOT / "docs" / "results-index.jsonl")}
    reg = json.loads((ROOT / "docs" / "claims.deny.json").read_text())
    bad = [r["pattern"] for r in reg["deny"]
           if r["superseded_by"] not in ids]
    assert not bad, f"unresolvable superseded_by for: {bad}"


def test_scoped_rule_does_not_fire_off_thread():
    """'statistically indistinguishable' in an unrelated properly
    powered study must not ERROR; in a stream-wdistill draft it must."""
    off = ("The two samplers are statistically indistinguishable "
           "at n=2000 paired seeds on the integer battery.")
    assert not [f for f in lint_text(off) if f.rule == "superseded-reading"]
    on = ("In the stream-wdistill capture analysis the twins are "
          "statistically indistinguishable.")
    assert [f for f in lint_text(on) if f.rule == "superseded-reading"]
