"""Historical adjudications replay byte-identically (phase 2D freeze).

tests/fixtures/prereg_adjudication_pins.json was written from the
tree at 80fd095a, before any phase 2D change to llmopt/lab/metrics.py
or llmopt/lab/prereg.py. Each pin pairs a booked pre-registration with
its booked observations document and records the library outcomes,
the refutation string, and the CLI stdout and return code.

A pin failure means one of two things, and the sha pins tell them
apart: the inputs changed (prereg_sha256 / observations_sha256
differ) or the adjudicator changed its answer on unchanged inputs.
Neither is acceptable silently; regenerate only with a booked reason.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PINS = json.loads((ROOT / "tests" / "fixtures"
                   / "prereg_adjudication_pins.json").read_text())
PAIRS = PINS["pairs"]
IDS = [f"{p['prereg'].split('/')[-1]}:{p['observations'].split('/')[-1]}"
       for p in PAIRS]


def test_contract_and_coverage():
    assert PINS["contract"] == "prereg-adjudication-pins-v1"
    assert len(PAIRS) == 22
    assert len({p["prereg"] for p in PAIRS}) == 20
    assert sum(len(p["outcomes"]) for p in PAIRS) == 106


@pytest.mark.parametrize("pin", PAIRS, ids=IDS)
def test_inputs_unchanged(pin):
    for key, sha_key in (("prereg", "prereg_sha256"),
                         ("observations", "observations_sha256")):
        got = hashlib.sha256((ROOT / pin[key]).read_bytes()).hexdigest()
        assert got == pin[sha_key], f"{pin[key]} changed on disk"


@pytest.mark.parametrize("pin", PAIRS, ids=IDS)
def test_library_outcomes_identical(pin):
    from llmopt.lab.prereg import (adjudicate_prereg, adjudicate_refutation,
                                   load)
    prereg = load(ROOT / pin["prereg"])
    obs = json.loads((ROOT / pin["observations"]).read_text())
    outcomes = adjudicate_prereg(prereg, obs)
    got = [{"bar_id": o.bar_id, "bar_name": o.bar_name,
            "outcome": o.outcome, "reasons": list(o.reasons)}
           for o in outcomes]
    assert got == pin["outcomes"]
    assert (adjudicate_refutation(prereg, obs, bar_outcomes=outcomes)
            == pin["refutation"])


@pytest.mark.parametrize("pin", PAIRS, ids=IDS)
def test_cli_output_identical(pin):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "adjudicate.py"),
         str(ROOT / pin["prereg"]), str(ROOT / pin["observations"])],
        capture_output=True, text=True, cwd=ROOT, check=False)
    assert proc.stdout == pin["stdout"]
    assert proc.stderr == pin["stderr"]
    assert proc.returncode == pin["returncode"]


def test_no_historical_pair_was_unresolved():
    """Every booked pair adjudicated cleanly; the UNRESOLVED paths are
    exercised only by synthetic fixtures. Recorded so a future pin
    regeneration that introduces an UNRESOLVED row is visible."""
    assert all(p["returncode"] == 0 for p in PAIRS)
    assert all(o["outcome"] in ("FIRE", "NO-FIRE")
               for p in PAIRS for o in p["outcomes"])
