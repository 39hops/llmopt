"""SCIENCE-INCIDENT FIXTURE: frozen_receipt_mutation.

ORIGINAL FAILURE (2026-08-16, STREAM-WDISTILL-0). The audit-repair
was launched after manually renaming logs/streamwd/pass12_B1.jsonl,
a path OBSERVATION -EXEC1 cites as its booked evidence. The rename
defeated the driver's refuse-if-exists guard; the booked citation
then resolved to a DIFFERENT run's row, and the cited run_B1.log
did not exist at all. Both auditors caught it; no test did, because
a receipt is just a file and a changed file leaves no trace in the
ledger.

Two halves, per the graduation rule:
  LIVE INVARIANT   every receipt sha in docs/receipts.lock.json
                   still matches the bytes on disk
  HISTORICAL       the exact manoeuvre is reproduced against a
                   temporary lock and must be caught
"""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "docs" / "receipts.lock.json"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _locked():
    if not LOCK.exists():
        pytest.skip("receipt lock absent — run scripts/gen_receipt_lock.py")
    return json.loads(LOCK.read_text())["receipts"]


def test_locked_receipts_are_unchanged():
    """THE INVARIANT: a booked receipt's bytes may never change.

    A new execution belongs at a new path. If this fails, either a
    receipt was overwritten (the incident) or a legitimate change
    needs `gen_receipt_lock.py --accept "reason"` so it lands in a
    reviewable diff.
    """
    drift = []
    for rel, rec in _locked().items():
        if not rec.get("exists"):
            continue
        p = ROOT / rel
        if not p.is_file():
            drift.append(f"{rel}: VANISHED since it was locked")
        elif _sha(p) != rec["sha256"]:
            drift.append(f"{rel}: CONTENT CHANGED since it was locked")
    assert not drift, (
        "booked receipts mutated — a booked receipt is evidence, and a "
        "new run belongs at a new path:\n  " + "\n  ".join(drift))


def test_cited_but_absent_receipts_do_not_grow():
    """Cited paths that do not exist are a pre-existing backlog.

    Seven were already absent when the lock was first built (older
    driver logs never force-added). This ratchets that count so new
    bookings cannot cite evidence they never committed.
    """
    # prereg-declared paths whose run has not fired yet are PENDING,
    # not backlog — the ratchet is for prose citations without evidence
    absent = [k for k, v in _locked().items()
              if not v.get("exists") and v.get("source") != "prereg"]
    assert len(absent) <= 7, (
        f"{len(absent)} cited receipts are missing (ratchet 7). A new "
        "booking must commit the receipt it cites:\n  "
        + "\n  ".join(sorted(absent)))


def test_the_original_manoeuvre_is_caught(tmp_path):
    """HISTORICAL: rename-then-rerun, reproduced end to end.

    Mirrors 2026-08-16 exactly: a booked receipt is renamed aside,
    a second run writes a different row to the freed canonical path,
    and the citation silently resolves to the wrong data.
    """
    logs = tmp_path / "logs" / "streamwd"
    logs.mkdir(parents=True)
    cited = logs / "pass12_B1.jsonl"
    cited.write_text(json.dumps({"run": "exec1", "wall_s": 3394.5}) + "\n")

    lock = {"logs/streamwd/pass12_B1.jsonl":
            {"exists": True, "sha256": _sha(cited), "bytes": cited.stat().st_size}}

    # the manoeuvre: rename the cited receipt aside, rerun into the
    # freed path (this is what defeated the driver's exists-guard)
    cited.rename(logs / "pass12_B1_exec1.jsonl")
    cited.write_text(json.dumps({"run": "repair", "wall_s": 6384.6}) + "\n")

    drift = [rel for rel, rec in lock.items()
             if _sha(tmp_path / rel) != rec["sha256"]]
    assert drift == ["logs/streamwd/pass12_B1.jsonl"], (
        "the lock must catch a rename-then-rerun; if this passes "
        "silently the invariant is not doing its job")


def test_the_invariant_is_not_merely_failing_always(tmp_path):
    """NEGATIVE CONTROL: an untouched receipt must pass.

    An invariant that flagged everything would satisfy the test
    above while being useless.
    """
    p = tmp_path / "receipt.jsonl"
    p.write_text(json.dumps({"run": "only"}) + "\n")
    lock = {"receipt.jsonl": {"exists": True, "sha256": _sha(p)}}
    assert not [r for r, rec in lock.items()
                if _sha(tmp_path / r) != rec["sha256"]]
