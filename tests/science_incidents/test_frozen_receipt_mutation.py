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
import sys
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
        if not rec.get("exists") or "sha256" not in rec:
            continue          # prereg-pending: run may still be writing
        p = ROOT / rel
        if not p.is_file():
            drift.append(f"{rel}: VANISHED since it was locked")
        elif _sha(p) != rec["sha256"]:
            drift.append(f"{rel}: CONTENT CHANGED since it was locked")
    assert not drift, (
        "booked receipts mutated — a booked receipt is evidence, and a "
        "new run belongs at a new path:\n  " + "\n  ".join(drift))


# The exact legacy identities absent when the lock was first built
# (older driver logs never force-added). An IDENTITY ratchet, not a
# count (external-review adoption 2026-08-16): with a count alone,
# recovering one legacy file would free a slot a NEW booking could
# silently spend on new uncommitted evidence.
LEGACY_ABSENT = {
    "logs/merge_space1/driver.log", "logs/merge_space2/driver.log",
    "logs/merge_space3/driver.log", "logs/merge_space4/driver.log",
    "logs/merge_space5/driver.log", "logs/microstar/microstar_run.log",
    "logs/pincer/labels_v2.jsonl",
}


def test_cited_but_absent_receipts_are_only_the_legacy_set():
    """No NEW booking may cite evidence it never committed.

    prereg-declared paths whose run has not fired are PENDING, not
    backlog. Everything else absent must be in the pinned legacy set;
    shrinking the set is welcome, substituting into it is not.
    """
    absent = {k for k, v in _locked().items()
              if not v.get("exists") and v.get("source") != "prereg"}
    new = absent - LEGACY_ABSENT
    assert not new, (
        "booking(s) cite receipts that were never committed (and are "
        "not the pinned legacy backlog):\n  " + "\n  ".join(sorted(new)))


def test_lock_covers_every_current_citation():
    """The lock must KNOW about every cited/declared path.

    Coverage hole named by external review 2026-08-16: nothing
    regenerated the lock on new citations, so a new receipt citation
    could exist that the lock never learned about. Fix: this test
    recomputes the citation set and requires lock keys to match.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import gen_receipt_lock as g
    cited = set(g.cited_paths())
    lock_keys = set(_locked())
    missing = cited - lock_keys
    stale = lock_keys - cited
    assert not missing, (
        "cited paths the lock has never seen — run "
        "scripts/gen_receipt_lock.py:\n  " + "\n  ".join(sorted(missing)))
    assert not stale, (
        "lock entries no longer cited anywhere — regenerate the lock:"
        "\n  " + "\n  ".join(sorted(stale)))


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
