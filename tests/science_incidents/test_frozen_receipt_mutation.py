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
import os
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
    unverifiable = []
    for rel, rec in _locked().items():
        if not rec.get("exists") or "sha256" not in rec:
            continue          # prereg-pending: run may still be writing
        p = ROOT / rel
        if rec.get("local_only"):
            # machine-local evidence (large streams over the logs
            # doctrine line): verify where present; on the evidence
            # host (LLMOPT_FULL=1) absence IS failure; in a clean
            # clone it is reported, never silently green
            assert rec.get("tracked") is False, \
                f"{rel}: local_only but tracked — reclassify"
            if p.is_file():
                if _sha(p) != rec["sha256"]:
                    drift.append(f"{rel}: CONTENT CHANGED (local)")
            elif os.environ.get("LLMOPT_FULL") == "1":
                drift.append(f"{rel}: local-only evidence ABSENT on "
                             f"the evidence host")
            else:
                unverifiable.append(rel)
            continue
        if not p.is_file():
            drift.append(f"{rel}: VANISHED since it was locked")
        elif _sha(p) != rec["sha256"]:
            drift.append(f"{rel}: CONTENT CHANGED since it was locked")
    if unverifiable:
        print(f"[receipt-lock] {len(unverifiable)} local-only "
              f"receipts unverifiable in this checkout: "
              f"{unverifiable}")
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


def test_brace_citations_expand_to_individual_paths():
    """REGRESSION (2026-08-20, HOMEO booking): a RESULTS receipt list
    written as logs/dir/{a.json, b.jsonl (annotation), ...} cited
    NOTHING under the plain path regex, so booked receipts sat in the
    lock as pending with no sha. Both brace forms must expand."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import gen_receipt_lock as g
    # full-filename members, prose-wrapped, with annotations
    text = ("Receipts: logs/qwenhomeo/{homeo_rows.jsonl,\n"
            "homeo_observations.json (producer,\n"
            "non-authoritative), traj_RL_i0.json,\n"
            "remote_sha256.txt}, plus logs/x/plain.json")
    got = set(g.expand_braces(text))
    assert got == {"logs/qwenhomeo/homeo_rows.jsonl",
                   "logs/qwenhomeo/homeo_observations.json",
                   "logs/qwenhomeo/traj_RL_i0.json",
                   "logs/qwenhomeo/remote_sha256.txt"}
    # mid-filename affix form; multi-word prose members are ignored,
    # never guessed into fabricated paths
    text2 = ("logs/qwenattrib/compose_{BLe,BLm,BLl}.json and "
             "logs/qwenmodel1/score_{same six}.json")
    assert set(g.expand_braces(text2)) == {
        "logs/qwenattrib/compose_BLe.json",
        "logs/qwenattrib/compose_BLm.json",
        "logs/qwenattrib/compose_BLl.json"}


# prereg-declared receipts that exist but were never cited in a
# scanner-visible form by their booking. PINNED legacy set: shrinking
# it (by citing the path in an amendment, or a new booking) is
# welcome; a NEW booking adding to it means its receipt citations are
# invisible to the lock — cite the full paths in RESULTS instead.
LEGACY_PRESENT_PENDING = {
    "logs/qwencycle/impulse_rows.jsonl",
    "logs/qwencycle/impulse_summary.json",
    "logs/qweneffort2/summary_tower_BLe.json",
    "logs/qwenloopstate1/headswap_observations.json",
    "logs/qwenmodel1/score_BLe.json",
    "logs/qwenmodel1/score_BLl.json",
    "logs/qwenmodel1/score_BLm.json",
    "logs/qwenmodel1/score_FLe.json",
    "logs/qwenmodel1/score_FLl.json",
    "logs/qwenmodel1/score_FLm.json",
    "logs/qwenmodel1/tree_observations.json",
    "logs/streamwd/run_B1.log",
    "logs/streamwd/run_B1_repair.log",
}


def test_present_but_pending_receipts_are_only_the_legacy_set():
    """RATCHET (2026-08-20): a receipt that EXISTS and stays
    sha-less in the lock is evidence the record cannot freeze. New
    bookings must cite receipt paths the scanner resolves (plain or
    brace form) so their shas lock at booking time."""
    pending = {k for k, v in _locked().items()
               if v.get("exists") and v.get("pending")}
    new = pending - LEGACY_PRESENT_PENDING
    assert not new, (
        "booked-and-present receipts with no locked sha (cite the "
        "full path in RESULTS so the lock can freeze it):\n  "
        + "\n  ".join(sorted(new)))


def test_brace_annotations_do_not_fabricate_paths():
    """NEGATIVE fixture: filename-like tokens inside parenthetical
    annotations must not become cited paths."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import gen_receipt_lock as g
    text = ("logs/d/{a.json (compare rows.jsonl, see spare.txt), "
            "b.json}")
    assert set(g.expand_braces(text)) == {"logs/d/a.json",
                                          "logs/d/b.json"}


# booked preregs whose declared receipts predate scanner-visible
# citation and therefore hold no locked sha. PINNED: shrink-only.
LEGACY_BOOKED_UNLOCKED = {
    "docs/preregs/qwen-ble-freegen-2.json":
        {"logs/qweneffort2/summary_tower_BLe.json"},
    "docs/preregs/qwen-cycle-impulse-0.json":
        {"logs/qwencycle/impulse_rows.jsonl",
         "logs/qwencycle/impulse_summary.json"},
    "docs/preregs/qwen-lband-1.json":
        {"logs/qwenmodel1/score_BLe.json",
         "logs/qwenmodel1/score_BLm.json",
         "logs/qwenmodel1/score_BLl.json",
         "logs/qwenmodel1/score_FLe.json",
         "logs/qwenmodel1/score_FLm.json",
         "logs/qwenmodel1/score_FLl.json"},
    "docs/preregs/qwen-loop-state-1-headswap.json":
        {"logs/qwenloopstate1/headswap_observations.json"},
    "docs/preregs/qwen-model1-tree.json":
        {"logs/qwenmodel1/tree_observations.json"},
    "docs/preregs/stream-wdistill-0.json":
        {"logs/streamwd/run_B1.log",
         "logs/streamwd/run_B1_repair.log"},
}


def test_booked_prereg_receipts_are_sha_locked():
    """BOOKING-TIME INVARIANT (2026-08-20): every receipt path a
    BOOKED prereg declares must resolve to exists + sha256 in the
    lock — independent of how the booking prose cited it. A prereg
    is booked when a verdict/null row links its results_id."""
    import glob as _glob
    idx = [json.loads(l) for l in
           (ROOT / "docs" / "results-index.jsonl").read_text()
           .splitlines() if l.strip()]
    booked_ids = set()
    for r in idx:
        if r.get("type") in ("verdict", "null"):
            booked_ids.update(r.get("links") or [])
    lock = _locked()
    problems = []
    for p in sorted(_glob.glob(str(ROOT / "docs/preregs/*.json"))):
        if p.endswith(".params.json"):
            continue
        d = json.loads(Path(p).read_text())
        if d.get("results_id") not in booked_ids:
            continue
        rel_p = str(Path(p).relative_to(ROOT))
        allowed = LEGACY_BOOKED_UNLOCKED.get(rel_p, set())
        for rel in d.get("receipts", []):
            if rel in allowed:
                continue
            rec = lock.get(rel, {})
            # a booked NOT-RUN (e.g. CONTROL-MATCH-FAILED) leaves
            # declared treatment receipts legitimately absent —
            # those stay visible as prereg-awaiting-run. The
            # invariant bites when the file EXISTS without a sha.
            if rec.get("exists") and not rec.get("sha256"):
                problems.append(f"{rel_p}: {rel}")
    assert not problems, (
        "booked prereg receipts with no locked sha (cite the full "
        "path in the booking so the lock freezes it):\n  "
        + "\n  ".join(problems))
