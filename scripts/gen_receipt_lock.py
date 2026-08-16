"""Content-address every receipt path a booked entry cites.

THE INCIDENT (2026-08-16, STREAM-WDISTILL-0). The audit-repair was
launched after manually renaming logs/streamwd/pass12_B1.jsonl — a
path OBSERVATION -EXEC1 cites as its booked evidence. The rename
defeated the driver's own refuse-if-exists guard, and the booked
citation then resolved to a DIFFERENT run's numbers while the cited
log file no longer existed at all. Both auditors caught it
independently; neither the driver guard nor any test did.

The doctrine ("never append a new run into a path a booked verdict
cites as frozen") was already written. What was missing is that
nothing could OBSERVE a violation: a receipt is just a file, and a
file that changes leaves no trace in the ledger.

This closes that. It records, for every logs/ path named in
RESULTS.md, whether the file exists and the sha256 of its bytes.
tests/science_incidents/test_frozen_receipt_mutation.py then fails
if any locked receipt changes content, disappears, or is truncated.

A legitimate NEW execution writes to a NEW path — that is the whole
point of the doctrine, and the lock makes it enforceable rather
than merely stated. Adding a new cited path is a normal lock
update; CHANGING an existing one requires --accept plus a reason,
so it lands in a diff a reviewer can see.

    .venv/bin/python scripts/gen_receipt_lock.py
    .venv/bin/python scripts/gen_receipt_lock.py --accept "why"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "RESULTS.md"
LOCK = ROOT / "docs" / "receipts.lock.json"
# logs/<dir>/<file.ext> as cited in prose; kept deliberately narrow so
# directory-only mentions ("receipts land in logs/foo/") are ignored.
CITE = re.compile(r"\blogs/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+\.[A-Za-z0-9]+\b")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cited_paths() -> list[str]:
    text = RESULTS.read_text()
    return sorted({m.group(0) for m in CITE.finditer(text)})


def build() -> dict:
    out = {}
    for rel in cited_paths():
        p = ROOT / rel
        out[rel] = ({"exists": True, "sha256": sha256(p), "bytes": p.stat().st_size}
                    if p.is_file() else {"exists": False})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accept", metavar="REASON",
                    help="accept changed shas for ALREADY-LOCKED paths")
    a = ap.parse_args()

    fresh = build()
    old = (json.loads(LOCK.read_text()).get("receipts", {})
           if LOCK.exists() else {})

    changed = [k for k, v in fresh.items()
               if k in old and old[k].get("exists")
               and (not v.get("exists") or v.get("sha256") != old[k].get("sha256"))]
    if changed and not a.accept:
        print("REFUSING: locked receipts changed or vanished — a NEW run "
              "belongs at a NEW path.\n  " + "\n  ".join(changed))
        print("\nIf the change is legitimate, re-run with "
              '--accept "reason" so it lands in a reviewable diff.')
        return 1

    payload = {
        "_doc": "sha256 of every receipt path cited by docs/RESULTS.md. "
                "A booked receipt is evidence; changing one silently "
                "rewrites the record. Regenerate with "
                "scripts/gen_receipt_lock.py.",
        "receipts": fresh,
    }
    if a.accept:
        payload["_last_accept"] = {"reason": a.accept, "paths": changed}
    LOCK.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    live = sum(1 for v in fresh.values() if v.get("exists"))
    print(f"{len(fresh)} cited receipt paths -> {LOCK.relative_to(ROOT)} "
          f"({live} present, {len(fresh) - live} cited-but-absent)")
    if changed:
        print(f"ACCEPTED {len(changed)} changed: {a.accept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
