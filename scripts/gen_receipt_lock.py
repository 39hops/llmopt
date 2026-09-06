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
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "RESULTS.md"
LOCK = ROOT / "docs" / "receipts.lock.json"
# logs/<dir>/<file.ext> as cited in prose; kept deliberately narrow so
# directory-only mentions ("receipts land in logs/foo/") are ignored.
CITE = re.compile(
    r"\blogs/(?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_.\-]+\.[A-Za-z0-9]+\b")
# brace-list citations — logs/<dir>/{a.ext, b.ext, ...} — expand to one
# path per member. Whitespace/newlines inside the braces are prose
# wrapping, not structure; members that do not look like filenames
# (no extension) are ignored rather than guessed at.
BRACE = re.compile(r"\blogs/(?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_.\-]*"
                   r"\{[^{}]+\}[A-Za-z0-9_.\-]*")
_MEMBER = re.compile(r"^[A-Za-z0-9_.\-]+$")
_FILE = re.compile(r"^[A-Za-z0-9_.\-]+\.[A-Za-z0-9]+$")


def expand_braces(text: str) -> list[str]:
    out = []
    for m in BRACE.finditer(text):
        s = m.group(0)
        head, rest = s[:s.index("{")], s[s.index("{") + 1:]
        inner, tail = rest[:rest.rindex("}")], rest[rest.rindex("}") + 1:]
        whole_name = head.endswith("/") and tail == ""
        for part in inner.split(","):
            toks = part.split()
            if whole_name and toks:
                # full-filename members may carry trailing prose
                # annotations ("x.json (producer, ..."): first token
                part = toks[0]
            elif len(toks) == 1:
                part = toks[0]
            else:
                continue  # multi-word prose ("same six"), not a name
            if _MEMBER.fullmatch(part) and _FILE.fullmatch(
                    (head + part + tail).rsplit("/", 1)[-1]):
                out.append(head + part + tail)
    return out


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cited_paths() -> dict[str, str]:
    """path -> source. "results" = cited in RESULTS.md prose (absence
    is the booked-without-evidence backlog, ratcheted). "prereg" =
    declared by docs/preregs/*.json (structured receipt references —
    closes the bare-filename gap); a declared path may legitimately
    not exist yet (PENDING, the run has not fired) and is not
    ratcheted. A prereg-declared path that EXISTS also stays
    pending (sha-less) until a booking cites it in RESULTS in a
    scanner-visible form — deliberately, because sha-ing a
    prereg-declared file can freeze a mid-run write (caught live
    2026-08-16). The booking-time invariant lives in
    tests/science_incidents/test_frozen_receipt_mutation.py:
    every receipt of a BOOKED prereg must end up sha-locked."""
    text = RESULTS.read_text()

    def norm(p: str) -> str:
        # a doubled prefix in prose ("logs/x/logs/x/f.log", one
        # historical typo at RESULTS L2421) cites the file at the
        # LAST logs/ root; no real receipt nests a logs/ dir.
        return p[p.rindex("logs/"):] if "/logs/" in p else p

    out = {norm(m.group(0)): "results" for m in CITE.finditer(text)}
    out.update({norm(p): "results" for p in expand_braces(text)})
    # prereg-DECLARED paths win the pending classification while the
    # file does not exist, even when the registration prose also
    # names them (a pre-reg legitimately cites paths its run WILL
    # write; the absence ratchet must not fire on those, and the
    # sha freeze arrives with the verdict booking).
    for prereg in sorted((ROOT / "docs" / "preregs").glob("*.json")):
        for rel in json.loads(prereg.read_text()).get("receipts", []):
            if not (ROOT / rel).is_file():
                out[rel] = "prereg"
            else:
                out.setdefault(rel, "prereg")
    return out


def build() -> dict:
    out = {}
    for rel, src in sorted(cited_paths().items()):
        p = ROOT / rel
        if p.is_dir():
            # a DIRECTORY citation names a receipt set whose files are
            # locked individually (a dotted directory name such as
            # transport_3.7B parses like a file path; 2026-09-06)
            rec = {"exists": True, "directory": True}
        elif not p.is_file():
            rec = {"exists": False}
        elif src == "prereg":
            # a prereg-DECLARED receipt whose run may still be
            # WRITING: record presence, do not freeze bytes — the sha
            # locks when the booking makes the path results-cited
            # (caught live 2026-08-16: the lock sha'd run_0s.log
            # mid-run and the invariant fired on the growing file)
            rec = {"exists": True, "pending": True}
        else:
            rec = {"exists": True, "sha256": sha256(p),
                   "bytes": p.stat().st_size}
            # trackedness is part of the record: exists=true for a
            # file only one machine holds is not evidence-in-repo.
            # Small text receipts get force-added (logs doctrine);
            # large streams stay machine-local and say so.
            r = subprocess.run(["git", "ls-files", "--error-unmatch",
                                rel], cwd=ROOT, capture_output=True)
            rec["tracked"] = r.returncode == 0
            if not rec["tracked"]:
                rec["local_only"] = True
        rec["source"] = src
        out[rel] = rec
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
               and old[k].get("sha256")
               and (not v.get("exists")
                    or v.get("sha256") != old[k].get("sha256"))]
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
    locked = sum(1 for v in fresh.values()
                 if v.get("exists") and v.get("sha256"))
    live_pending = sum(1 for v in fresh.values()
                       if v.get("exists") and v.get("pending"))
    backlog = sum(1 for v in fresh.values()
                  if not v.get("exists") and v.get("source") == "results")
    awaiting = sum(1 for v in fresh.values()
                   if not v.get("exists") and v.get("source") == "prereg")
    print(f"{len(fresh)} cited receipt paths -> {LOCK.relative_to(ROOT)} "
          f"({locked} sha-locked, {live_pending} present-but-pending, "
          f"{backlog} cited-but-absent, {awaiting} prereg-awaiting-run)")
    if changed:
        print(f"ACCEPTED {len(changed)} changed: {a.accept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
