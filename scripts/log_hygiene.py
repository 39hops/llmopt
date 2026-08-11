#!/usr/bin/env python
"""Print-only log hygiene planner (reviewer design, handoff 2026-08-11-0).

Classifies every file under logs/ into four classes and prints a PLAN.
This module NEVER deletes and never edits; the only mutation it can ever
perform is a MOVE into logs/archive/<today>/ behind the double gate
described under --apply, and the default invocation performs none.

Classes (first match wins, top to bottom):
  FROZEN              path OR any parent directory is cited in
                      docs/RESULTS.md ("files cited by booked verdicts
                      stay frozen in place" — scratch doctrine
                      2026-08-06; directory-level citation freezes the
                      whole dir). Citation set built in ONE grep pass
                      over RESULTS.md. The known doubled path
                      logs/archive/logs/archive/ (RESULTS 2421 cites
                      it) is indexed IN PLACE, never re-sorted:
                      everything under it classifies FROZEN.
  PRESERVE-AS-RECEIPT name matches *oomkilled*, *interrupted*,
                      *poisoned*, *_oom* — killed/poisoned-run exhaust
                      kept per the 43x tripwire doctrine (poisoned runs
                      are KILLED not trusted; their logs are receipts).
  SWEEPABLE           matches neither, mtime older than --age-days
                      (default 14).
  UNKNOWN             everything else. Default action: nothing.

--apply is a plan-executor, not a deleter: it refuses unless BOTH
(a) ARTIN_GO=1 is in the environment (task holds are explicit — queued
work runs only on Artin's GO, never on inference from context) and
(b) the plan contains zero FROZEN/UNKNOWN rows. Even when both hold it
only MOVES the SWEEPABLE rows to logs/archive/<YYYY-MM-DD>/ preserving
relative paths. Bulk deletion stays Artin-GO by hand (logs doctrine
2026-08-06); this tool has no delete path at all.

Second plan section: the COMPLETION-SIGNAL CONSOLIDATION MAP — a grep
of scripts/ and scratch/ (top level only, scratch/leancheck excluded)
for writers of .DONE / .rc / .marker / .ep sentinels, printed as
file -> signal(s) -> proposed llmopt/lab/runfiles.py contract note.
Background: the .ep sentinel currently means two different things and
"no marker" means both "finished cleanly" and "never ran" (handoff
2026-08-11-0, eight inconsistencies). Print-only; migration is a
separate adoption thread.

Usage:
  .venv/bin/python scripts/log_hygiene.py [--age-days 14] [--json OUT]
  .venv/bin/python scripts/log_hygiene.py --apply   # refuses w/o gate
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

RECEIPT_GLOBS = ("*oomkilled*", "*interrupted*", "*poisoned*", "*_oom*")
DOUBLED = Path("logs/archive/logs/archive")  # RESULTS 2421; index in place
SIGNAL_EXTS = (".DONE", ".rc", ".marker", ".ep")

FROZEN = "FROZEN"
RECEIPT = "PRESERVE-AS-RECEIPT"
SWEEPABLE = "SWEEPABLE"
UNKNOWN = "UNKNOWN"


def build_citation_set(results_path: Path) -> set[str]:
    """One pass over RESULTS.md: every `logs/...` token becomes a citation.

    Tokens are normalized (trailing punctuation stripped, no leading ./).
    A cited path freezes itself AND, if it is a directory, everything
    under it (checked by prefix in classify_one).
    """
    cites: set[str] = set()
    if not results_path.exists():
        return cites
    # charclass includes glob/brace metachars (red-team 1b): a cite
    # like logs/lean_kernel_sample{,2,3}.log otherwise parses to a
    # token matching no file, and its receipts go uncited
    pat = re.compile(r"logs/[\w\-./{},*]+")
    text = results_path.read_text(errors="replace")
    for m in pat.finditer(text):
        tok = re.sub(r"[.,:;)\]}'\"`/]+$", "", m.group(0))
        if not tok:
            continue
        if any(c in tok for c in "{}*"):
            # glob/brace cite: freeze the PARENT directory — safe
            # direction is over-freezing
            cites.add(str(Path(tok).parent))
        else:
            cites.add(tok)
    return cites


def is_cited(rel: str, cites: set[str], basenames: set[str] | None = None,
             ) -> bool:
    """True if rel or any parent dir of rel appears in the citation set,
    OR (red-team 1a) its basename appears bare anywhere in RESULTS —
    entries routinely cite one receipt with the logs/ prefix and name
    its siblings bare; over-freezing is the safe direction."""
    p = Path(rel)
    for anc in [p, *p.parents]:
        s = str(anc)
        if s in ("logs", "."):
            break
        if s in cites:
            return True
    if basenames is not None and p.name in basenames:
        return True
    return False


def cited_basenames(results_path: Path) -> set[str]:
    """Basenames of every *.log/*.jsonl-looking token in RESULTS,
    prefix or no prefix — the second citation pass."""
    if not results_path.exists():
        return set()
    text = results_path.read_text(errors="replace")
    return set(re.findall(r"([\w][\w\-.]*\.(?:log|jsonl|w9|json))", text))


def is_receipt(name: str) -> bool:
    import fnmatch

    return any(fnmatch.fnmatch(name, g) for g in RECEIPT_GLOBS)


def classify_one(rel: str, mtime: float, cites: set[str], age_days: float,
                 now: float | None = None,
                 basenames: set[str] | None = None) -> tuple[str, str]:
    """Return (class, reason) for one logs/-relative path."""
    now = time.time() if now is None else now
    if str(Path(rel)).startswith(str(DOUBLED)) or is_cited(rel, cites,
                                                           basenames):
        if str(Path(rel)).startswith(str(DOUBLED)):
            return FROZEN, "under doubled path logs/archive/logs/archive/ (RESULTS 2421); index in place"
        return FROZEN, "path, parent dir, or bare basename cited in docs/RESULTS.md"
    if is_receipt(Path(rel).name):
        return RECEIPT, "killed/poisoned-run receipt (name pattern)"
    if (now - mtime) > age_days * 86400:
        return SWEEPABLE, f"uncited, older than {age_days:g} days"
    return UNKNOWN, "uncited but recent; default action nothing"


def scan(root: Path, cites: set[str], age_days: float,
         basenames: set[str] | None = None) -> list[dict]:
    rows = []
    logs = root / "logs"
    if not logs.is_dir():
        return rows
    for p in sorted(logs.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        rel = str(p.relative_to(root))
        st = p.stat()
        cls, reason = classify_one(rel, st.st_mtime, cites, age_days,
                                   basenames=basenames)
        rows.append({"path": rel, "class": cls, "bytes": st.st_size,
                     "reason": reason})
    return rows


def consolidation_map(root: Path) -> list[dict]:
    """Grep scripts/ + scratch/ (top level, minus scratch/leancheck) for
    completion-signal writers. Print-only; proposes runfiles.py migration."""
    out = []
    files: list[Path] = []
    for d in ("scripts", "scratch"):
        dd = root / d
        if dd.is_dir():
            files += [f for f in sorted(dd.iterdir())
                      if f.is_file() and f.suffix in (".py", ".sh")]
    pat = re.compile(r"['\"][^'\"\n]*\.(DONE|rc|marker|ep)\b")
    self_path = Path(__file__).resolve()
    for f in files:
        if f.resolve() == self_path:
            continue  # the planner names the extensions; it writes none
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        sigs = sorted({"." + m.group(1) for m in pat.finditer(text)})
        if sigs:
            out.append({
                "file": str(f.relative_to(root)),
                "signals": sigs,
                "note": ("migrate to llmopt/lab/runfiles.py: write_marker() "
                         "one-JSON-line contract; is_done/rc_of replace "
                         "ad-hoc " + ",".join(sigs) + " sentinels"),
            })
    return out


def print_plan(rows: list[dict], cmap: list[dict], out=sys.stdout) -> None:
    w = out.write
    w("== LOG HYGIENE PLAN (print-only) ==\n")
    w(f"{'CLASS':<20} {'BYTES':>12}  PATH  -- REASON\n")
    for r in rows:
        w(f"{r['class']:<20} {r['bytes']:>12}  {r['path']}  -- {r['reason']}\n")
    counts: dict[str, list] = {}
    for r in rows:
        counts.setdefault(r["class"], [0, 0])
        counts[r["class"]][0] += 1
        counts[r["class"]][1] += r["bytes"]
    w("\n-- summary --\n")
    for cls in (FROZEN, RECEIPT, SWEEPABLE, UNKNOWN):
        n, b = counts.get(cls, (0, 0))
        w(f"{cls:<20} n={n:<6} bytes={b}\n")
    w("\n== COMPLETION-SIGNAL CONSOLIDATION MAP (print-only) ==\n")
    for c in cmap:
        w(f"{c['file']}  ->  {','.join(c['signals'])}  ->  {c['note']}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--age-days", type=float, default=14.0)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--root", default=str(REPO), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    root = Path(args.root)

    cites = build_citation_set(root / "docs" / "RESULTS.md")
    basenames = cited_basenames(root / "docs" / "RESULTS.md")
    rows = scan(root, cites, args.age_days, basenames=basenames)
    cmap = consolidation_map(root)
    print_plan(rows, cmap)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"plan": rows, "consolidation_map": cmap}, indent=1))

    if args.apply:
        blocked = [r for r in rows if r["class"] in (FROZEN, UNKNOWN)]
        if os.environ.get("ARTIN_GO") != "1" or blocked:
            print("refusing: apply requires ARTIN_GO=1 env and no "
                  "FROZEN/UNKNOWN rows in plan")
            return 1
        # Double gate held: MOVE (never delete) SWEEPABLE rows only.
        today = _dt.date.today().isoformat()
        dest_root = root / "logs" / "archive" / today
        for r in rows:
            if r["class"] != SWEEPABLE:
                continue
            # already-archived rows are skipped (red-team 1c: moving
            # logs/archive/x re-creates the doubled-path pathology)
            if Path(r["path"]).parts[:2] == ("logs", "archive"):
                print(f"skip (already archived): {r['path']}")
                continue
            src = root / r["path"]
            rel_under_logs = Path(r["path"]).relative_to("logs")
            dst = dest_root / rel_under_logs
            if dst.exists():
                # rename() clobbers on POSIX (red-team H1) — a
                # no-delete tool must never overwrite
                print(f"refusing move (destination exists): {dst}")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            print(f"moved {r['path']} -> {dst.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
