#!/usr/bin/env python3
"""Regenerate whatever a ledger edit just invalidated.

The ledger has four generated surfaces, each derived from a file a
session edits by hand:

  docs/RESULTS.md   -> docs/results-index.jsonl (gen_results_index.py)
                       + the FINDINGS ratchet headroom line
  docs/FINDINGS.md  -> README's honesty-ledger region and the
                       honesty_ledger figure in docs/figures.json
                       (gen_readme.py owns both)
  scratch/*.py      -> scripts/INDEX.md (gen_index.py)
  scripts/*.py

This hook fires on Edit|Write AND on Bash. Bash matters because
RESULTS.md is ~30k lines: the booking ritual appends with a heredoc
(`cat >> docs/RESULTS.md << 'EOF'`), which is a Bash call, so an
Edit|Write-only hook never sees the largest and most frequent ledger
mutation in the repo. For Bash the trigger is a path mentioned in the
command text — deliberately loose, because the generators are
idempotent and a spurious regen costs a second.

Never fails a tool call: every generator runs best-effort and the
hook exits 0 regardless. It is a convenience, not a gate; the gates
are tests/test_docs_integrity.py and tests/test_gen_readme.py.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "bin" / "python")


def run(*args: str) -> str:
    """Best-effort generator call. Returns whatever the tool said on
    either stream — findings_headroom.py writes its warning to stderr
    and exits 2, so stdout alone would drop exactly the message worth
    surfacing."""
    try:
        r = subprocess.run([PY, *args], cwd=ROOT, capture_output=True,
                           text=True, timeout=120)
        return (r.stdout.strip() or r.stderr.strip())
    except Exception:
        return ""


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return
    ti = data.get("tool_input") or {}
    if data.get("tool_name") == "Bash":
        subject = ti.get("command", "")
    else:
        subject = (ti.get("file_path")
                   or (data.get("tool_response") or {}).get("filePath")
                   or "")
    if not subject:
        return

    notes = []
    if "docs/RESULTS.md" in subject:
        run("scripts/gen_results_index.py")
        # a new booking may cite new receipt paths — the lock must
        # learn them (coverage hole, external review 2026-08-16);
        # gen_receipt_lock refuses changed shas on its own
        run("scripts/gen_receipt_lock.py")
        head = run(".claude/hooks/findings_headroom.py")
        notes.append("results-index + receipt lock regenerated."
                     + (f" {head}" if head else ""))
    if "docs/preregs/" in subject:
        run("scripts/gen_receipt_lock.py")
        notes.append("receipt lock regenerated for prereg-declared "
                     "receipts.")
    if "docs/FINDINGS.md" in subject:
        run("scripts/gen_readme.py")
        notes.append("README + figures.json honesty ledger regenerated "
                     "(counts follow FINDINGS; commit them with the "
                     "booking or the suite goes red).")
    if any(p in subject for p in ("scratch/", "scripts/")) \
            and ".py" in subject:
        run("scripts/gen_index.py")
        notes.append("INDEX regenerated. If this file is NEW: commit it, "
                     "then rerun gen_codemap.py (tracked files only) or "
                     "the suite goes red.")
    if notes:
        print(" ".join(notes))


if __name__ == "__main__":
    main()
