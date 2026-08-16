"""Lint DRAFT verdict/analysis prose before it books.

    .venv/bin/python scripts/claim_lint.py <draft.md>
    .venv/bin/python scripts/claim_lint.py <draft.md> \
        --prereg docs/preregs/<name>.json --obs <observations.json>

Layers (llmopt/lab/claimlint.py): superseded-reading registry
(docs/claims.deny.json, ERROR), overclaim words carrying proof
obligations (WARN), and — with --prereg/--obs — prose checked
against the deterministic adjudicator (ERROR on contradiction, on a
fire/no-fire sentence about an UNRESOLVED bar, and on contest
wording while an UNRESOLVED bar is in scope).

Exit 1 on any ERROR; WARNs alone exit 0. NEVER run this over
historical RESULTS.md text — old wording is the record.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.claimlint import lint_text            # noqa: E402
from llmopt.lab.prereg import adjudicate_prereg, load  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("draft", help="draft prose file to lint")
    ap.add_argument("--prereg", help="docs/preregs/<name>.json")
    ap.add_argument("--obs", help="observations JSON (with --prereg)")
    a = ap.parse_args()
    outcomes = None
    if a.prereg:
        if not a.obs:
            ap.error("--prereg requires --obs")
        outcomes = adjudicate_prereg(
            load(a.prereg), json.loads(Path(a.obs).read_text()))
    findings = lint_text(Path(a.draft).read_text(), outcomes)
    errors = 0
    for f in findings:
        errors += f.severity == "ERROR"
        print(f"{a.draft}:{f.line}: {f.severity} [{f.rule}] "
              f"{f.excerpt!r} — {f.message}")
    print(f"{len(findings)} finding(s), {errors} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
