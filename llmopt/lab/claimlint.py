"""Claim linter: prose verdict language checked against machine objects.

Built LAST of the five automation items, deliberately — Artin's
ordering made it conditional on the prereg schema giving words like
"matched", "only", and "independent" actual objects to interrogate
(bars, arms, admissibility reasons, populations) instead of vibes.

THE INCIDENT CORPUS (all one thread, STREAM-WDISTILL-0, each caught
by user-relayed review AFTER the prose shipped): "Lloyd-optimal";
"differ ONLY in vector width" (five things differed);
"independent support" from the registered control arm itself;
"statistically indistinguishable" with no distribution;
"near-isotropic" refuted by the same receipt's own numbers;
"matched bytes" for a contrast whose scalar arm was over budget.
Every one is a WORD claiming more than an object supports.

THREE LAYERS, weakest to strongest:

1. DENY REGISTRY (docs/claims.deny.json): superseded readings.
   ERROR on match. The registry is append-only provenance — a
   reading correction books, its pattern lands here, and the dead
   phrase cannot quietly reappear in new prose.

2. OVERCLAIM WORDS (context-free): phrases that are not banned but
   carry a proof obligation the linter cannot see. WARN, naming the
   obligation and the incident that earned the rule.

3. ADJUDICATION CHECKS (with --prereg/--obs): the strong layer.
   ERROR when prose contradicts the deterministic adjudicator —
   verdict words (fires / no-fire) that disagree with a bar's
   computed outcome, or contest words (matched / winner / loses /
   beats) applied to the arms of an UNRESOLVED bar.

SCOPE. New drafts only (a verdict about to book, a FINDINGS bullet,
analysis prose). NEVER run over historical ledger text — RESULTS.md
is append-only and its old wording is the record, including its
mistakes and their amendments.

Findings are proposals in the house sense: an ERROR blocks a clean
exit, but the session model resolves each one by fixing the prose,
narrowing the claim, or (for a genuine false positive) saying so in
the booking with the rule it overrides.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DENY = ROOT / "docs" / "claims.deny.json"

# Layer 2: phrase -> the proof obligation it carries.
OVERCLAIM = [
    (re.compile(r"\bdiffer(s|ed)? only in\b|\bonly difference\b", re.I),
     "enumerate the FULL difference set — the 'differ only in vector "
     "width' claim hid five differences (deltas incident)"),
    (re.compile(r"\bstatistically significant\b", re.I),
     "requires a measured distribution and stated n; single pooled "
     "scalars cannot bear this"),
    (re.compile(r"\bindependent(ly)? (verifie[sd]|confirm(s|ed))\b", re.I),
     "independence requires a DISTINCT registered source — a second "
     "look at the same arm is convergent, not independent"),
    (re.compile(r"\bproves?\b", re.I),
     "measured support is not proof; state the measurement and its "
     "fences"),
    (re.compile(r"\bexactly captured energy\b|\bexact(ly)? capture[sd]?\b",
                re.I),
     "1 - rel_frob^2 is a capture PROXY after fp16 basis rounding; "
     "exact form is ||V^T W||^2 / ||W||^2"),
]

# Layer 3 vocab.
FIRE_CLAIM = re.compile(
    r"\bBAR\s*(\d+)\b[^.\n]{0,80}?\b(NO[- ]FIRE|FIRES?|FIRED)\b", re.I)
CONTEST = re.compile(
    r"\b(matched[- ]bytes?|winner|wins|loses?|lost|beats?|beaten)\b", re.I)


@dataclass(frozen=True)
class Finding:
    severity: str      # "ERROR" | "WARN"
    line: int          # 1-indexed in the linted text
    rule: str          # short rule id
    excerpt: str
    message: str


def _deny_rules() -> list[dict]:
    return json.loads(DENY.read_text())["deny"]


def lint_text(text: str, outcomes: list | None = None) -> list[Finding]:
    """Lint prose. `outcomes` is an optional list of BarOutcome from
    llmopt.lab.prereg.adjudicate_prereg — supplying it arms layer 3.
    """
    findings: list[Finding] = []
    lines = text.split("\n")

    for rule in _deny_rules():
        rx = re.compile(rule["pattern"], re.I)
        for i, ln in enumerate(lines, 1):
            m = rx.search(ln)
            if m:
                findings.append(Finding(
                    "ERROR", i, "superseded-reading", m.group(0),
                    f"{rule['reason']} (superseded by "
                    f"{rule['superseded_by']})"))

    for rx, obligation in OVERCLAIM:
        for i, ln in enumerate(lines, 1):
            m = rx.search(ln)
            if m:
                findings.append(Finding(
                    "WARN", i, "overclaim-word", m.group(0), obligation))

    if outcomes is not None:
        by_id = {o.bar_id: o for o in outcomes}
        for i, ln in enumerate(lines, 1):
            for m in FIRE_CLAIM.finditer(ln):
                bar_id, word = int(m.group(1)), m.group(2).upper()
                o = by_id.get(bar_id)
                if o is None:
                    findings.append(Finding(
                        "ERROR", i, "unknown-bar", m.group(0),
                        f"prose names BAR {bar_id}; the pre-reg has no "
                        "such bar"))
                    continue
                claimed = ("NO-FIRE" if "NO" in word else "FIRE")
                if o.outcome == "UNRESOLVED":
                    findings.append(Finding(
                        "ERROR", i, "verdict-on-unresolved", m.group(0),
                        f"BAR {bar_id} adjudicated UNRESOLVED "
                        f"({'; '.join(o.reasons)}) — no fire/no-fire "
                        "sentence may be written about it"))
                elif o.outcome != claimed:
                    findings.append(Finding(
                        "ERROR", i, "prose-contradicts-adjudicator",
                        m.group(0),
                        f"prose claims {claimed}, adjudicator computed "
                        f"{o.outcome} for BAR {bar_id}"))
        unresolved = [o for o in by_id.values()
                      if o.outcome == "UNRESOLVED"]
        if unresolved:
            for i, ln in enumerate(lines, 1):
                m = CONTEST.search(ln)
                if m:
                    findings.append(Finding(
                        "ERROR", i, "contest-word-unresolved",
                        m.group(0),
                        "contest wording with an UNRESOLVED bar in "
                        "scope (bars "
                        + ", ".join(str(o.bar_id) for o in unresolved)
                        + ") — comparisons touching an inadmissible "
                        "arm stay descriptive"))
    return sorted(findings, key=lambda f: (f.line, f.rule))
