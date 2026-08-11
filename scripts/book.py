"""Programmatic booking — the /book ritual as a refusing machine.

Turns a runfiles marker (llmopt/lab/runfiles.py, the Spark-_SUCCESS
contract) plus a human-written entry body into a RESULTS.md booking,
mechanically enforcing the fences that were each learned the hard
way (SKILL.md .claude/skills/book/, 2026-08-01 checksums session):

- REFUSES killed/nonzero/missing markers — booking a killed run is
  the checkpoint selection-effect (bit three times; CLAUDE.md fork
  doctrine corollary, 2026-07-12).
- CHECKSUM RULE: a claimed gate total ('= N/120') must equal the
  sum of the marker's gate_dict — the dict is the checksum (the
  '48 booked from valid 48.27' class, survived TWO review passes).
- A marker's weights_sha must appear in the entry text — a gate
  books WITH its sha (provenance rule graduated at RESULTS 13463;
  dtype-sensitive, never compared across precisions).
- STATISTICAL FENCE (resolution law 2026-07-31): a verdict claiming
  a gate delta |d| < 7 solves off a single seed needs
  --fence-acknowledged AND an explicit single-seed fence sentence
  in the entry, or it does not book.

Appends (append-only, house heading format), then CALLS
scripts/gen_results_index.py (frozen, results-cited — never
reimplemented here), then curates threads/links onto the new index
row, then PRINTS the git command. It never commits — commits are
the session model's accountability, not this script's.

    .venv/bin/python scripts/book.py --marker logs/run/run.marker.json \
        --entry /path/entry.md --title "NAME: claim (2026-08-11, mac)" \
        --type verdict --threads engine,alphabet --links some-id [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llmopt.lab.runfiles import read_marker, rc_of  # noqa: E402

# Module-level path constants so tests can redirect onto a tmp copy
# (never the live RESULTS.md).
RESULTS_PATH = REPO_ROOT / "docs" / "RESULTS.md"
INDEX_PATH = REPO_ROOT / "docs" / "results-index.jsonl"
GEN_INDEX_SCRIPT = REPO_ROOT / "scripts" / "gen_results_index.py"
# gen_results_index.py resolves docs/ relative to its cwd; tests
# point this at a tmp repo root.
INDEX_CWD = REPO_ROOT

TYPE_WORD = {"verdict": "VERDICT", "prereg": "PRE-REG",
             "amendment": "AMENDMENT"}

GATE_TOTAL_RE = re.compile(r"=\s*(\d+)\s*/\s*120")
# Delta detection (2026-08-11 review, bypasses B1/B2): never match a
# bare `d=` — `d=512` (an architecture width) was the first match in
# most house entries and made the fence unreachable. Match the words
# delta/Δ with `=`, `of`, or bare (`delta +3`), AND the house's most
# natural phrasing `+3 solves` / `3-solve`.
DELTA_RE = re.compile(
    r"(?:delta|Δ)\s*(?:=|of)?\s*([+-]?\d+(?:\.\d+)?)"
    r"|([+-]?\d+(?:\.\d+)?)[\s-]*solves?\b",
    re.IGNORECASE)
FENCE_SENTENCE_RE = re.compile(r"single[- ]seed", re.IGNORECASE)


class Refusal(SystemExit):
    """A booking refusal; message names its doctrine."""

    def __init__(self, msg: str):
        print(f"REFUSED: {msg}", file=sys.stderr)
        super().__init__(2)


def validate_marker(marker_path: Path) -> dict:
    """Fence 1: only clean, finished runs book. Absence is 'never
    ran or still running', never 'finished cleanly' (runfiles
    contract); a killed/nonzero run booking is the checkpoint
    selection-effect."""
    m = read_marker(marker_path)
    if m is None:
        raise Refusal(
            f"no parseable marker at {marker_path} — absence means "
            "never-ran-or-still-running, not finished (runfiles "
            "contract; booking it would be the checkpoint "
            "selection-effect).")
    rc = rc_of(marker_path)
    if rc is None:
        raise Refusal(
            f"marker rc is non-integer ({m.get('rc')!r}) — a killed "
            "run does not book (checkpoint selection-effect: the "
            "killed class becomes invisible to whatever reads the "
            "ledger).")
    if rc != 0:
        raise Refusal(
            f"marker rc={rc} != 0 — a failed run does not book as "
            "if it finished (checkpoint selection-effect).")
    return m


def validate_gate_checksum(entry: str, marker: dict) -> None:
    """Fence 2: gate numbers book as DICTS, not totals — the dict
    is the checksum ('48 from valid 48.27', 2026-08-01)."""
    claims = [int(g) for g in GATE_TOTAL_RE.findall(entry)]
    if not claims:
        return
    gate_dict = marker.get("gate_dict")
    if isinstance(gate_dict, str):        # lake stores it as JSON text
        try:
            gate_dict = json.loads(gate_dict)
        except json.JSONDecodeError:
            raise Refusal("marker gate_dict is an unparseable string "
                          "— fix the marker before booking.")
    if gate_dict is None:
        return
    if not isinstance(gate_dict, dict):
        raise Refusal(f"marker gate_dict has unexpected shape "
                      f"({type(gate_dict).__name__}) — refusing "
                      "rather than skipping the checksum.")
    try:
        total = sum(int(v) for v in gate_dict.values())
    except (TypeError, ValueError):
        raise Refusal(f"marker gate_dict values are not integers "
                      f"({gate_dict}) — refusing rather than "
                      "skipping the checksum.")
    # EVERY total in the entry must match some accounting: the
    # marker's dict must equal at least one claimed total, and any
    # OTHER totals are allowed only if the entry carries their own
    # dicts (checked by eye at review; here we pin the marker's).
    if total not in claims:
        raise Refusal(
            f"entry claims totals {claims}/120 but marker gate_dict "
            f"sums to {total} ({gate_dict}) — the dict is the "
            "checksum (the '48 booked from valid 48.27' class, "
            "2026-08-01).")


def validate_weights_sha(entry: str, marker: dict) -> None:
    """Fence 3: a gate books WITH its weights sha (provenance rule
    graduated at RESULTS 13463; dtype-sensitive)."""
    sha = marker.get("weights_sha")
    if sha and sha not in entry:
        raise Refusal(
            f"marker carries weights_sha {sha} but the entry text "
            "does not quote it — a gate books WITH its sha "
            "(RESULTS 13463 provenance rule).")


def validate_statistical_fence(entry: str, marker: dict,
                               entry_type: str,
                               fence_acknowledged: bool) -> None:
    """Fence 4 (resolution law 2026-07-31): gate deltas < 1.5 sigma
    (< ~7 solves on the 120 gate) need n>=3 paired seeds before a
    direction is claimed; a single-seed verdict books only with
    --fence-acknowledged AND an explicit fence sentence."""
    if entry_type != "verdict":
        return
    # scan ALL delta mentions; the fence fires on the SMALLEST (B1:
    # first-match let `d=512` shadow a real `delta = +3`).
    deltas = [abs(float(a or b)) for a, b in DELTA_RE.findall(entry)]
    if not deltas or min(deltas) >= 7:
        return
    # n_seeds absent or non-int = UNKNOWN, which is fenced, not
    # skipped (B3: absence of evidence is not multi-seed evidence).
    try:
        n_seeds = int(marker.get("n_seeds"))
    except (TypeError, ValueError):
        n_seeds = 1
    if n_seeds > 1:
        return
    if not (fence_acknowledged and FENCE_SENTENCE_RE.search(entry)):
        raise Refusal(
            f"verdict claims sub-sigma gate delta |{min(deltas)}| < 7 "
            "solves without n_seeds>1 evidence — resolution law "
            "(2026-07-31) needs n>=3 paired seeds for a direction, or "
            "--fence-acknowledged plus an explicit single-seed fence "
            "sentence in the entry.")


def append_entry(heading: str, entry: str) -> None:
    """Append-only, house heading format (SKILL.md step 1)."""
    text = RESULTS_PATH.read_text()
    block = ("" if text.endswith("\n\n") else
             ("\n" if text.endswith("\n") else "\n\n"))
    block += f"## {heading}\n\n{entry.rstrip()}\n"
    RESULTS_PATH.write_text(text + block)


def regen_index() -> None:
    """CALL the frozen results-cited script — never reimplement."""
    r = subprocess.run([sys.executable, str(GEN_INDEX_SCRIPT)],
                       cwd=INDEX_CWD, capture_output=True, text=True)
    if r.returncode != 0:
        raise Refusal(f"gen_results_index.py failed:\n{r.stderr}")
    print(r.stdout, end="")


def curate_index_row(heading: str, threads: list[str],
                     links: list[str]) -> dict:
    """SKILL.md step 3: set threads/links on the new row, pop
    needs_link, rewrite."""
    rows = [json.loads(ln) for ln in
            INDEX_PATH.read_text().splitlines() if ln.strip()]
    target = None
    for row in rows:
        if row["title"] == heading:
            target = row  # last match = the just-appended entry
    if target is None:
        raise Refusal(f"appended entry not found in index: {heading!r}")
    if threads:
        target["threads"] = threads
    if links:
        target["links"] = links
    target.pop("needs_link", None)
    with INDEX_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--marker", required=True)
    ap.add_argument("--entry", required=True,
                    help="path to the human-written entry body (md)")
    ap.add_argument("--title", required=True,
                    help="'<NAME>: <claim> (<date>, <machine>)'")
    ap.add_argument("--type", required=True, dest="entry_type",
                    choices=sorted(TYPE_WORD))
    ap.add_argument("--threads", default="",
                    help="comma-separated kebab-case program names")
    ap.add_argument("--links", default="",
                    help="comma-separated related entry ids")
    ap.add_argument("--fence-acknowledged", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    marker = validate_marker(Path(a.marker))
    entry = Path(a.entry).read_text()
    validate_gate_checksum(entry, marker)
    validate_weights_sha(entry, marker)
    validate_statistical_fence(entry, marker, a.entry_type,
                               a.fence_acknowledged)

    heading = f"{TYPE_WORD[a.entry_type]} {a.title}"
    threads = [t for t in a.threads.split(",") if t]
    links = [l for l in a.links.split(",") if l]
    git_cmd = ("git add docs/RESULTS.md docs/results-index.jsonl && "
               f"git commit -m 'book: {a.title}\n\n"
               "Co-Authored-By: Claude <noreply@anthropic.com>'")

    if a.dry_run:
        print("DRY RUN — nothing written.")
        print(f"heading: ## {heading}")
        print(f"threads: {threads}  links: {links}")
        print(f"marker: rc=0 kind={marker.get('kind')} "
              f"git_sha={marker.get('git_sha')}")
        print("entry body:\n" + entry)
        print("would run: " + git_cmd)
        return 0

    # atomic-ish booking (B8): snapshot both ledger files; restore on
    # any downstream refusal so a half-booking never survives.
    snap_results = RESULTS_PATH.read_text()
    snap_index = INDEX_PATH.read_text() if INDEX_PATH.exists() else None
    try:
        append_entry(heading, entry)
        regen_index()
        row = curate_index_row(heading, threads, links)
    except BaseException:
        RESULTS_PATH.write_text(snap_results)
        if snap_index is not None:
            INDEX_PATH.write_text(snap_index)
        print("rolled back RESULTS.md + index to pre-booking state",
              file=sys.stderr)
        raise
    print(f"booked: {row['id']}")
    print("NOW RUN (this script never commits):")
    print(git_cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
