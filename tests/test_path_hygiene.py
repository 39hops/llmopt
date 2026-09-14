"""PATH-HYGIENE lint (adopted 2026-09-14, PATH-HYGIENE SCRUB GO): no tracked
file may carry a machine-local home path (/Users/<user>/, /home/<user>/,
C:\\Users\\<user>\\ and the WSL form /mnt/c/Users/<user>/). Artifacts are named
by logical locators (llmopt.lab.locator: worktree_role + relative_path +
commit + digest) and resolved at runtime.

Allowlist, narrow and explicit: immutable legacy receipts under logs/
(sha-locked evidence, never mutated), the append-only ledger
docs/RESULTS.md at its frozen count of offending lines (a ratchet: new
entries may not add any), and the enumerated legacy documents below,
which are historical records rather than instruments."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"(/Users/[A-Za-z0-9_.-]+/|/home/[A-Za-z0-9_.-]+/|[A-Za-z]:\\Users\\[A-Za-z0-9_.-]+\\|/mnt/[a-z]/Users/[A-Za-z0-9_.-]+/)")
# the lint's own pattern text and the locator docstring name the forbidden forms
SELF = {"tests/test_path_hygiene.py", "llmopt/lab/locator.py"}
LEGACY_DIRS = ("logs/",)
LEGACY_FILES = {
    "checkpoints/mathnative_19m_mw1_theta0.json",        # receipt-like artifact of MATHWORLD-1 theta0 (interpreter path recorded at run time)
    "docs/handoffs/2026-09-09-0-writer-dfa-1-unstable.md",
    "docs/sol/NOTES.md", "docs/sol/RESULTS-SOL.md", "docs/sol/SESSION-NOTES-PRESENT-1.md",
    "docs/superpowers/plans/2026-08-01-sol-answer-only-gate.md", "docs/superpowers/plans/2026-08-12-phase3-gate-shim.md",
    "docs/superpowers/plans/2026-08-12-phase6-7.md", "docs/superpowers/specs/2026-08-12-code-quality-program-design.md",
    "scratch/manim_feasibility_2026-08-13.md",
}
RESULTS_FROZEN_OFFENDING_LINES = 5      # docs/RESULTS.md at the scrub commit; append-only, so this may never grow


def _tracked():
    r = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True)
    return [p for p in r.stdout.split("\0") if p]


def _offending_lines(rel):
    p = ROOT / rel
    try:
        text = p.read_text(errors="replace")
    except (OSError, UnicodeDecodeError):
        return []
    return [i + 1 for i, line in enumerate(text.splitlines()) if PATTERN.search(line)]


def test_no_machine_local_paths_in_tracked_files():
    bad = {}
    for rel in _tracked():
        if rel in SELF or rel in LEGACY_FILES or rel.startswith(LEGACY_DIRS) or rel == "docs/RESULTS.md":
            continue
        if Path(rel).suffix in {".pt", ".bin", ".png", ".gif", ".mp4", ".npy", ".npz", ".jsonl"} and rel.startswith("data/"):
            continue
        lines = _offending_lines(rel)
        if lines:
            bad[rel] = lines[:5]
    assert not bad, "machine-local home paths in tracked files (use llmopt.lab.locator):\n  " + "\n  ".join(f"{k}: lines {v}" for k, v in sorted(bad.items()))


def test_results_ledger_offending_lines_do_not_grow():
    n = len(_offending_lines("docs/RESULTS.md"))
    assert n <= RESULTS_FROZEN_OFFENDING_LINES, f"docs/RESULTS.md now has {n} lines with machine-local paths (frozen at {RESULTS_FROZEN_OFFENDING_LINES}); cite a locator instead"


def test_legacy_allowlist_entries_still_exist_and_still_offend():
    """The allowlist is narrow: an entry that no longer offends must be removed from it."""
    tracked = set(_tracked())
    stale = [f for f in LEGACY_FILES if f not in tracked or not _offending_lines(f)]
    assert not stale, f"allowlist entries that no longer need it: {stale}"
