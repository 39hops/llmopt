"""Regression tests for the commit-guard hookify rules.

These rules exist because the same mistake shipped four times
(2026-08-07, 08-10, and twice on 08-11): a test suite's exit code is
discarded by a pipe, and a commit that looks gated is not.

The 08-11 pair got through because `block-pipe-gated-commit`'s
pattern allowed exactly ONE `&&` between the pipe and the commit,
while real command chains carry three or four. These tests pin the
widened patterns against the exact strings that escaped, so a future
tightening cannot silently re-open the hole.

Test strings live here rather than in a shell command on purpose:
the live hooks (correctly) block any Bash call containing them.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

RULES = Path(__file__).resolve().parents[1] / ".claude"

# the two commands that actually shipped red commits on 2026-08-11
ESCAPED_1 = (".venv/bin/pytest tests/ -q 2>&1 | tail -1 "
             "&& .venv/bin/python scripts/gen_index.py > /dev/null "
             "&& git add -A llmopt/lab && git commit -q -m 'msg'")
ESCAPED_2 = (".venv/bin/pytest tests/test_docs_integrity.py -q 2>&1 | tail -1\n"
             "rc=$?; if [ $rc -eq 0 ]; then git commit -q -m 'msg'; fi")

# shapes that must stay allowed — a guard that blocks the correct
# pattern trains people to work around it
SAFE = [
    "pytest -q > /tmp/o.txt 2>&1; rc=$?; tail -2 /tmp/o.txt\n"
    "if [ $rc -eq 0 ]; then git commit -m 'msg'; fi",
    "pytest -q && git commit -m 'msg'",
    "pytest -q | tail -3",
    "git commit -m 'msg'",
]


def _pattern(rule_name: str) -> re.Pattern:
    """Read the live pattern out of the rule file's frontmatter.

    Hookify rules are `.local.md` and gitignored by convention, so they
    exist on a working machine and not on a clean checkout. Skip rather
    than fail when absent: these tests protect the rules wherever the
    hooks actually run, which is the only place they can fire.
    """
    path = RULES / f"hookify.{rule_name}.local.md"
    if not path.exists():
        pytest.skip(f"{path.name} not present (gitignored local rule)")
    text = path.read_text()
    m = re.search(r"^pattern:\s*(.+)$", text, re.M)
    assert m, f"no pattern in {path}"
    enabled = re.search(r"^enabled:\s*(\w+)$", text, re.M)
    assert enabled and enabled.group(1) == "true", f"{rule_name} disabled"
    return re.compile(m.group(1).strip())


@pytest.mark.parametrize("rule,command", [
    ("pipe-gated-commit", ESCAPED_1),
    ("rc-after-pipe", ESCAPED_2),
])
def test_guard_catches_the_command_that_escaped(rule, command):
    assert _pattern(rule).search(command), (
        f"{rule} does not match the 2026-08-11 command it exists to stop")


@pytest.mark.parametrize("rule", ["pipe-gated-commit", "rc-after-pipe"])
@pytest.mark.parametrize("command", SAFE)
def test_guard_allows_correct_shapes(rule, command):
    assert not _pattern(rule).search(command), (
        f"{rule} false-positives on a correctly gated command")
