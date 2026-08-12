"""Tests for the wsl.sh permission helper and its argument checks.

CONTEXT: scratch/wsl.sh hands training jobs to the lab's second
computer — Artin's Windows desktop with the 3080, his own machine on
his own network, same repo checked out twice. `.claude/hooks/
wsl_guard.py` decides which of those jobs run without prompting him
(reads: yes) and which are worth showing first (anything that changes
state there). Neither file is defending against an intruder; both
exist so a command this lab composes programmatically cannot ruin
Artin's own work through a typo or a bad loop variable.

What these tests pin:
  1. the decision table — which verbs run unprompted, which prompt,
     and the three shapes refused outright;
  2. that an "allow" is never issued for a command that is more than
     the single wsl.sh call examined (an allow covers the WHOLE Bash
     command, so a trailing safe verb must not cover what precedes
     it);
  3. that wsl.sh refuses malformed arguments locally, before any
     connection is attempted.

Fixtures below use harmless stand-ins (touch, /etc/hostname) — the
property under test is "unexpected characters are refused", and the
target does not need to be dramatic to prove it.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / ".claude" / "hooks" / "wsl_guard.py"
WSL = ROOT / "scratch" / "wsl.sh"


def decide(command):
    """-> permissionDecision, or None when the helper abstains."""
    payload = json.dumps({"tool_name": "Bash",
                          "tool_input": {"command": command}})
    r = subprocess.run([sys.executable, str(GUARD)], input=payload,
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    if not r.stdout.strip():
        return None
    return json.loads(r.stdout)["hookSpecificOutput"]["permissionDecision"]


# The full decision table. This is the contract: edits to the helper
# (including comment-only rewrites) must leave every row unchanged.
DECISIONS = [
    # reads — run without interrupting
    ("allow", "scratch/wsl.sh tail logs/microstar/run.log 10"),
    ("allow", "scratch/wsl.sh check train_mathnative"),
    ("allow", "scratch/wsl.sh clean-marker logs/microstar.DONE"),
    ("allow", "scratch/wsl.sh mkdir logs/microstar"),
    ("allow", 'scratch/wsl.sh run "ls -la logs/; git status --short"'),
    ("allow", "scratch/wsl.sh run 'ls -t logs/*.log | head -5'"),
    ("allow", "scratch/wsl.sh run 'git log --oneline -1'"),
    # pgrep self-match WITHOUT a kill acting on it: a status peek that
    # lists an extra pid, not a hazard (narrowed 2026-08-11)
    ("allow", "scratch/wsl.sh run 'pgrep -af metallicity1; "
              "tail -4 logs/metallicity1/driver2.log'"),
    # changes the remote box — show Artin first
    ("ask", "scratch/wsl.sh kill merge_space"),
    ("ask", "scratch/wsl.sh launch 'bash x.sh' logs/a.log logs/a.DONE"),
    ("ask", "scratch/wsl.sh run 'python train.py > out.log'"),
    ("ask", 'scratch/wsl.sh run "rm -f logs/x; pkill -f foo"'),
    ("ask", "scratch/wsl.sh run 'git reset --hard origin/main'"),
    # effect not readable from the text
    ("ask", 'scratch/wsl.sh run "ls $(touch /tmp/side_effect)"'),
    ("ask", 'scratch/wsl.sh run "ls `hostname`"'),
    # no rule covers it: show rather than guess
    ("ask", "scratch/wsl.sh run 'nvidia-smi'"),
    # unrecoverable — refuse outright
    ("deny", "scratch/wsl.sh run 'rm -rf ~'"),
    ("deny", "scratch/wsl.sh run 'git push --force origin main'"),
    ("deny", "scratch/wsl.sh run 'dd if=/dev/zero of=/dev/sda'"),
    # not our business
    (None, "ls -la"),
    (None, "git status"),
]


@pytest.mark.parametrize("expected,cmd", DECISIONS)
def test_decision_table(expected, cmd):
    assert decide(cmd) == expected, f"decision changed for: {cmd}"


# An allow covers the entire Bash command, so it may only be issued
# when the entire command is the one wsl.sh call that was examined.
NOT_A_BARE_CALL = [
    "rm -rf ~/important && scratch/wsl.sh tail logs/x.log",
    "scratch/wsl.sh tail logs/x.log; rm -rf ~/important",
    "scratch/wsl.sh clean-marker logs/x.DONE && rm -rf ~",
    "scratch/wsl.sh mkdir logs/x `touch /tmp/side_effect`",
    'scratch/wsl.sh run "ls $(rm -rf ~)"',
]


@pytest.mark.parametrize("cmd", NOT_A_BARE_CALL)
def test_no_allow_when_more_than_one_call(cmd):
    assert decide(cmd) != "allow", f"helper approved a compound command: {cmd}"


def test_self_match_pattern_is_flagged():
    """A pkill -f pattern that also appears elsewhere in the command
    matches the job's own argv and kills it mid-run (cost a night in
    July, hence the check)."""
    assert decide(
        "scratch/wsl.sh run 'pkill -f trainer; pgrep -af trainer'") == "ask"


def test_self_match_needs_a_kill_to_escalate():
    """pgrep alone matching itself is a read; the same shape piped to
    xargs kill is the July hazard and still asks."""
    assert decide(
        "scratch/wsl.sh run 'pgrep -f trainer | xargs kill; "
        "tail logs/trainer.log'") == "ask"


# wsl.sh's own argument checks, verified locally: these must fail
# before any connection is attempted.
MALFORMED = [
    ("clean-marker", "logs/x;touch /tmp/oops.DONE"),
    ("clean-marker", "logs/../../../etc/hostname.DONE"),
    ("clean-marker", "logs/$(whoami).DONE"),
    ("kill", "x'; touch /tmp/oops; echo '"),
    ("kill", "foo`id`bar"),
    ("mkdir", "logs/x; touch /tmp/oops"),
    ("mkdir", "/etc/somewhere"),
    ("mkdir", "../../escape"),
]


@pytest.mark.parametrize("verb,arg", MALFORMED)
def test_wsl_sh_refuses_malformed_arguments(verb, arg):
    """Refused LOCALLY (rc=3), no connection attempted."""
    r = subprocess.run(["bash", str(WSL), verb, arg],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 3, (
        f"{verb} {arg!r} was not refused (rc={r.returncode}): "
        f"{r.stdout}{r.stderr}")
    assert "refuse" in r.stderr
