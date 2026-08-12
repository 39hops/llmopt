---
name: handoff
description: Session-close ritual - write the dated handoff, repoint BOARD, run the suite on a real exit code, commit and push. Use at natural stopping points, before any compact, or when Artin says "hand off", "wrap up", "get to a good spot".
---

# Closing a session (the handoff ritual)

State lives in the REPO, never in memory. A session's working state
must never be the single point of failure. Do ALL steps, in order.

## 1. Sweep for unbooked work

Anything measured but unbooked? `/book` it first — a handoff that
says "result X landed, not booked" is the backlog being born. Check
`/labstatus` if runs were live this session.

## 2. Write the handoff file

`docs/handoffs/YYYY-MM-DD-N-<kebab-slug>.md` — N is the day's
0-indexed sequence (`ls docs/handoffs | grep <date>` first).

Required content, in rough order:
- Seat line: model, machine, HEAD at close, 3080 window state.
- What landed: each verdict/amendment with its commit sha and the
  one-line claim. Honest failures included.
- Conditions that bite next session (ratchet headroom, README
  drift, anything armed).
- Next session: where to start (this handoff, then BOARD, then
  RESULTS tail), and the recommended first batch.
- Open decisions for Artin, numbered.
- Also standing: relays unsent, runs finished-unbooked, banked work.

## 3. Repoint BOARD

`docs/BOARD.md` line 3 points at the newest handoff. Refresh any
LIVE/BANKED rows the session changed. Living-docs check: does a
verdict from this session touch THEORY.md or RIFF-LEDGER.md rows?
Update in the same commit.

## 4. Verify — real exit code, never piped

```bash
.venv/bin/python -m pytest -q > /tmp/pytest_close.log 2>&1; rc=$?
tail -3 /tmp/pytest_close.log; echo "PYTEST_RC=$rc"
```

Never gate the commit on piped pytest — rc must break the chain
(fired 2026-08-07). Red suite = fix or book the redness honestly in
the handoff; never push a silent red.

## 5. Commit and push

Handoff + BOARD (+ living-doc updates) in one commit:
`handoff YYYY-MM-DD-N: <one-line session summary>`.
PUBLIC REPO: end with exactly
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`, NEVER a
Claude-Session URL. Then verify push: local and remote HEAD match.

## 6. Report

Close with: commands run, verification status, HEAD sha, what bites
next session. The Stop hook wants exact commands and a pass/blocked
verdict — write it that way the first time.

## Gotchas that earned this skill

- Closeout blocked 3x on 2026-08-12-adjacent session: stale
  verification evidence, missing "Commands run:" block, mixed
  passed/failed verdict. Fresh verification, explicit commands,
  one honest status.
- A handoff BEFORE compact is mandatory, not optional — post-compact
  resume = resume-protocol memory + BOARD + newest handoff + RESULTS
  tail.
- Untracked big jsonl/checkpoints stay untracked (file-handoff
  convention); do not "clean up" by adding them.
