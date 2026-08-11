---
name: book
description: Book a verdict/pre-reg/amendment into RESULTS.md the house way - append entry, regenerate index, link threads, commit+push with the public-repo trailer. Use whenever a result lands or an experiment is pre-registered.
---

# Booking a result (the house ritual)

Given the entry text (from the conversation) and its thread/link
metadata, perform ALL of these steps in order:

1. **Append** the entry to `docs/RESULTS.md` — it is 27k lines, so
   append with a `cat >> ... << 'EOF'` heredoc, never Read-then-Write.
   Append-only: corrections are new `AMENDMENT` entries naming their
   target. Heading:
   `## VERDICT|PRE-REG|AMENDMENT <NAME>: <one-line claim> (<date>, <machine>)`
2. **Regenerate the index**: `.venv/bin/python scripts/gen_results_index.py`.
   Always run it yourself. The PostToolUse hook matches `Edit|Write`
   only, so a heredoc append does NOT trigger it.
3. **Link**: `grep -n needs_link docs/results-index.jsonl` to find the
   new rows, then patch them in place with
   `llmopt.lab.jsonl.read_jsonl` / `write_jsonl` — set `threads`
   (kebab-case program names) and `links` (related entry ids, not
   titles), and pop `needs_link`. Never re-emit the file from context.
   Verify: `grep -c needs_link` returns 0, and
   `scripts/results_query.py --chain <new-id>` shows the entry.
4. **Curate FINDINGS in the same commit** when the entry is a verdict.
   `tests/test_docs_integrity.py` ratchets the uncurated backlog, so a
   booking without its bullet turns the suite red — that is the guard
   working. One bullet, one maturity tag from the controlled
   vocabulary, its scope fences, and the `RESULTS.md#L<line>` anchor.
5. **Commit and push** `docs/RESULTS.md docs/results-index.jsonl`
   `docs/FINDINGS.md` (plus any script the entry references) with a
   one-line message summarizing the verdict. Gate the commit on a real
   exit code — read `$?` from a redirected pytest, never from a piped
   one. PUBLIC REPO: end the message with exactly
   `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and NEVER
   a Claude-Session URL.

## Fences that travel with every booking

- Pre-register BEFORE the run fires; book verdicts (including
  honest failures) the moment they land.
- Resolution law (2026-07-31): gate deltas < 1.5 sigma (< ~7 solves
  on the 120 gate) need n>=3 paired seeds before a direction is
  claimed; single-seed readings get an explicit fence sentence.
- Cross-device comparisons are forbidden for fp gates; the
  deterministic integer battery is its own instrument (pooling
  legal within it only).
- If the verdict touches a THEORY row or a RIFF-LEDGER bank,
  update those in the same commit (living-docs discipline).

## Checksums learned the hard way (2026-08-01 session)

- **Gate numbers book as DICTS, not totals** — quote the per-level
  solves dict and verify it SUMS to the claimed total before the
  entry is written. The one number in a gate line that is NOT a
  solve count is the validity float; a "48" was booked from
  `valid 48.27` once and survived TWO review passes. The dict is
  the checksum.
- **Amendments set their `amends` target in the index in the same
  commit** — an amendment without a target is the needs_link
  backlog being born.
- **Before booking any constant** (GBOOST, SHIFT, Q, lr), grep the
  SHIPPED script for its current value — a diagnostic edit that
  was never reverted becomes the shipped value silently (GBOOST
  64->256 was booked as 64). What the artifact/JSON records
  governs; the prose must match it.
- **Provenance line**: a gate booking quotes the `weights sha`
  line that gate_eval now prints (dtype-sensitive — never compare
  shas across precisions).
