---
name: book
description: Book a verdict/pre-reg/amendment into RESULTS.md the house way - append entry, regenerate index, link threads, commit+push with the public-repo trailer. Use whenever a result lands or an experiment is pre-registered.
---

# Booking a result (the house ritual)

Given the entry text (from the conversation) and its thread/link
metadata, perform ALL of these steps in order:

1. **Append** the entry to `docs/RESULTS.md`. Append-only — never
   edit past entries; corrections are new `AMENDMENT` entries naming
   their target. Heading format:
   `## VERDICT|PRE-REG|AMENDMENT <NAME>: <one-line claim> (<date>, <machine>)`
2. **Regenerate the index**: `.venv/bin/python scripts/gen_results_index.py`
   (a PostToolUse hook may have already run it — running twice is
   harmless; it preserves curation).
3. **Link**: for any `needs_link` rows in `docs/results-index.jsonl`,
   set `threads` (kebab-case program names) and `links` (related
   entry names), pop `needs_link`, rewrite the file.
4. **Commit and push** `docs/RESULTS.md docs/results-index.jsonl`
   (plus any script the entry references) with a one-line message
   summarizing the verdict. PUBLIC REPO: end the message with
   exactly `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
   and NEVER a Claude-Session URL.

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
