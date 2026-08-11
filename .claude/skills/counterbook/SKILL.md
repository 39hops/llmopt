---
name: counterbook
description: Use when the axiom lab (the sibling C++/exact-arithmetic repo Artin relays with) ships a result, receipt, census, or relay — or the user says "counter-book", "verify axiom's numbers", "axiom pushed something", "their relay landed". Recomputes their numbers from their artifacts (never accepts their tables), books house-side, and writes the reply relay.
---

# Counter-booking an axiom result

The verification standard (set 2026-08-10): re-derive from THEIR
commits and artifacts, never accept tables. Order matters.

## Steps

1. **Pull**: `git -C /Users/artin/code/axiom pull --ff-only`; note
   the commit hash — the booking cites it as the frozen evidence.
2. **Recompute every load-bearing number** from the committed
   artifact (jsonl, dumps, source): schedules re-evaluated from
   their diffs, counts re-counted, splits re-bucketed. House-side
   strengthenings from their own data (e.g. an uncharted mid-zone)
   book as house findings.
3. **Read against the pre-reg as written**: which bars FIRE / MISS /
   NOT-APPLICABLE; refutation clauses read literally; scope fences
   (coverage bounds, (budget,path)-scoping) stated BEFORE numbers.
4. **Book** (use the `book` skill): heading names the verdict class;
   per rule 4 report every event in a class then explain; per rule 3
   explanations of constants book at their own evidence level; name
   estimators; fit the artifact never a printed summary.
5. **BOARD row** updated same commit.
6. **Reply relay** (use the `relay` skill) citing the new RESULTS
   line.
7. **Commit rc-gated**: `pytest tests/test_docs_integrity.py` with
   the REAL exit code breaking the chain — never a piped rc
   (`cmd > /dev/null 2>&1; rc=$?` then branch). Small text receipts
   may be `git add -f`'d under `logs/<name>/` (seedslad pattern)
   with the why booked in the entry.

## Checks that have caught real defects

- Does a retracted/corrected figure of theirs appear anywhere in
  OUR ledger? grep RESULTS/BOARD/FINDINGS/relays before assuming.
- Is a constant count structural or truncated? Census the call
  sites in their source (the two 64s lesson: same digits, opposite
  evidence class).
- FINDINGS curation ratchet: run the docs test after booking; if
  over 320, curate in the same sitting.
