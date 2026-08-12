# Handoff 2026-08-12-1 — quality-foundations executed: 13 tasks, 15 commits, .git 47 GB -> 203 MiB

Seat: Fable 5, Mac. HEAD at close: `86fd2e9` (plus this handoff's
commit). Both checkouts in lockstep (3080 ff-only synced 62e4352 ->
86fd2e9 at ~14:00 EDT). 3080 idle; nothing queued. pytest at close:
658 passed / 7 skipped, rc=0. ruff (tiered) exit 0.

## What landed (no experiments — pure engineering; nothing to book)

The plan `docs/superpowers/plans/2026-08-12-quality-foundations.md`
(spec Phases 0, 1, 2, 7b, 5b-pass) executed end-to-end by
subagent-driven development: 13 tasks, each implemented by a fresh
subagent and reviewed by an independent one, two fix rounds total,
final whole-branch review CLEAN. Actuals postscript appended to the
spec (commit `86fd2e9`).

- **Ratchet** (`e1ff62d`): 20 oldest entries curated into FINDINGS,
  `MAX_UNCURATED` 320 -> 300. Cap == backlog: every booking now adds
  its FINDINGS bullet in the same commit or CI reds. The
  `findings_headroom` hook warns at RESULTS-edit time.
- **`.git` reclaim** (no commit — ops): 23.48 GiB loose + 23.10 GiB
  packs -> ONE pack, 203.26 MiB. Artin ran the gc himself (hook
  gates destructive git from the model — correct behavior). fsck
  clean before and after; 6/6 branches; all commits retained.
- **CODEMAP truth** (`d98b4e9`, `ee2db8c`): `cited by` column, then
  refs split into imports/mentions — `library` now requires a real
  import. library 165 -> 62; the 113 cited-but-library files now
  show their citations; `gate_ckpt.py` correctly `results-cited`
  (the edit-guard hook now asks on it).
- **`/tmp` probes materialized** (`507295e`, `edc1a01`):
  `scratch/frozen_products/` holds the exact programs the five
  sed-patching drivers executed, with provenance headers. All three
  g19 substitutions verified still biting — the booked g19 numbers
  ran on the source we think they did. Drivers untouched (evidence).
- **Provenance index** (`2d225c1`, `59ec47a`, `4c85c6e`): every
  ledger row carries `files` (auto-extracted) and `code_commit`
  (880/930 populated, 50 honest nulls). Rule is FILE-AWARE after a
  plan defect was caught by Task 8's hand-verification: parent sha
  only if every cited file exists there, else the booking commit,
  else null (Artin adjudicated). `results_query.py --repro <id>`
  prints the exact worktree command.
- **Guardrails** (`f20b08b`, `c33a856`): ruff tiered (F-rules on
  llmopt/tests/scripts, scratch report-only; found one REAL bug —
  `llmopt/context/gist.py` F821, fixed with the TYPE_CHECKING
  pattern); layering test (scripts never imports scratch); `-m docs`
  marker splits ledger guards from code tests.
- **Generated README** (`6d089fa`): honesty-ledger counts are a
  generated region, `gen_readme.py --check` in CI. README said 187;
  ledger truth is 217 (38/54/82/39/4). The number is derived from
  the same grammar the integrity test enforces.
- **Voice pass** (`b466719`): THEORY/REPRODUCE/FINDINGS first-person
  and "the house" removed per spec §5b; rule added to CLAUDE.md.
  All 314 FINDINGS anchors byte-identical.
- **Closeout** (`86fd2e9`): spec actuals postscript + BOARD row.

Earlier same session (pre-plan): tooling commit `576f779` (codemap
edit-guard hook, findings-headroom hook, /handoff + codemap-check
skills, spec §5b extended with Artin's front-facing style rules) and
the plan itself (`c72bef6`).

## Conditions that bite next session

1. **Ratchet is cap==backlog (300/300) BY DESIGN.** The next booking
   MUST add its FINDINGS bullet in the same commit. The plan
   promised >=20 slack; the curate-and-lower rule produced the
   stricter regime instead — deliberate, but say so to Artin if the
   friction is wrong.
2. **First booking under the new /book flow**: record `code_commit`
   (launch sha via /rung, or parent-if-files-exist rule) on the new
   index row — the skill now says how.

## Next session: Phase 3 module 1 (keepsets shim)

Start: this handoff -> BOARD -> spec §Phase 3. Write its own plan
(writing-plans over the Phase 3 section only). `keepsets` first
because `tests/test_lab_keepsets.py:98-134` already pins booked
numbers byte-exact (0.8013/0.5331/0.5280). Procedure per spec: move
body -> shim original -> delete source-identity guard same commit ->
battery green = proof. One module per commit, never more.

## Open decisions for Artin (spec §9, still open)

1. Device precedence for `pick_device()` (Phase 4).
2. `llmopt/common/` vs `lab/` for shared helpers (Phase 4).
3. Phase 5 `lab/` split timing (BOARD housekeeping gate?).
4. New this session: `house-crystal` term — lab-wide rename or keep.
5. THEORY L4: Artin's quote is now a paraphrase ("must show proof,
   not vibes") — revert to verbatim quote if preferred.

## Also standing

- Relay 2026-08-11-1 remains DRAFT/UNSENT — Artin sends manually.
- llmopt_dump on WSL verified: copy not move (originals intact on
  Mac, 3/3 hashes match), `data/corrupt/` is AppleDouble sidecar
  junk (536 KB, deletable). Substrate for the lake/sorting idea.
- Follow-ups parked by the final review: three FINDINGS parsers
  (CI-fenced; consolidate during Phase 3+), three new tests lack the
  docs marker, README-check CI step placement cosmetic.
- Banked/unqueued (unchanged): BASIN-CENSUS-1, Q9 births + 1/2/4/8
  ladder pre-reg, ignition-mass cell B, GROW-DECOMP n=3, MPS decay
  probe arms, excluded-experts anatomy, GT-7 candidate.
- v4anat finished clean earlier (rc=0), exploration-grade, unbooked.
