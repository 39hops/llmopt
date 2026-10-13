# Rehabilitation phase 2B and onward: implementation plan after the integrity corrections

Status: PLAN for Artin. Nothing armed. Baseline: the 2026-10-13
engineering audit (outside the repo) and phase 2A (commits d7975baa,
fcbf54a3, 8a499562, 3f0d6cfb on main). Every phase below is docs/code
plus tests; none trains, downloads, deletes or rewrites history. The
destructive items (storage, checkpoint triage) are listed only to fix
their order and their GO requirement.

Phase boundaries follow the audit's P-items (08-rehabilitation-plan).
Each phase is one reviewable commit set with acceptance criteria that a
watcher can check from artifacts, never from the claim.

## 2B. Figure-generation determinism and `--check` (P1-H, P0-G)

Problem: `scripts/gen_figures_web.py` rewrites `docs/figures.json` with
`ensure_ascii=True`; `scripts/gen_readme.py` rewrites it with
`ensure_ascii=False`; run order is enforced only by prose; the figure
renderer has no `--check`; PNG capture silently skips without Chrome;
48 transcribed figure values are unguarded; fence anchors carry no ids;
the merge_space fence omits MERGE-SPACE-3/-4 (two of its seven values).

Change:
1. One serializer for figures.json (the gen_readme form), used by both
   generators; `gen_figures_web.py --check` renders to memory and
   compares SVG bytes; PNG staleness (PNG older than its SVG, or absent)
   is reported as a failure in check mode, with a documented skip
   variable for machines without Chrome.
2. `check_source.sh` runs `gen_figures_web.py --check`.
3. Fences gain `id:` entries; the merge_space fence names -3 and -4;
   `test_docs_integrity` parses fences as ledger links.
4. A numeric re-read test for the table-shaped entries (loss_floor,
   effective_context, routing_crest seed lines) against RESULTS.
5. The 66 versus 66.33 full-model mean in routing_crest is stated in
   the fence (rounding), not changed.

Acceptance: running the two generators in either order leaves
`git status` clean; the six published SVG/PNG pairs are byte-identical
before and after except for the merge_space fence text; `--check` for
both generators in CI; new tests fail when a figure value is edited by
one digit (mutation check by hand, recorded in the commit message).

## 2C. Generator correctness guards, no seed changes (P0-E, P0-F, P1-K, P1-L)

Problem: `llmopt/mathgen/problems.py:make_dataset` loops forever on an
exhausted cell; its per-item seed `seed * 1_000_000 + i` and the
`seed + 1_000_003` retry sites alias neighbouring seed bands, so only
`exclude=` separates train from eval; sympy makers run unboxed inside
it; `meter()` can emit nan kurtosis on a constant tensor;
`llmopt/reproduce.py` validates no operand and hand-maintains
`CONTRACT_ENV`.

Change (guards and tests only; the seed arithmetic is frozen because
changing it silently alters every generated dataset and every pin):
1. Iteration cap in `make_dataset` (for example `50 * n` attempts)
   raising a named error carrying kind, level, n and the exclude size.
2. A pinned digest of `make_dataset(200, seed=0)` over the default kinds
   and levels, recorded BEFORE the change and asserted after.
3. `exclude=` as the enforced contract: a split helper that builds eval
   first and passes its prompts; a metamorphic disjointness test; a
   caller census of the unguarded `make_dataset` sites with their role
   (gate versus train) booked as an OBSERVATION. No call-site behaviour
   changes in this phase.
4. Fork-boxed sympy inside the makers that can hang (the
   `solve_isolated` pattern), with a boundary test that a deliberately
   slow maker is killed and reported, never silently dropped.
5. `meter()`: constant and near-constant tensors return a finite,
   documented value (kurtosis 0 with a flag, or an explicit exception);
   boundary tests for empty, constant, single-row and all-equal inputs.
6. `reproduce.py`: `is_file()` checks with the repo-bound message; an
   AST test that every `os.environ` key the runner reads is in
   `CONTRACT_ENV`.

Acceptance: the pinned `make_dataset` digest is identical before and
after; exhaustion raises within 2 s on a 1-kind, 1-level, exclude-all
call; the disjointness test runs 200 cases; `meter` tests pass on the
four boundary inputs; the sixteen reproduce pins still PASS (one serial
replay, about 23 minutes, run once at the end of the phase, log kept
under `logs/rehab-2c/`).

## 2D. Adjudicator fence and negative controls (P1-J, P1-P)

Problem: `llmopt/lab/metrics.py:adjudicate` and `prereg.py` have no
fence representation; the house resolution law (deltas under about 5
solves on the 120 gate are unresolved) lives in prose; the adjudicator
tests replay successful historical verdicts only.

Change:
1. Optional `fence` field on a bar; `|observed - bar| < fence` yields
   UNRESOLVED; absent field keeps today's behaviour exactly.
2. Negative-control fixtures: a malicious prereg (bar edited after the
   observations), a mismatched prereg (observations keyed to a different
   bar name), an invalid prereg (direction neither above nor below,
   missing results_id), a fence-band observation, and a prereg whose
   receipts list a path the lock has never seen. Each must fail with a
   typed error or UNRESOLVED, never FIRE.
3. The gate seed-collision count (`llmopt/lab/gate.py:129-131`,
   problem x ply x slot) computed and booked as an OBSERVATION; no code
   change to the frozen lineage; GLOSSARY gate entry gains the note.

Acceptance: re-running the 20 JSON-preregistered adjudications gives
byte-identical output when no bar declares a fence; every negative
fixture produces its named outcome; the observation is booked with the
arithmetic in the entry.

## 2E. Checkpoint classification and physical storage analysis (P2-R prep, non-destructive)

Problem: `checkpoints/` holds about 70 GB; 36.5 GB in 691 files is cited
by no verdict, lock, handoff or spec; 2.35 GB is byte-duplicate evidence
under cited paths; 13 GiB free blocks the 30B vehicle download.

Change (print-only; no delete, no move):
1. A classifier (extend `scripts/log_hygiene.py --tree checkpoints`)
   that assigns FROZEN / ESSENTIAL / DUPLICATE / CANDIDATE / UNKNOWN per
   file with every reference found (RESULTS, lock, EVIDENCE, handoffs,
   specs, scratch path literals).
2. Physical APFS analysis: for every duplicate group, report whether the
   copies share blocks (clones) or occupy separate blocks. Logical
   duplicates (same sha256) reclaim nothing if they are already clones;
   only distinct-block copies are physically reclaimable. Use the
   per-file allocated block count and compare against logical size.
3. A dated triage sheet per family (name, size, newest mtime, every
   reference, proposed action) booked as an OBSERVATION. Artin decides
   per family.

Acceptance: the sheet lists every file under `checkpoints/`; the sum of
"physically reclaimable" bytes is derived from block counts, not from
logical sizes; nothing on disk changed (inventory sha list identical
before and after).

## 2F. Artifact retention and disk reclamation protocol (P2-S, P2-T, P2-R execution)

Requires explicit Artin GO per family and per step; the logs doctrine
and the BOARD housekeeping gate apply. Order: lossless first (APFS
clone the 26 frozen-with-frozen duplicate groups, delete DUPLICATE-class
non-retained copies after confirming the retained copy is the cited
one), then CANDIDATE families per the 2E sheet with
`scripts/ckpt_manifest.py` recording shas before removal.

Acceptance: `gen_receipt_lock.py --check` current;
`test_evidence_availability` green; no RESULTS-cited path deleted; free
space measured before and after and booked.

## 2G. Code and documentation consolidation (P2-U, P2-V, P2-W, P2-X)

`.ep` sentinel meanings unified through `runfiles.write_marker`;
`pick_device` and `lab.hash.sha256_file` adopted in non-frozen package
modules (CODEMAP class checked per file); RIFF-LEDGER and BOARD indexed
the way RESULTS is (`gen_riff_index.py`, `results_query.py --riff`);
`figs/<date>/` untracked going forward; `test_asset_links` extended to
gif/mp4. Regression coverage per touched module; digests produced by
touched sha helpers unchanged on a fixture tree.

## 2H. Mac-native replication (after 2F frees about 40 GB)

Download the 30B vehicle; re-run the crest confirmation at seeds
111/222/333 and the three controls (six gate runs); book as a Mac
replication with its own sampling fence; re-run `scratch/keff_probe.py`
on the floorhk checkpoints and compare within tolerance. One-resident-
30B rule observed; no digit-level identity claimed.

## Standing constraints across every phase

- Never modify booked verdicts, registrations or locked receipts in
  place; corrections are AMENDMENT entries naming their target.
- Seed arithmetic and digest pins are frozen; add guards and tests.
- Check `scripts/liverun.py status` before every commit.
- merge_space artifacts: if recovered from the 3080, force-add the
  originals, shrink `LEGACY_ABSENT`, update EVIDENCE.md; never re-create.
- Do not push from a rehabilitation session unless Artin says so.
