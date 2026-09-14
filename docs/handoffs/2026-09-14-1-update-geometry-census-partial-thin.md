# Handoff 2026-09-14-1: UPDATE-GEOMETRY-CENSUS-0 booked PARTIAL-THIN / WRITER-SHARED / LEARNED-DIFFERS-DIFFUSED; census stopped as sealed

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run. Nothing armed.

## What landed (after handoff 2026-09-14-0)

- Artin decisions 17:02 EDT: CROSSFOSTER stays PARKED under the booked
  revival condition; chain.json moved to the untracked sha-anchored
  stream (AMENDMENT L73144, c0ea6281).
- Artin note: cybernetics.bio is a domain name only; no bio/chem
  capability, ever (charter unchanged; memory saved).
- Artin GO 17:22 EDT: UPDATE-GEOMETRY-CENSUS-0 (design + prereg +
  implementation + test/smoke + zero-training census; the house
  included the census run after a clean auditor).
- PRE-REG (RESULTS L73187, 71122a1b): Opus prereg audit, two blockers
  folded (four-memmap peak; a wrong m015300 provenance sentence).
  Instrument scratch/update_geometry_census.py (+ launcher, 9 tests,
  non-target seed-6 smokes, clean-tree seal smoke at 62ce288d).
- Run ugc0 at b108cbfe: 6.6 min (the cost paragraph over-priced it by
  an order of magnitude), rc 0, all 15 cells parity <= 4.4e-13, all
  digests asserted, memmaps deleted.
- VERDICT (RESULTS L73446, this commit): RESOLVED / PARTIAL-THIN /
  RESIDUAL-INDETERMINATE / WRITER-SHARED / LEARNED-DIFFERS-DIFFUSED.
  Top-8 FIT subspace reproduces on held batches (S_8 0.39 / 0.40) but
  captures 0.25 / 0.27 of a held gradient (0.37 at k = 16; five orders
  above random); centering is a no-op at the finals (Q 0.02); writers'
  subspaces on the same batches overlap 0.645 (same-writer rerun
  0.88); W_0 is one descent direction (C_8 0.94) that training
  diffuses (PR 1.7 to 14); the finals' capture lives in embedding /
  head / norm (77 to 80 % of gradient energy, 31,104 params) while
  blocks read 0.005 to 0.03; checkpoint velocity lies almost entirely
  outside the top-16 gradient subspace. Prior 4 hits 7 misses (family
  46 / 31). Both auditors ran before booking (prereg: two blockers
  folded, a wrong parameter count and a headline velocity bound;
  receipt: no blocker). RIFF bank TASK-GRADIENT GEOMETRY added with
  the measured anchors; FINDINGS bullet; BOARD line 5.
- Consequence as sealed: the census STOPPED; no projected-gradient
  rung licensed. Banked unarmed: PROJECTED-BP-CAUSAL-1 (its condition
  did not fire globally; an OUTSIDE-weighted target would need its
  own pre-reg), DATA-LOCUS-GEOMETRY-1 (instrument validated),
  FAILED-WRITER-PROJECTION-DESK-0.

## Also landed (Artin GO 18:15 EDT): PATH-HYGIENE SCRUB

NOTE PATH-HYGIENE-SCRUB-0 (RESULTS, the scrub commit): llmopt/lab/
locator.py adopted (roles main / repair / axiom; env or git worktree
resolution); every active instrument, shell script, pre-reg and the
BOARD scrubbed of machine-local home paths; scripts/liverun.py emits
worktree role + repo-relative cwd; tests/test_path_hygiene.py is the
CI lint (allowlist: logs/ receipts, RESULTS at a frozen 5 lines, ten
legacy docs). Frozen-cited instruments changed bytes (disclosed; their
receipts pin launch commits). No verdict changed.

## Conditions that bite next session

- New instruments name artifacts by locator, never by an absolute
  path; the lint is red otherwise. Windows-box paths come from
  scratch/remote.env.sh (AXIOM_WSL_QUAL for the poly pipelines).

- Nothing armed. Foreign-writer and CROSSFOSTER programs parked.
- The instrument is reusable as is (scratch/update_geometry_census.py:
  frozen probe law, Gram algebra, ladder); any new use is a new
  pre-reg (different specimens, k, or thresholds).
- Ledger-fold discipline: a fold script that asserts mid-way leaves
  the draft unfolded; the booking script must run only after the fold
  script exits 0 (bit once this session, reverted before commit).
- Disk about 31 GiB free; logs/ugc0 holds 1 MB of receipts.

## Open decisions for Artin

1. Whether any of the three banked geometry follow-ups gets a GO
   (each is a new pre-reg); the house's reading is that the global
   gradient geometry is not thin enough to license PROJECTED-BP-
   CAUSAL-1 as framed, and that DATA-LOCUS-GEOMETRY-1 is the cheapest
   informative next use of the instrument.

## Next session: where to start

This handoff, BOARD line 5, RESULTS tail from L73446, the RIFF bank
TASK-GRADIENT GEOMETRY (2026-09-14).
