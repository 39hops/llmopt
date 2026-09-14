# Handoff 2026-09-14-2: the three-phase GO landed — locator hardening + /fold-book, the scheduler-law amendment, and OPTIMIZER-GEOMETRY-DESK-0 booked GEOMETRY-NOT-RESOLVED (the write is carried history)

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run. Nothing armed.

## What landed (after handoff 2026-09-14-1)

- PATH-HYGIENE-SCRUB-0 (NOTE, RESULTS L73631, 7f0e140c): locator module
  + lint, no machine-local paths in active instruments; liverun emits
  role + relative cwd.
- Artin GO 18:37 EDT, three phases, hard gates, no birth.
- Phase 0 (4d4ed423): locator fails closed (POSIX / drive / UNC
  absolute, `..`, empty; resolve refuses escapes incl. symlinks;
  repo_relative raises off-root), 12 + 3 tests; `/fold-book` skill +
  scripts/fold_book.py (fold on a copy, rc-gated, lint, checks, then
  book; 4 tests). No needs_link hook (by decision).
- Phase 1 (AMENDMENT WRITER-INTERVENTION-SCHEDULER-LAW, L73710,
  6282f731): stock OneCycle cycles AdamW beta1 0.95 -> 0.85 -> 0.95
  inversely to LR; the backward writer's SequenceLR holds beta1 at 0.9
  for all 15,420 steps; REVERSE=0 is LR-equivalent to rounding
  (1.81e-16) but not optimizer-law equivalent; serialized milestone
  lr / betas equal the reconstructed step-s values exactly (10 / 10).
  Prospective wording adopted and propagated (FINDINGS x2, RIFF, the
  UGC0 pre-reg JSON); BACKWARD-SCHEDULE-1's COMMUTES reading now names
  a compound intervention. Receipt logs/schedaudit0/audit.json is
  untracked sha-anchored (local_only; re-derivable by rerunning the
  audit). Auditor blocker (birth-time torch version unreceipted) and
  five should-fixes folded through /fold-book.
- Phase 2 (PRE-REG OPTIMIZER-GEOMETRY-DESK-0, L73817, 67f5e49e; seal
  smoke 600fa419; run ogd0, 3.5 min): VERDICT (this commit)
  GEOMETRY-NOT-RESOLVED under the sealed gate (raw S_8(FIT, HELD) of the
  full write u 0.13 to 0.25 v floor 0.25). Descriptively: the
  counterfactual AdamW write is ONE common direction per state (PR 1.1
  to 1.7, Q 0.73 to 0.94), the zero-gradient write u_0 is 85 to 96 % of
  the full write's norm, the batch-sensitive remainder b is near-
  random (held C_8 0.01 to 0.08 late, reliability 0.06 to 0.16) and
  less reusable than the raw gradient, the deformation enters at
  g -> first moment (D 1.9 to 3.9) not at the preconditioner (0.02 to
  0.14), the writers' write directions are unrelated (S_8 0.002 to
  0.005) while their gradients overlap 0.5; raw g reproduced UGC0
  exactly at all six cells. Had the gate passed, the chain would have
  read HISTORY-DOMINATED (labeled as not adjudicated). Prior 6 hits 6
  misses (family 52 / 37). Instrument review + prereg audit before
  sealing, prereg audit + receipt audit before booking, all folded
  through /fold-book.

## Conditions that bite next session

- Nothing armed. The GO ended at the booking. OPTIMIZER-MEMORY-
  ABLATION-1 is banked unarmed (RIFF bank TASK-GRADIENT GEOMETRY);
  PRECONDITIONER-CROSS-FOSTER-1 and UPDATE-SUBSPACE-CAUSAL-1 were NOT
  banked (the desk gave no reason). DATA-LOCUS-GEOMETRY-1 and
  FAILED-WRITER-PROJECTION-DESK-0 stay banked. PROJECTED-BP-CAUSAL-1
  is never launched from these results.
- A re-run of the optimizer desk with a resolution rule fit to one-
  directional writes (k = 1 reliability, or a residual-after-first-
  direction panel) would be a new pre-reg; the sealed gate is what it
  is and fired for a real reason.
- Writer descriptions from here on use the corrected scheduler
  wording (LR direction plus beta1 law), never "only the LR direction".
- The mps float32 writer's rounding envelope (1.6e-2 of the update
  scale at step 1, parameter quantization) is receipted in the smokes;
  the virtual law is exact in float64 (4e-11 / 2e-12).
- Disk about 33 GiB free; logs/ogd0 holds 0.3 MB of receipts.

## Open decisions for Artin

1. Whether OPTIMIZER-MEMORY-ABLATION-1 (same gradient stream, reset or
   cross-fostered Adam moments; the causal test of the carried-history
   direction) gets a design GO. It is the one follow-up the desk
   supports; it is a training intervention and needs its own pre-reg.
2. Whether to re-run the optimizer desk under a k = 1 / residual
   resolution rule (new pre-reg) or leave the descriptive reading as
   the record.

## Next session: where to start

This handoff, BOARD line 5, RESULTS tail from the OPTIMIZER-GEOMETRY-
DESK-0 verdict, the RIFF bank TASK-GRADIENT GEOMETRY (two measured
anchors), AMENDMENT WRITER-INTERVENTION-SCHEDULER-LAW (L73710) for the
writer-pair wording.
