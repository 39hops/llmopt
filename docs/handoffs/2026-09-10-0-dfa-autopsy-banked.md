# Handoff 2026-09-10-0: DFA autopsy booked (rank collapse supported), FROZEN-RANDOM-BACKBONE banked, frontier set pruned

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run.

## What landed (after handoff 2026-09-09-2)

- PRE-REG DFA-LOWER-HARM-DESK-0 (L69542) and its instrument
  (scratch/dfa_harm_desk.py, scratch/caf_actpost.py, driver), smoked on
  the seed-11 arms, committed before any gate was read.
- Run dfaharm0 under liverun (pid 54947, rc 0, 19 min): six transplant
  gates on mps, 54 ACT rows and 18 alignment rows on CPU.
- OBSERVATION DFA-LOWER-HARM-DESK-0: ZERO-TOP <- HYBRID-LOWER gates
  0 / 120 at k = 1, 2, 4 (endpoints 28 / 55 / 62); HYBRID-TOP <- W0-LOWER
  0 / 120 at every k. Registered support law met: hybrid residual stream
  rank 1.0 to 1.6 at every block from step 463 (backprop segment
  included), frozen controls keep W_0's rank 20 to 23 below the boundary
  and grow to 15 to 63 above it. Rank-collapse reading SUPPORTED as an
  intervention result on one seed; DFA family closed on evidence.
- RIFF: FROZEN-RANDOM-BACKBONE banked (62 / 120 with emb + bottom four
  blocks frozen at W_0, one seed, not replicated; candidate rung
  FROZEN-BACKBONE-1 = fresh paired seeds, full BP v bottom-half-frozen
  BP; NOT armed). Foreign-writer bank amended in place.
- checkpoints/writercaf1/ pruned to the keep set after a 0-mismatch
  digest inventory (logs/writercaf1/prune_inventory.json): finals and
  step_00463 of the three zero controls and the three lr 3e-4 hybrids,
  feedback artifacts, one canonical W_0. The 868 MB WRITER-DFA-1 keep
  set retained.

## Conditions that bite next session

- Synthetic gradients / DNI is the next training writer; design only,
  needs its own pre-reg and Artin GO; nothing armed.
- Index regen quirk (memory note): re-pop needs_link by id after regen.
- The receipt guard's NOT_RUN_ABSENT table pins the frontier's
  births.jsonl; the harm-desk receipts are all present and locked.

## Open decisions for Artin

1. GO / no-GO to design PRE-REG SYNTHETIC-GRADIENT-WRITER-1 (predictor
   family, target law, integrity smoke of the weaker "true gradients
   update only the predictors" law).
2. Whether FROZEN-BACKBONE-1 (paired seeds, full BP v bottom-half frozen)
   is worth a rung before or after the synthetic-gradient writer.
3. Whether to test the untested candidate cause of the collapse (B_l e
   confined to the 40-dimensional column space of B_l) with a zero-
   training desk on the existing snapshots.

## Next session: where to start

This handoff, BOARD line 5, RESULTS tail from the harm-desk observation.
