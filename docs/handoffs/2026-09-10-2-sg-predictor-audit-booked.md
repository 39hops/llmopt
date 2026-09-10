# Handoff 2026-09-10-2: SG arena adopted, predictor audit booked, SG credit law + integrity smoke built, seal proposal on file

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run.

## What landed (after handoff 2026-09-10-1)

- AMENDMENT SYNTHETIC-GRADIENT-WRITER-1-ARENA (RESULTS L70126, commit
  29a63875): Artin's seven folds of 09:45 EDT booked as design (arena
  = top four blocks over the frozen backbone, seed-27 paired control
  band c +- 7 with adequate-control clause, running-mean target scaling
  REMOVED, teacher-pass integrity resolution, ordered LINEAR / MLP-256
  ladder, checkpoint policy). No SG birth authorized.
- PRE-REG SG-PREDICTOR-AUDIT-0 (L70219, same commit, instrument
  smoked and committed before any retained checkpoint was read) and
  its run sgaudit0 under liverun (92 s, HEAD 7993b468).
- OBSERVATION SG-PREDICTOR-AUDIT-0 (L70301, commit ea4b51ca):
  HEALTHY (0 nonfinite, 0 exact-zero token norms, 12/12 states);
  targets five to six decades below unit scale (element RMS 7.5e-7 to
  7.8e-6; per-token dynamic range 5.3 to 6.1 decades on trained
  states); registered trigger FIRES; fixed constants booked: arena
  s_4..7 = 5.464e-6 / 4.479e-6 / 3.564e-6 / 2.441e-6, full-stack
  s_0..7 = 4.282e-6 .. 1.837e-6 (BS-32 mean-CE grain only). Prior 4
  hits 3 misses (family 11 / 8). Auditor blocker (prior tally) and
  should-fixes folded before booking. FINDINGS bullet [REPLICATED].
- scratch/sg_credit.py: the SG credit law (LINEAR / MLP-256 with zero
  output layers, parameter-detached functional_call teacher pass,
  normalized regression, registered step order) + tests/test_sg_credit.py.
- scratch/sg_integrity_smoke.py: fold-5 checks (a) to (g) PASS 4/4
  (arena and full-stack, both families) on the frozenbb1_smoke finals
  with bit-exact equalities; receipt logs/sgwriter1/integrity_smoke.jsonl
  (locked). Design correction surfaced by the smoke: zero-output
  predictors give exactly zero block GRADIENTS at step 0, but the
  blocks still move by AdamW's decoupled weight decay; L69765's "the
  first steps move only head / norm" is restated in the proposal.
- checkpoints/frozenbb1 pruned per fold 7 (scratch/fb_prune.py; 93
  files, 6.55 GiB freed, 0 digest mismatches; kept W_0 anchor per seed
  + step_00463 + final per arm, 1.1 GB; inventory
  logs/frozenbb1/prune_inventory.json locked). The inventory was
  force-added in the SAME commit as the observation because the
  observation cites it (the receipt guard refuses a cited-but-absent
  path); the prune ran after the audit receipts were digest-locked.
- Seal proposal: docs/superpowers/specs/2026-09-10-sg-writer-1-seal-
  proposal.md (writer law, integrity law, ordered ladder with a
  first-match stopping law, prior, cost, open points). NOT sealed.

## Conditions that bite next session

- Nothing is armed. The seed-27 qualification needs, in order: Artin's
  word on the proposal's three open points; AMENDMENT -SEAL booked;
  birth driver (sibling of birth19m_fb.py with MODE=sg, FAMILY, PLR)
  + source-invariant test + endpoint identity test + 300-step smokes
  + integrity smoke rerun; clean-tree prereg-auditor; separate Artin
  GO for births.
- Index regen quirk persists (act-envelope row); re-pop by id.
- Two pre-existing needs_link rows are not this thread's.
- Disk 29 GiB free.

## Open decisions for Artin

1. Seal proposal open points: first-match stopping law v full ladder;
   predictor LR order 3e-4 then 3e-5; alignment diagnostic descriptive
   only.
2. GO to write and smoke the SG birth driver on the proposal (zero
   training beyond 300-step smokes), then seal.
3. Whether the SG cells should also snapshot the predictors (adds 1.3
   MB x 17 per cell; proposed yes, for the alignment desk).

## Next session: where to start

This handoff, BOARD line 5, RESULTS tail from L70301, the seal
proposal spec.
