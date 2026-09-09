# Handoff 2026-09-09-0: WRITER-DFA-1 built, GO'd, qualified, STOPPED as DFA-UNSTABLE

Seat: Fable 5.1 on the Mac. HEAD at close: see the commit that carries
this file (parent 5568f890). 3080 window: untouched this session, no
remote jobs. No live registered run (liverun status: none).

## What landed (newest last)

- 282200dc AMENDMENT WRITER-DFA-1-PRECISION (RESULTS L68644): Artin's
  four implementation-precision folds (writer-integrity smoke,
  alignment reduction, effective rank, 8 x 5 census mandatory on band
  pass) plus operational clarifications; no threshold changed.
- 912e50bb instrument set: scratch/birth19m_dfa.py (MODE=bp|dfa,
  sealed seed law), scratch/dfa_credit.py (feedback law, detached
  forward, DFA objective), dfa_probe.py + probe.json, dfa_leakage_smoke.py
  + leakage.json (nine checks pass, BP/DFA parity 0.0), dfa_act.py,
  dfa_align.py, dfa_trajcensus.py, writertraj_depend.py DEPEND_SET=dfa,
  dfa_depthclass.py, dfa_qualgate.py, dfa_verify.py, three liverun
  drivers, tests/test_dfa_source_invariant.py; 300-step smokes both
  modes; OBSERVATION WRITER-DFA-1-ACT-ENVELOPE-0 (L68742): ACT null
  envelope 8.904, ACT(A, B) 15.714, booked before any DFA snapshot read.
- 521fb58e clean-tree prereg-auditor folds (alignment chunk 32, mirror
  paths, selection staging, receipt keys, dirty-tree disclosure).
- b73681a5 / be72538f qualification launches under liverun
  (writerdfa1-qual-c1, writerdfa1-qual-c234); receipt-auditor on cell 1
  clean (param_dtype fold).
- 5568f890 VERDICT WRITER-DFA-1 (L68802): DFA-UNSTABLE. All four cells
  gate 0 / 120 at 0.0% validity, loss 3.34 to 3.51 (uniform 3.69).
  Sealed STOP at qualification; no seed-2 birth; accessibility only.
  FUNCTION-BAND prior a miss under the literal S13 reading (adjudicated
  explicitly in the entry; the conditional "unscored" reading is named).
  Family record 5 hits, 2 misses. FINDINGS bullet, RIFF bank and
  program amended in place, BOARD, JSON outcome, receipts locked.

## Conditions that bite next session

- Untracked artifacts: checkpoints/writerdfa1/ (4 qualification cells,
  68 snapshots + finals + feedback, 5.1 GB, digest-anchored in
  logs/writerdfa1/qual.jsonl) and checkpoints/writerdfa1_smoke/ (578 MB).
  Disk was 36 GB free at launch. Disposition is Artin's call; the smoke
  directory can go on GO.
- /Users/artin/code/llmopt-repair (detached worktree at ec6de1ae) still
  holds the ATOM repair snapshots; the N2 / N4 null specimens live there.
- scripts/gen_results_index.py re-derives `type` and `needs_link` on
  every regen; the ACT-envelope observation row is typed "amendment"
  because its title cites the SEAL amendment. Curation fields (threads,
  links) survive; re-check needs_link after any regen. Backlog rows
  from 2026-09-04 / 09-05 (two observations) still carry needs_link.
- FINDINGS ratchet headroom: one bullet added this session; the
  observation entries carry none (they are not verdicts).
- The two DFA amendments cite the pre-reg as L68321; the heading is at
  L68323 (noted in the verdict, no amendment needed).

## Next session: where to start

1. This handoff, then docs/BOARD.md line 5 (the live-run law line now
   carries the DFA status), then RESULTS tail from L68802.
2. Nothing is armed. Recommended first batch, all decision-gated:
   - If Artin wants a readout on WHY DFA plateaued: a small pre-reg for
     a descriptive alignment / ACT pass on the four qualification arms
     (zero training, CPU, frozen probe; the sealed diagnostic is defined
     on seed-2 arms which do not exist).
   - The next foreign writer candidate from the RIFF bank (synthetic
     gradients / DNI, hybrid DFA with a true-gradient last block, or a
     DFA regime pre-reg at another width / horizon / normalization).

## Open decisions for Artin

1. Next foreign-writer candidate (or park the mechanism-attribution
   program at accessibility).
2. Whether to pre-register the descriptive qualification-arm alignment
   / ACT readout before choosing the next writer.
3. FUNCTION-BAND scoring: the verdict books the literal S13 reading
   (MISS). If the conditional reading is preferred, an AMENDMENT flips
   the family record to 5 hits, 1 miss, 1 unscored.
4. Disposition of checkpoints/writerdfa1/ (5.1 GB) and the smoke tree.

## Standing

- Relays unsent: none. Runs finished-unbooked: none. Banked, unarmed:
  WRITER-INVARIANCE-DESK-0, the coupled-dynamics and Hamiltonian-writer
  banks, the three Lean / verifier-invariance banks.
