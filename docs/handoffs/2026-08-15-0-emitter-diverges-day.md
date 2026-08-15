# Handoff 2026-08-15-0 — emitter-diverges day: dose ladder + SOFT-NEXT refuted + rule-ablate LIVE

Seat: Fable 5, Mac. HEAD at close: see the handoff commit.
Session: continuation of the 08-14 seat (IV6/IV7 acceptances,
ATOM-DIET-LADDER-1 replication booked earlier in it). Mac runs
RULE-ABLATE-1 at close (LIVE, watcher armed); 3080 idle after
SOFT-NEXT-1 (checkpoints remain there).

## What landed (all pushed, suite green 860/11 at last full run)

- **VERDICT ATOM-DOSE-LADDER-1** (L29662, booking afa4f92, prereg
  a783088): EMITTER-DIVERGES — axiom-emitted atoms at matched
  dose read 66/L4=7 v sympy's 70/L4=12; within the axiom family
  monotone 64/66/72 through ~6.3% real exposure, NO flooding
  scar (bar 3 the only fire). Per-row worth ~0.67x total / ~0.42x
  L4, booked INDICATIVE-ONLY under the resolution law. Farm:
  12,000/12,000 axiom rows (build-iv7 5a8ae70, sympy re-verified,
  8s L4 wall for the bimodal-heurisch band; 3.5h). Prior 1 hit in
  6 legs; family 3 hits 13 misses.
- **VERDICT SOFT-NEXT-1** (L29733, same booking commit, prereg
  2545cd6): REFUTED at recipe — trie soft targets on the 15.7%
  conflicted rows moved calibration +0.0033 (bar +0.05) and gate
  within noise (61 v 64, sub-resolution). Model already parks 63%
  of teacher-forced mass on valid-answer sets under one-hot.
  First entries of the NEW 3080-bf16 gate family (never compared
  to Mac numbers). Prior 0 hits in 2 legs; family 3 hits, 15
  misses.
- **Auditor discipline paid twice**: dose draft had the exchange
  rate backwards (~0.5x claimed; receipts say 0.67x and axiom 12k
  bought MORE than sympy 6k) + a wrong code-commit story;
  softnext draft quoted weights shas with no Mac-side receipt
  (3080 logs pulled + force-added) and a booking-order dependency
  in the family record. All adopted before booking.
- **PRE-REG RULE-ABLATE-1** (5da3ba7) + instruments + LAUNCH:
  heurisch ablation (3,218 non-heurisch rows) v dose-matched
  random control on the frozen sympy shard, seed 3, matched
  horizon. BAR 1 = ctrl_L4 - noheur_L4 >= 3 (the carrier bar).
  Disclosed confound: ablated arm is L4-richer by rows (43.8% v
  ~40%). Prior: ctrl 68/L4 10, noheur 66/L4 7.
- **Riffs banked**: pre-calculus basics tier (+ Artin's catch:
  Series rung 1c L3444 already measured explicit arithmetic 4.3x
  — bank amended); engine-signals-to-FFT-weights (gauge fence,
  desk residue SPECTRAL-CURRICULUM-0); SOFT-SPEED-1 (collapse
  conflicted rows to soft rows, ~13% diet cut — the lossless
  speed lever, next Mac rung candidate); HINT-DIET-0 (bigger
  distributions / hints, desk census first). RIFF living-doc
  amendments: atomic-op bank now SHARD-FAMILY-SCOPED; SOFT-NEXT
  residue marked dead with revival condition.
- **wsl_guard loosened** (Artin's overnight-autonomy ask):
  tracked-driver launches with log+marker under logs/ now allow;
  cd/nvidia-smi/mkdir-logs read segments allow; deny +
  friendly-fire rules unchanged; test table pins the new
  behavior. One classifier-blocked edit left undone: the
  /dev/null-redirect exemption for the CHANGES_STATE matcher
  (cosmetic; a run with >/dev/null outside the sync shape still
  asks).
- Cross-lab: IV6 + IV7 acceptances booked earlier this session
  (L29337, L29417), house re-pinned to build-iv7 (5a8ae70), Lean
  eligible=True documented lexical-only upstream (their c154ac1),
  zero open items both sides. AXIOM-SURFACE.md is current through
  IV7 and updates whenever axiom changes.

## Conditions that bite next session

- **RULE-ABLATE-1 is LIVE** (rjob id ruleablate, ~2h from 11:13
  launch): if this session's watcher dies with the compaction,
  check jobs/ruleablate.rc + logs/ruleablate1/arms.jsonl and BOOK
  against pre-reg L(grep RULE-ABLATE-1) — prereg-auditor pass
  first, FINDINGS bullet same commit (ratchet at cap==backlog).
- Newly FROZEN (results-cited): scratch/farm_atoms_axiom.py,
  birth19m_atoms_dose.py, atomdose1_driver.sh,
  birth19m_softnext.py, softnext1_driver.sh; receipts
  logs/atomdose1/, logs/softnext1/ — extend via siblings, never
  edit; never append into frozen receipt files (the rule driver
  already repoints RECEIPTS for exactly this reason).
- Untracked evidence: data/micromodel_atoms_axiom_shard0.jsonl
  (72M band SPENT), derived shards
  data/micromodel_atoms_{noheur,ctrl3218}.jsonl (selections, re-
  derivable via scratch/make_ruleablate_shards.py). Exhaust now
  ~10.3GB incl. dose + softnext checkpoints (softnext pair lives
  on the 3080).
- 3080-bf16 gate family exists now — its numbers NEVER compare to
  Mac/fp32 families.
- Booked family record stands at 3 hits, 15 misses BEFORE
  RULE-ABLATE-1's 4 registered legs.

## Next session, in order

1. This handoff -> BOARD -> RESULTS tail (resume protocol).
2. Book RULE-ABLATE-1 if the watcher didn't already.
3. HOUSE PICK next rung: SOFT-SPEED-1 (banked, Mac, zero farm —
   the equivalence is measured, the lever is compute).
4. Then: basics-diet pre-reg (1c anchor), HINT-DIET-0 desk census,
   wider-n atoms ladder, watch-it-think flagship (Artin asked
   about thinking/effort — the queued flagship is the answer).

## Open Artin decisions

1. Phase C frozen-paths ruling (181 files) — unchanged.
2. Rebirth MEDIUM substitutions — unchanged.
3. Static-figure unpause — unchanged.
4. Exhaust keep-or-delete — now ~10.3GB.
5. CHECKERS-1 disposition — unchanged (house lean: leave).
6. The classifier-blocked wsl_guard edit (apply yourself or skip).
7. Next pre-reg pick (house lean: SOFT-SPEED-1).
