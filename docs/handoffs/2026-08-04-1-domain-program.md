# Handoff 2026-08-04-1: the domain program (MOE-GT-2, resolved same-day)

Session: Fable seat, Mac (3080 untouched today after the morning),
post-compact continuation of 2026-08-04-0. Everything booked and
pushed through VERDICT MOE-GT-2-D4-CROSS.

## What happened, in order

1. Two Opus survey seats (self-reported 4.5, disclosed) audited the
   repo AGAINST the spec. Fable-verified catches: the decode-only
   split had no committed implementation (amendment prose only);
   mechanics L1 = exactly 30 unique prompts/kind (enumerated) and
   make_dataset's dedup loop HANGS on exhausted cells; place1's sim
   never charges prefetch inserts as traffic and is NE=64-hardcoded;
   R5 confirmed the crest FRACTION but never re-derived its LOCATION
   (the DIET-COND-SEED failure mode) — promoted to arm D1.
2. Instrument: TRAJ v2 (4f3dc6c) — phase (prefill/decode) + router
   scores recorded AT WRITE TIME (MLX prefill = multi-token router
   call, decode = 1). D0 regression: gate + tail bit-identical to
   the certified artifact, 590,736 rows, v1 sha-unchanged.
3. PRE-REG MOE-GT-2 (e557e4c), then the ladder, all same-day:
   - D1: crest location HOLDS on unspent seed 555 (argmax 45.3%,
     crest 82/120 vs full 60/120 = +22, program record).
   - D2: physics arm-0 (mechanics L2-3, seed 606, 120/120
     determinability pre-flight). Full baseline 36/120. Jaccard
     (math,phys) 0.767 vs split-half nulls 0.930/0.871 — the router
     IS domain-biased; killer (~1.0) did not fire.
   - D3: code arm-0 (ladder corpus, toolchain-scored, llvm on Mac
     confirmed; o2_asm rung EMPTY at seed 99 — flagged). Full
     baseline 48/120. THE ORDERING PREDICTION FIRED: registered
     desk numbers (corpus token overlap 0.329 vs 0.097) predicted
     Jaccard(math,phys) > Jaccard(math,code): measured 0.804 vs
     0.543, and 0.543 < code's own split-half null 0.653.
   - D4-PHYS: NULL — physics crest +3 pooled vs the +7 bar
     (-4/+4/+3). The beats-full crest does NOT transport. Physics
     crest closed recall 0.727 (math crest: 0.887); the gate/probe
     dissociation reappeared (seed 909 probe '3.3.3.3...' while its
     gate improved).
   - D4-CROSS: cross-masks (Jaccard 0.543 with the right coalition)
     degrade -56%/-70% but DO NOT DIE — 21/120 (math mask on code)
     and 19/120 (code mask on math) vs R6 random's 0/120. Coverage
     sigmoid now has three points: 44.7% -> 0, ~78-80% -> 19-21,
     86.5% -> 55-60, 90.1% -> 76-87. Under the math mask the code
     gate survives ENTIRELY via the natural-language diagnose rung
     (20/26; compile/assemble rungs 0-1) — coalitions look
     FUNCTIONAL, not topical.

## The corrected one-line summary

Coalitions are real, domain-organized (distance tracks corpus
distance), and functionally specific; the beats-full crest is
math-only on present evidence; capability follows a steep coverage
sigmoid, not a cliff.

## Operational

- 14:30 KERNEL PANIC, Fable's error: a smoke test loaded a second
  30B beside the running D4 job; swap storm; watchdog killed the
  machine. Rebooted clean; zero artifacts lost (everything commits
  before runs fire); D4 rerun sole-resident. Memory rule saved:
  one resident 30B, check rjob status before ANY model-loading
  command, smokes run BEFORE long jobs, never beside.
- New instruments: gt2_code_arm0.py (code arm-0 + masked cross-arm
  mode); arm2 ARM0/KINDS/LEVELS knobs; decode-only demand JSONs in
  checkpoints/ (untracked): gt2_{math,phys,code}_arm0_decode.json.
- Machines: Mac idle, all rjobs DONE; 3080 untouched since morning.

## Queue (next session)

1. Physics-native fraction ladder (the D4 null's sharpest follow-up:
   is the physics crest at a different fraction, or absent?).
2. Rung 2 PLACE-1: add the traffic counter + NE=128, THREE named
   arms (LRU / demand-template-pin / conditional-prefetch), pre-reg.
3. Rung 3 routing-bits ledger (router scores now in TRAJ v2 rows).
4. Rung 4 crest-as-data-generator (math domain — unaffected by the
   transport null).
5. Hygiene: ENTROPY-LICENSE-1 decode-only recount (v2 traj makes it
   trivial); GRAVMOE 1/8-vs-2/8 attribution; churn-judge spend
   registration; INDEX.md pointer for GT-1/GT-2 scratch instruments.
6. Axiom relay 2026-08-04-0 still to deliver (Artin carries).
