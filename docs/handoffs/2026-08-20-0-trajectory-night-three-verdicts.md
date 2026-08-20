# Handoff 2026-08-20-0: the trajectory night — FREEGEN-2 refuted, the autopsy, and the whole detect-retry chain measured in one evening

Seat: Fable (main session model), Mac. HEAD at close: see closing
commit (this file). Both machines IDLE, nothing armed, everything
measured is booked.

## What landed (all committed + pushed, in order)

- VERDICT QWEN-BLE-FREEGEN-2 (L35764, 4150024): BOTH BARS NO-FIRE
  AT ZERO (0/30 xhigh terminations, 0/60 correct) — REFUTED-IF
  triggers, early-band s16 repair does not reach the deliberation
  loop, BLe stays a scored reference arm. Runtime exonerated
  (~9.1 tok/s sustained, no fault). Prior wrong both legs. Both
  auditors pre-booking; remote sha sidecar receipted.
- OBSERVATION QWEN-BLE2-XHIGH-AUTOPSY (18a89fb): the 3072-token
  loops are DETECT-RETRY LIMIT CYCLES — exact token periods 88/242
  on two of three regenerated trajectories (greedy is deterministic;
  identity asserted v frozen rows), semantic full-restart cycling on
  the third; error DETECTION survives compression, error CORRECTION
  does not (the retry regenerates the same corrupted line).
- VERDICT QWEN-CHEAP-READOUT-0 (L36036, booked at aaec001): BOTH
  BARS FIRE — min R_256(A_W) 0.98034 (KNIFE-EDGE: fires by ONE
  position, 0.046 sigma; prefix class at ceiling 1.0), readout gap
  0.0056 at k=256 (vendor head on identical h nets out 2 of 7
  misses); LEVEL-2-GO granted (router census may be registered,
  Pareto R_256 v rows/bytes touched per bank amendment). Identity
  fixture bit-exact; prereg-auditor blocker (stdout-quoted
  qualification) fixed by landing the run log + artifact-identity
  sidecar. Prior half right (K-class predicted weak, measured at
  ceiling).
- VERDICT QWEN-CYCLE-IMPULSE-0 (L36117, booked at 97e1ac9-class,
  pin 496127d): bar 1 FIRES AT 5/18 — ALL on item 3 (semantic
  restart); the EXACT-ORBIT items 0/4 are locally INESCAPABLE at
  these doses (0 genuine escapes across 192 bursts, T 0.3/0.7);
  bar 2 NO-FIRE 0/18. ADJUDICATION BASIS: offline replay of the
  token-ID sidecars — the driver's in-run run_escaped (18/18) was
  inflated by two artifacts (burst cap disarmed the detector;
  strict end<f passed zero-gap re-fires), caught independently by
  BOTH auditors, verified line-by-line, corrected before booking.
  Prior right on the bars, wrong on the mechanism ("measure-zero
  under sampling" refuted at 1.7% burst escape; two T=0.3 replicate
  pairs bit-identical over 3072 tokens). Basin color: 14/18 exact
  tails, periods 22..352 (search cap 400 / window 800 stated),
  temperature reshuffles WHICH cycle, never restores termination.
- PRE-REGS consumed: QWEN-CYCLE-IMPULSE-0 (c663805),
  QWEN-CHEAP-READOUT-0 (fb84023) — both committed pre-run, both
  machine-projected, both adjudicated through
  llmopt/lab/prereg.adjudicate_prereg with observations docs.
- RIFF banks (the GPT-seat review chain, all verified in-house
  before adoption): TRAJECTORY-SIDECAR (+ first registered use paid
  off immediately), TRAJECTORY-PRESERVATION-AXIS,
  NO-REGRET-RETRY-CONTROLLER (temperature now the WEAK baseline),
  SEQUENTIAL-RESOURCE-CONTROL umbrella, JSPACE-PRESERVATION-CENSUS
  (tightened: residual stream != J-space, CoT != J-space, autopsy
  reading stays hypothesis), LOOP-STATE-READOUT (+ five prereg
  hardenings + cycle-aligned metric law + specimen order 0/4
  primary), CHEAP-READOUT MIPS + Pareto amendments,
  PRIMITIVE-EVIDENCE DOCTRINE (receipts persist primitives
  sufficient for independent recomputation; intervention caps must
  never disable observation), 3080 cleanup ON HOLD (Artin ruling:
  after dependent rungs drain).
- FINDINGS: three new bullets (FREEGEN-2, CHEAP-READOUT softened
  per GPT wording catch — k-scoped, no set-inclusion —
  CYCLE-IMPULSE); curated counts regenned.

## The chain (the night's shape, for the next reader)

FREEGEN-2 (deliberation collapses at zero) -> autopsy (collapse =
detect-retry cycles; detection intact, correction gone) ->
CHEAP-READOUT (on A + teacher-forced MODEL-1 states, readout
quantization is a SMALL contributor to candidate recall loss;
free-running BLe loop-state readout remains UNMEASURED — that
scoping is deliberate, do not compress it to "damage upstream") ->
CYCLE-IMPULSE (token-space perturbation moves BLe between recurrent
trajectories, never out; exact orbits fully restoring). Every arrow
points at LOOP-STATE-READOUT.

## Next session opens with

1. This handoff -> BOARD -> RESULTS tail (4 entries after the
   FREEGEN-2 prereg).
2. LOOP-STATE-READOUT prereg (the deliberate next registration —
   GPT + house agree it deserves a fresh session, not a midnight
   launch). Everything needed is banked: operand/provider fix
   (local margin, never teacher-style), tower-specific capture
   identity fixture, sparse event windows, cycle-aligned
   homologous-state metrics (t v t+88/t+242), hidden-state
   recurrence (cosine/rel-L2), specimens 0/4 primary + 3 contrast,
   mechanistic ladder (BLe head -> vendor head offline -> vendor
   body only if needed).
3. Also unlocked/queued: LEVEL-2 router census (Pareto framing),
   ATTN-ROUTER-CENSUS, RESIDUAL targeted patch (needs BLe recensus
   if patching BLe), phase-2 runtime (Nsight profile FIRST).
4. Artin decisions: LOOP-STATE GO; 3080 cleanup stays HOLD.

## Conditions that bite next session

- Banked forward fixes now several deep: OPERAND-PROVIDER
  prereg-schema field (still human-enforced); artifact sha in
  start_provenance; refuse-if-exists on sidecar writes; derived
  scoring in adjudicators never run loops; per-step
  art_dir/manifest_sha in ladder receipts; margin_edges +
  per-position miss masks in census receipts; qualify report into
  the receipt (not stdout-only).
- logs/qwencycle sidecars + logs/qwencheapread arrays: sidecars are
  force-added (evidence basis, seedslad exception booked in the
  verdict); census npz stays untracked, sha-pinned.
- The FREEGEN-2/CYCLE receipts on the 3080 are now frozen evidence
  paths — never append there; new runs get new dirs.
- Scorer interpreter is .venv_teacher on the Mac (transformers
  5.15 + accelerate); plain .venv lacks accelerate.
