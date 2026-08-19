# MODEL-2 design spec, r3 (design phase only — registration and launch go through /rung on Artin's GO)

Written 2026-08-18 evening on Artin's design GO; r2 same evening
after a GPT identity catch (verified in-house against the census
algebra) killed r1's arm set; r3 adds the HELD-OUT EVALUATION
SURFACE requirement (GPT catch: the frozen MODEL-1 corpus/prefix
positions selected this policy, so they are the development
surface and cannot validate it). Evidence base: MODEL-1 tree,
ATTN-ATTRIB-1, IO-ATTRIB-1, LBAND-1 (+diagnosis closure),
RK-CENSUS-0, CAPACITY-METER-1 r2/r3, EFFORT-QUANT-0. Booked
numbers only.

## 0. The r1 identity bug (recorded so it cannot recur)

r1 proposed P1 = "BLe + io". But BLe is BASE B + early band, and
B is A + the io pair — io is ALREADY INSIDE every B-based arm.
Census algebra from the receipts: A 402/0 (w4/s16), B 400/2,
BLe 352/50, FLe 288/114. So "BLe + io" is the EXISTING BLe
artifact (X 0.433 measured), and r1's desk prediction
X ~ 0.206 subtracted the io recovery TWICE from B. Likewise
"P1 + F" is the existing FLe (X 0.3485 measured). Neither is a
new experiment. Lesson: desk-price arms on the CENSUS LATTICE
(which keys are s16 in the composed artifact), never on named
payload sums.

## 1. What MODEL-2 is

The allocation instrument: fix B as the allocation baseline
(io always in — both per-byte orderings put io in the first two
picks with early-linear); the first two additional picks under
EITHER metric objective are early-linear + full-attn, whose
frozen state is FLe (X 0.3485, K 0.2248, measured). MODEL-2 asks
the THIRD-PICK question at exact iso-spend: which band does the
next 461,276,672 bytes buy the most of which metric?

## 1b. HELD-OUT EVALUATION SURFACE (registration precondition)

The MODEL-1 corpus/prefix positions chose this policy (every
marginal in §3 was measured on them) — they are now the
DEVELOPMENT surface. Before any compose: freeze a DISJOINT
MODEL-2 evaluation corpus + prefix set (fresh prompts, exclude=
against the MODEL-1 lists per data-contract doctrine), run the
locked teacher over it once, pin record shas. FLe, P_X, P_K, AND
C are all scored on the NEW surface (C's frozen receipts are
development-surface numbers; any P_X-beats-C claim needs C
rescored on the held-out surface). RESCORE-mode class receipts.

## 2. Arms (prospective, exact-spend, both NEW census states)

- P_X = FLe + mid-linear band   (FLe state + BLm payload keys)
- P_K = FLe + late-linear band  (FLe state + BLl payload keys)
Each adds exactly 461,276,672 bytes (manifest-derived constant,
already the adjudicator's BAND_BYTES). Neither census state
exists yet (P_X: 240 w4/162 s16; P_K same counts, different
keys). C is the ENDPOINT of this ladder (FLe + mid + late), not
a matched control — C's frozen receipts anchor the far end.

## 3. Registered predictions (transport of the F-conditioned
   LBAND marginals — this is the science)

From the booked conditional table: dX(mid|F) = 0.128,
dX(late|F) = -0.0037; dK(mid|F) = 0.0235, dK(late|F) = 0.0401.
Transport predictions register as NEW-SURFACE MARGINALS relative
to FLe (old absolute numbers are development-surface desk color
only):
- dX(P_X|FLe) ~ +0.128, dX(P_K|FLe) ~ -0.0037
- dK(P_X|FLe) ~ +0.0235, dK(P_K|FLe) ~ +0.0401
Desk color (development surface): X(P_X) ~ 0.2205 would beat C's
0.2486; the added mid band brings the ladder to 2/3 of C's
linear-attn spend (~74% of C's TOTAL attention-repair spend above
B once full-attn is counted — spend fractions per GPT
terminology fix). K(P_K) ~ 0.1847 improves FLe, not predicted to
beat C's 0.1618.
PRIMARY EXACT-SPEND CROSSOVER (the registered claim):
  X(P_X) < X(P_K)  AND  K(P_K) < K(P_X)
— mid is the X pick, late is the K pick, at identical bytes.
Prediction failure is interaction/transport science (the
conditioning table does not transport one level up), NEVER an
instrument alarm.

## 4. Bars sketch (frozen at /rung; gate_class now EXECUTABLE —
   shipped in llmopt/lab/prereg.py with validation + fixtures)

- SANITY bars (gate_class "sanity"; the ONLY bars
  refutation_precedence may name — validator enforces this):
  teacher identity, traversal 48/16, compose admissibility
  (base FLe / donor C / band keys / promoted 48 / derived
  461276672 bytes / chain identity), finite readings. NO
  behavioral X/K brackets in sanity — r1 had a C-anchored
  bracket that would have ALARMED ON THE HEADLINE SUCCESS
  (X(P_X) beating C). Behavioral excursions are RANGE outcomes.
- RANGE bars (gate_class "range"): the two crossover conjuncts
  (X and K legs as separate bars, floor-multiple form); transport
  bands on DELTAS-TO-FLe (|dX(P_X|FLe) - 0.128| within 0.05,
  K within 0.03) as SEPARATE interference-science bars, never floors-as-significance.
- refuted_if predicate: the registered crossover is CONJUNCTIVE
  — EITHER leg materially reversed past floors means the
  registered crossover claim failed (that is the refutation
  predicate). A BOTH-legs inversion books additionally as the
  stronger theory-kill, disclosed in prose, not required for
  refutation. Precedence: sanity bars only.

## 5. Costs (desk)

2 composes on the 3080 (FLe exists there as base; ~2 min each +
transfers), 2 scores on Mac (~8 min each), auditor pair. Well
under 1.5 h wall. No GPU-heavy jobs.

## 6. Separate threads (explicitly OUT of this rung)

- BLe free-generation screen: separate deployment OBSERVATION;
  frozen MODEL-2 registration first so it cannot steer arm
  selection.
- OPTIONAL mechanism rung (bank): compose ALe (A + early band,
  no io) to complete the A/B/ALe/BLe 2x2 and measure the
  io x early-linear interaction — existing data CANNOT test that
  additivity because every early reading is conditional on io
  being present (B-based). Register only if the interaction
  question earns a slot.
- Meter/kurtosis stays diagnostic, never allocator (booked 2x).

## 7. Decisions (Artin via GPT relay, 2026-08-18 evening)

1. P_X / P_K crossover design: GO — AFTER the held-out split
   (section 1b) is frozen.
2. Transport bands on DELTAS-TO-FLe: +-0.05 nats (X), +-0.03 (K),
   secondary science bars; the crossover is primary.
3. ALe mechanism 2x2: BANK ONLY, separate rung if earned.
