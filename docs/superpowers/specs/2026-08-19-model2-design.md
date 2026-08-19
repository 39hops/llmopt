# MODEL-2 design spec, r2 (design phase only — registration and launch go through /rung on Artin's GO)

Written 2026-08-18 evening on Artin's design GO; r2 same evening
after a GPT identity catch (verified in-house against the census
algebra) killed r1's arm set. Evidence base: MODEL-1 tree,
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
If the F-conditioned effects transport to the FLe base:
- X(P_X) ~ 0.3485 - 0.128  = 0.2205  (would BEAT C's X 0.2486
  at 74% of C's linear spend — the headline if it holds)
- K(P_K) ~ 0.2248 - 0.0401 = 0.1847  (improves FLe, NOT
  predicted to beat C's K 0.1618)
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
  bands |X(P_X) - 0.2205| etc. as SEPARATE interference-science
  bars with a registered nat-band, never floors-as-significance.
- refuted_if predicate: the transport prior is refuted if the
  crossover INVERTS on both metrics past floors (late beats mid
  on X AND mid beats late on K). Precedence: sanity bars only.

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

## 7. Open decisions for Artin (before /rung)

1. Approve the P_X / P_K third-pick crossover design (r2).
2. Transport-band width for the range bars (suggest +-0.05 nats
   around the desk predictions, stated as science bands).
3. ALe mechanism 2x2: bank only, or register alongside?
