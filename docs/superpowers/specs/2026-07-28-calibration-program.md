# Spec: The Calibration Program (2026-07-28)

Successor named by the exact-representations program's own closing
verdict (snap anatomy, RESULTS 2026-07-27): "precision below the
near-tie scale buys nothing; the capability lever is near-tie
density (calibration) and verification, not digits." This spec
makes near-tie density a measured instrument, a mechanism study,
a training lever, and an inference lever — in that order, each
rung gating the next.

## Thesis

Damage under any weight perturbation = near-tie density x horizon
(measured: full coarse snap flipped 2 of 2,512 decisions, both at
margin 1.6e-4 vs median 8.9; the same snap error that halved the
Mac-19M cost the union-45M one point — per-crystal property).
Frontier decisions are near-tie-superposed and currently collapsed
by hardware rounding (same checkpoint 18/24 cuda v 9/24 MPS). If
calibration at decision points is measurable and trainable, it is
the first new capability lever since the diet laws.

## Rung 1 — the instrument (Mac, hours, no training)

**flips-per-token under Q=16 snap** as the calibration probe:
snap a crystal's weights to the Q=16 rational lattice, count
next-token decision flips per token against the unsnapped model
on a fixed probe set (held-out cur states, greedy, one device).

- Validation targets (already-measured ground truth): Mac-19M
  (rat16 49->26 crack) must read HIGH; union-45M (rat16 immune,
  64) must read LOW; the d256 zoo must rank consistently with its
  known gate spread.
- Fences: same-device only (near-ties are resolved by hardware —
  sigma never transports); VOCAB_EXTRA atom order pinned per
  crystal; probe set fixed and committed.
- Pre-reg: probe rank-correlates with measured snap robustness
  across >=4 crystals (Spearman, direction pre-declared).
  FAILURE = probe is noise; the whole program closes at the cost
  of one script.

## Rung 2 — the mechanism study (Mac, zero training)

**The MASS leg of the branching-entropy floor** — AMENDED at plan
time: the CE-gate study's sampled-coverage form ALREADY RAN
(RESULTS 2026-07-26) and the CE-anti-tracks hypothesis FAILED
(CE on a fixed instrument tracks the gate 4/4; Muon dissociation
retracted as a cross-instrument artifact). What that verdict
explicitly left unmeasured (its #3): "the branching-entropy floor
stays a THEORY without a measured leg." This rung measures it in
MASS form, teacher-forced (no sampling): per held-out cur state,
enumerate the engine's verified-valid nxt set (successors +
verify), compute each model's sequence-probability mass on (a)
the full valid set, (b) the farm's banked pick alone, and (c) the
distribution entropy over the valid set. Specimens: the same four
d256 crystals (muon 34 / stream3 45 / stream4 57 / control 65).

- Pre-reg (weakened honestly per the 07-26 verdict): mass-on-
  valid-set tracks the gate at least as well as mass-on-farm-pick;
  the DELTA (a)-(b) — mass on valid-but-unpicked moves — is the
  novel readout, and rung 3's baseline. No CE-anti-tracking claim
  is re-registered.
- Books either way: (a)~(b) everywhere = the floor theory stays
  unmeasurable at this scale and rung 3 loses its mechanism
  motivation (rung 3 still runs — its gate readout is
  self-sufficient).

## Rung 3 — the training lever (one d256 birth first)

**Distribution rows**: per cur, engine rule-fire enumerates every
applicable move (~ms), wave oracle verifies, MarkovPrior weights
-> soft-label target over the verified-valid set. Train a paired
d256 arm vs standard farm-pick rows at matched dose, same seed,
same device.

- Primary readout: gate at L4 (the canary level — read every
  intervention there first).
- Secondary readout: rung-1 probe delta — do soft labels REDUCE
  near-tie density? (The first test of calibration as a
  TRAINABLE quantity.)
- Fences: soft labels arrive as GRADIENT (hints-as-text is
  twice-nulled); verified AND distinct (no identity moves in the
  label set); exclude=-guarded splits as always.
- Promotion: pays at d256 -> one W* (19M) confirmation birth.
- Null reading: calibration is a diagnostic, not a lever; rung 4
  still runs (independent leg).

## Rung 4 — the inference lever (rides rung 1)

**Judge-collapsed decoding** pilot at d256: at decode steps with
top-2 logit margin under the near-tie threshold (~0.02, the
measured tie class), branch both continuations a few tokens and
let a cheap judge pick (value head / oracle at step boundaries);
greedy elsewhere.

- Gated on rung 1 confirming ties concentrate where capability
  fails (starved-judge law: fire only where variance lives).
- Economics fence (regret-round-2 lesson): scored at EQUAL TOKEN
  BUDGET vs plain greedy AND vs best-of-N; primary battery = the
  frontier band where ties concentrate.

## Kill conditions / honest exits

1. Rung 1 pre-reg fails -> program CLOSED for the price of a
   probe script; the flips-per-token idea books as a null.
2. Rung 2 books either way (mechanism study, no exit needed).
3. Rung 3 L4-null -> lever demoted to diagnostic; rung 4 proceeds.
4. Rung 4 fails equal-budget economics -> banked with the regret
   lineage; the probe survives as an instrument regardless.

## Machine plan

Rungs 1-2 are Mac desk work (no training). Rung 3 is one d256
birth (~20 min Mac) then optionally one 19M birth. Rung 4 is
d256 decode-side, Mac. Nothing here needs the 3080; nightly GO
unaffected.

## Pre-registrations to book in RESULTS before each run fires

- R1: probe-vs-robustness Spearman direction.
- R2: mass-on-valid-set tracks the gate >= mass-on-farm-pick;
  the (a)-(b) delta is the novel readout (no CE-anti-track claim
  — that form already failed 2026-07-26).
- R3: distribution arm >= pick arm at L4; probe delta negative
  (fewer near-ties).
- R4: judge-collapsed >= greedy at equal tokens on the frontier
  band, else null.
