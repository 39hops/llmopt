# Metabolic v4 — practice food + the persistence verdict (spec'd 2026-07-23, GO pending)

The v3 stack survives intact (surprise-gating, wave-contrast,
control rod, absolute anchor — no rollbacks tripped in 4 arm-hours).
v4 changes exactly two things and adds one instrument:

## Change 1: PRACTICE FOOD (stuck states, not random frontier)

Food = the stuck-state worklist (data/stuck_states_*.jsonl — the
model's own dead-wave states, vfrac=0 by construction) + fresh
stuck states logged live as the session generates them. Rationale:
v3's paired arms proved the organism eats random frontier food
without capability conversion; practice mode concentrates every
update on the transitions that actually fail. This is the
model-side mirror of axiom's adopted mining policy.

## Change 2: LONGER HORIZON

75-min arms were enough for the mechanism verdict (2.7x flips),
not conversion (LLMUE precedent: conversion is slow). v4 = one
overnight session (6-8h), fp64 masters ON (the mechanism winner),
single arm (the paired question is settled; the open question is
conversion, which needs duration more than pairing).

## Instrument: THE FLIP-PERSISTENCE CENSUS (latent-vs-churn decider)

Snapshot the ternary sign pattern every 30 min. Report per window:
new flips, flip-BACKS (reverted vs previous window), net cumulative.
- CHURN signature: high flip-back rate, stagnant net.
- STRUCTURE signature: monotone net accumulation, low revert rate.
Decides the fork left open by the flat arm-rarity curves. Also log
flip locations vs the committee table if net accumulates.

## Pre-registrations

(a) PRIMARY (conversion): post-session, the stuck-state RESOLUTION
    rate — fraction of the input worklist now solvable — beats a
    no-training control re-probe. Target: >=20% of stuck states
    resolve. (Gate/rarity read secondary; anchor guards as always.)
(b) PERSISTENCE: flip-back rate < 30% per window and net monotone
    => structure; flip-back > 60% => churn (the fp64 "extra
    learning" was noise commitment and the masters claim demotes
    to mechanism-only).
(c) Artin's ceiling theory, final form: if (a) passes with fp64 ON
    at stuck-state food, precision + aimed food = the slow-learning
    lift, and metabolic doctrine becomes "fp64 masters + practice
    food" permanently.

## Machine

3080 (cuda fp64 native) if free post-axiom-emission; Mac viable
(CPU masters ~free on unified memory — Artin's catch, 2026-07-23).

## v4.1 revisions (Artin GO 2026-07-23 2:45 PM — fast, lossless, ledger-harvested)

Wall-clock target: verdict TODAY (~2.5h session + probes), not
overnight. Ledger scan folded three banked upgrades in:

1. **HOT-BUT-GUARDED LR = 1e-5** (was 2.5e-6). Basis: the measured
   LR frontier (late-layer control rod survived 1e-4 abuse at
   71/120; 1e-5 is 10x under that ceiling) + the banked discrete-
   plasticity prong B ("flips may need hot LR / nudge
   accumulation") + absorption is moot under fp64 masters. 4x more
   conversion per wall-hour, safety = anchor + snapshot + rod
   (all live in v3, zero rollbacks tripped in 4 arm-hours).
2. **SKIP-PAIR BANKING** (macro-distillation riff, GO'd previously;
   + step-dropout riff, first live use): every practice chain that
   RESOLVES a stuck state banks both granularities — the per-step
   rows AND the (stuck_state -> final) skip pair, verified free by
   transitivity. Practice yields shortcuts and longcuts in one pass.
3. **PRE-PROBE CONTROL**: the stuck worklist is probed BEFORE the
   session (same seeds/budget as the post-probe) so resolution
   delta is paired, not absolute — a stuck state that resolves on
   re-roll variance alone must not count.

Also: rollouts START at the stuck cur (not the root — the whole
point); fresh stuck states logged live and eaten in-session
(self-feeding practice); verify cache pays compounding rent on
revisited states; persistence census every 20 min.

Pre-reg amendments: (a) resolution-rate delta (post - pre) >= +20
points on the worklist; (b/c) unchanged.
