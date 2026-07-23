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
