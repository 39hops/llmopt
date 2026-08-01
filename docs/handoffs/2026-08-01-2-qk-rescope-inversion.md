# Handoff 2026-08-01-2 — late night: the rescope inversion + SS null

Artin's GO was "1, 2 on mac" — Mac only. NOTHING ran on the 3080
this leg (a near-miss: a "run P4?" misread was stopped by Artin
before any launch; P4 stays [HOLD] until an explicit GO).

## Verdicts (RESULTS tail, in order)

1. **VERDICT QK-RESCOPE** (amends QK-COND): the review BLOCKER's
   owed cell INVERTED the verdict. wq/wk-only soft init = loss
   8883 -> 2496 (-72%), the biggest win in the battery; the old
   +73% failure belonged to the other four softened families.
   "Peaked attention is load-bearing" RETIRED; battery default is
   now COND=1 QK=1. Surviving legs: S16 null transports (3668);
   temperature is basin-dependent (soft basin tts [947, 502] —
   no re-sharpening). Process adoption: init pre-regs print draw
   bounds per weight family at arm start.
2. **RIDER on QK-RESCOPE**: gate leg G-RB1 = FIRST NONZERO SOLVES
   in the battery — TRAIN 2/8 sympy-verified, token-acc 56 ->
   94/140, loss 2564. The init win holds on BOTH axes. Heldout
   0/8 (memorization regime).
3. **NULL GRAVMOE-SS**: one-step scheduled sampling HALVES
   free-run token-acc (56 -> 22/140), doubles teacher-forced loss
   — capacity binds at 60k/8 rows; the exposure-bias rung moves
   to the brute steps/params leg. S0 regression sha-exact to
   G-A0 (SS code path certified read-only when off).

Cross-link worth keeping: the free-run gap SS could not treat,
the init fix largely did — "exposure bias" was substantially a
capability deficit of the saturated-attention init.

## Files

- scratch/detbwd_gravmoe.py: SS/SSW knobs (parallel scheduled
  sampling, one-step, deterministic), find_split helper.
- Logs (local): logs/qkrescope_arms_0801.log,
  logs/ss_arms_0801.log. New sha pins: RB1 c6766da2, RB3
  6968b583, RB1-S16 14981553, G-RB1 1fcfd187, S1 e1b633a9.

## Queue (next session)

1. **P4 device/lab legs [HOLD — explicit GO required]**: 3080 via
   rjob, both contracts + the NEW COND+QK default arms (pins
   above); ETA estimate before firing. Write the P4 relay to
   axiom carrying the CORRECTED contract (COND=1 QK=1 default,
   rescope pins, gate protocol) + the window-cycling
   engine-surface ask; their -4 receipt still owed.
2. **Exposure-bias brute leg**: more steps/params on the intbirth
   engine, GATE=1 standing readout (SS nulled; revive SS only at
   the scale where capacity stops binding).
3. Seeds/windows ladders if COND+QK graduates toward doctrine
   (n=1 fence stands); heldout-solve rung (first generalization)
   wants a wider diet.
4. Carried: diet meter, ckpt forensics [Artin GO], FOURIER-4,
   exact-manipulation diet share.

## Fences

- All new numbers n=1-exact; COND+QK "-72%" is one seed/window
  set; gate cells still not loss-comparable to truncated cells.
- tt basin-dependence: sharpening claims must name the basin.
