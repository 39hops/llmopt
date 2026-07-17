# Run 3: small cycles, dense readings (2026-07-17)

Base: best 45M checkpoint at launch time (consol-2 if its gate
passes, else run-2 best / consol). One variable vs run 2: the
run DESIGN (speed-first recipe, Artin 2026-07-17); diet and model
unchanged.

## Recipe (all landed in step_grpo_micro.py)

- `--groups 16` — quarter-size cycles, ~4 min mining each
- Two-tier gates: GATE_N=8 proxy every off-cycle (visibility
  only), honest GATE_N=24 every GATE_EVERY=2 (owns all verdicts)
- Snapshot-before-verdict: every honest-gate candidate saved as
  `*_candNNN.pt` before promotion/rollback
- Solves-primary criterion: promote iff solves >= best AND
  validity >= best - 2.0 (drift alarm, not veto)
- Per-level mined/groups logging every cycle

## Cadence

Honest reading every ~2 cycles (~20 min incl gate) vs run 2's
~35; proxy readings between. 24 cycles ~= run 2's 12 in wall
time, with 4x the gradient/verdict granularity.

## Pre-registered questions

1. Do small cycles climb PAST 64/120 where big cycles oscillated?
   (smaller steps at the frontier = less overshoot)
2. Does the validity autopsy's dominant failure class shrink
   during the climb, or is RL blind to it? (informs run 4 reward
   shaping — potential-shaped reward is the banked successor)
3. Snapshot drawer: do burned candidates (validity-veto class)
   look better under the solves-primary rule retroactively?

## Tripwires

Standard: HALT on two consecutive honest-gate rollbacks. New:
if 4 consecutive proxy readings drop >=3 solves below the last
honest gate, gate immediately (don't wait for the even cycle).

## Non-lossless ledger (standing)

bf16: cuda-only, ~2-pt validity debit, NOT used on Mac runs.
Proxy gate: noisy per-reading, lossless for decisions (honest
gate arbitrates). Proper packing: parity run pending on 3080.
