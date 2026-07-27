# Relay 2026-07-27-3: axiom's tranche-2 reply (recorded) + house follow-ups + the exact-layers delta

NOTE: axiom answered the relay -2 doc AS OF commit 564d8d0 — the
1e-4 instrument fence and rung 2b (exact inference mode) were
amended in AFTER Artin relayed (a4b8282). The delta rides below
as the next send.

## Recorded deliveries (axiom commits a628297, 9d279aa, e1f512c; suite 456/456)

1. **solve_batch DELIVERED**: pool-parallel, GIL-released,
   pure-native rules -> deterministic labels; deadline expiries
   report solved=False. Measured 23 states/s (197/200 qual roots
   in 8.7s) — the S2-class job drops ~1h -> ~15min. Parity fence
   honored in-code: labels cache under engine=axiom until OUR
   200-state agreement gate passes.
2. **ax::nn rung 1 ACCEPTANCE PASS**: max |dlogit| 1.0e-07 v
   torch fp32 (bar 1e-4), 100 prompts, d256/8L/ffn1024/h4
   vocab-47, two convention variants; all conventions
   container-declared (AXNN v1) and validated at load. Honest
   speed: ~162ms/prompt (naive rung-1 loops, double
   accumulation); SIMD/quant later under own gates.
3. **Bridge v2 PINNED**: axiom_sym.INTERFACE_VERSION == 2,
   twelve pinned names incl predecessors + solve_batch;
   arm-time assertion supported.
4. **Fourier batch 1 EMITTED**: 10,000 rows (ZX batch-1 scale),
   stay-in-Q honored (Fraction coeffs, amp-phase excluded),
   organic kind mix, structural boundary anchoring, per-row
   native parse + 16-pt soundness (zero aborts) + dedupe. At
   data/fourier/fourier_batch1.jsonl awaiting OUR batch-1 gate.

## House follow-ups (queued, ours)

- Run the 200-state solve_batch agreement gate v our budget-150
  python solver (pre-reg before firing; decides shared-cache v
  engine=axiom namespace).
- Point nn_crosscheck.py at real crystal weights (export
  pairs_3e + the S2 winner when born) — acceptance re-run on
  OUR tensors, then gates batteries migrate under a paired-arm
  fence (1e-4 runtime = different instrument).
- Fourier batch-1 adjudication (ZX playbook) after the S2 race.
- knock-4 [HOLD] unchanged.

## THE DELTA TO SEND (what axiom has not seen)

(a) The 1e-4 instrument fence — already half-honored by their
1.0e-07 result, but the doctrine line stands: ax-gate v
torch-gate numbers never compare without a paired arm.
(b) **Rung 2b — EXACT inference mode** (now doctrine-linked: the
precision-doctrine AMENDMENT 2026-07-27 names this as the SOLE
retest condition, one paired arm at sub-sigma resolution).
Requirements for precise layers, concretely:
  - **Exact GEMM**: fp32 weights/activations are dyadic
    rationals; int-sliced fixed-point products + wide-integer
    accumulation (associative -> order-free). Our
    scratch/ozaki_* lineage: int8-sliced exact beat native fp64
    on SPEED — this is not a slow mode.
  - **Declared nonlinearities**: exp/gelu/rsqrt are
    transcendental — replace with DECLARED exactly-computable
    fixed-point forms (polynomial/table + stated rounding rule)
    that become part of the model definition, container-declared
    like every AXNN convention.
  - **Norm layers**: mean/variance exact in rationals; 1/sqrt(v)
    is the one irrational — declared fixed-point Newton form.
  - **Greedy gate mode needs NO softmax**: argmax over exact
    logits is exact comparison — the entire gate battery can be
    bit-exact without ever normalizing. Sampling mode (declared
    exp + exact division + integer threshold sampling) can come
    later under its own gate.
  - ACCEPTANCE: bit-identical logit hashes across Mac / 3080 /
    axiom C++ on 100 prompts; then the house runs the amendment
    arm (exact-mode gate v rounded gate, same weights).

## Axiom's first response to the delta (recorded)

FX-V1 accepted as well-defined: integer-only profile, every
transcendental container-shipped as a table, every rounding
declared — bit-identity by construction. CORRECTION BANKED
(axiom caught the house's gloss): "greedy needs no softmax" is
true of the OUTPUT HEAD only — softmax lives inside every
attention layer regardless of decode mode; FX-V1 handles it with
a declared exp table + floor division. House claim amended
accordingly.
