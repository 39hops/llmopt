# SPEC: the exact stack (FX-V1 era) — house verification queue + axiom tools tranche

Origin: precision AMENDMENT 2026-07-27 (doctrine holds ABOVE
instrument sigma; sole retest = exact-mode paired arm) + axiom's
FX-V1 delivery (docs/specs/2026-07-27-axnn-exact.md axiom-side:
Q.16 operands, int64-accumulated exact GEMM, five-line rounding
rulebook, container-carried tables, greedy = argmax over exact
Q.32 logits, ties -> lowest token id). Axiom acceptance: two
zero-shared-code implementations (C++ + numpy-int64) bit-identical
FNV-1a battery hashes, 0/100 mismatches, both convention
variants. Speed: exact ~148ms/prompt v rounded fp32 ~162ms —
exact is ALREADY the faster path at rung-1 (ozaki lineage holds).

## House-side runs (ours, in order)

- **E1 — hash reproduction on Mac + 3080**: same container +
  spec must reproduce axiom's battery hashes (seeded fixtures
  ec93e7b9058fdffb / 7dc7062f302bf0e4). PORT TRAP (axiom-flagged,
  check FIRST in review): table-lerp upper boundary
  (idx == size-1, frac == 0) hits on EVERY softmax max element
  (d2 == 0 lands on the exp table endpoint) — must return
  t[size-1]; an index-clamp to t[size-2] diverges silently on
  essentially every prompt.
- **E2 — real-crystal export**: pairs_3e (+ S2 winner when
  born) -> AXNN container; nn_crosscheck.py (1e-4 gate) + FX-V1
  export + exact-mode battery hash published.
- **E3 — THE AMENDMENT ARM**: exact-mode gate v rounded-mode
  gate, same weights, 120 probes — the precision doctrine's sole
  named retest, at zero arithmetic noise floor. Pre-reg before
  firing; any capability delta reopens nothing by itself (one
  arm, n=1) but earns a replication.
- **E4 — solve_batch parity gate**: 200 states v our budget-150
  python solver (solved-bit exact; plies within known tie
  classes). Decides shared cache v engine=axiom namespace.
- **E5 — Fourier batch-1 adjudication** (10,000 rows emitted,
  stay-in-Q honored) — after the S2 race reads out.
- **SR birth (3080, [HOLD] until Artin GO)**: the
  stochastic-rounding bf16 birth (RESULTS L2143, open since
  07-17) — cheapest live precision experiment; house-run.

## Axiom tools tranche (asks 4-6) — DELIVERED 2026-07-27 (relay -5)

All three shipped; bridge INTERFACE_VERSION == 3, 14 pinned
names. frontier_eval 18ms forward full-verify; gate_battery
~78ms/probe (certification enforced at load; FX-V1 stepper
refactor bit-identical, hashes reproduce); certify_tables
standing (+-1 LSB, argmax-relevant monotonicity, rope circle,
midpoint fuzz; exp-underflow equal-entry fuzz trap banked).
E2's export now ships AXNN + PROMPT SPEC (format + token map) —
three consumers. Original asks kept below for the record.

## Original ask statements (asks 4-6, moved from relay -3)

4. **frontier_eval — the fused NNUE-template inner loop**:
   successors(state) -> dead_mask prune -> FX-V1 scorer forward
   over the enumerated set -> mass-descending order (R0b) ->
   optional verify_edge top-k; mirror call with predecessors()
   = the exact backward layer. The B-b pincer inner loop as one
   ms-class primitive (scorer weights ship as AXNN after the S2
   race picks a winner).
5. **Native gate-battery runner**: 120-probe greedy gate in one
   FX-V1 call (decode + oracle verify native) — kills the
   nightly gate wall; makes E3-class paired arms trivial.
6. **FX-V1 table certification fuzz**: bounded-error
   certificates v float reference (+ monotonicity where argmax
   correctness depends on it) as a standing CI node — a bad
   table IS the model definition (value-cache fossilization
   class).

## Fences

Ax-runtime v torch-runtime gate numbers never compare without a
paired arm (instrument fence, held by axiom: hashes published,
no cross-instrument comparisons). Graded magic scoring stays
GATED on S5's variance read. Exact TRAINING out of scope
(training stays torch; declared-rounding update-site design
banked as research rung only).
