# Spec 2026-08-06: the 3080 LOCKSTEP window (RULER slot replaced)

Status: DESIGN — fires only on Artin GO (task-hold doctrine). Drafted
from a two-agent survey (asset map + ledger sweep), every load-bearing
claim Fable-verified against the ledgers before this spec was written.

## The RULER/gist decision: KILL the queue slot, keep the library

Both agents converge and Fable verified: RULER/gist was never an
experiment — July-5 library scaffolding (llmopt/context/ruler.py +
gist.py, tested, self-contained), driver scripts/eval_ruler.py
3080-only and OFF-PIN (Qwen2.5-3B-Instruct, not the repo's 0.5B pin),
sole ledger appearance = "Fillers (non-gating)" item 5 in handoff
2026-08-03-3 line 91. ZERO RESULTS hits; CODEMAP class UNCITED.
Nothing in the current program turns on long-context retrieval; no
named spread exists for a RULER number to resolve ("prediction pays
only where variance lives"). If long-context returns it returns as a
RESIDENCY question (paged attention / prefetch), not a benchmark.
The library modules stay on the shelf as-is (tests keep them green).

NAMING FENCE (verified): GLOSSARY.md at repo root is the controlled-
vocabulary doc under the doc-integrity ratchet — a new battery must
NOT take that name. This spec's working name: LOCKSTEP (bit-identity
theme). C1 rungs live inside the existing controlled scope value
`deterministic integer battery`; no vocabulary addition needed.

## The replacement: pair a GPU leg and a CPU leg in one window

### Leg A (GPU) — INTBIRTH-SCALE-CUDA: the flagship's owed NEXT

Serves BOARD:34's explicit NEXT ("deterministic gravmoe pair ...,
plateau-break scaling on the intbirth engine"; specs
2026-08-01-deterministic-birth.md + -gravmoe.md). The rare property:
every bar is DIGEST EQUALITY — exact at n=1, immune to the ~5-solve
sigma fence, and the ONLY class of Mac result the 3080 can legally
cross-check (TIER-A A1 precedent: full 120-prompt battery digest
identical Mac/3080; GRAVMOE-SEEDS-LADDER fence: solves are not
device-free even when trajectories are).

Rungs (each pre-registers before firing, per booking discipline):
1. MB-S14-CUDA — replay the 2-block mini-LM (SHIFT=14 + integer lr
   decay) on cuda. Bar: all milestone digests + final sha identical
   to the Mac artifact. A digest miss is a FINDING (cuda integer-path
   defect), not a failure to hide. Minutes.
2. DIET-BRIDGE-CUDA — the real gen-4 next-token CE bridge on cuda.
   Bar: identical digests; secondary: booked plateau 15909->12518
   reproduces to the token. Minutes.
3. PLATEAU-BREAK — first non-trivial bar: does MB-S14's width+decay
   keep loss strictly falling past 1000 steps at diet scale? Bar
   pre-registers as a loss-trajectory SHAPE (integer, so device-free
   to read), never a gate delta. ~1 window.
4. DETERMINISTIC GRAVMOE PAIR — answers transport EXACTLY (the fp
   seed-ladder null was device-scoped because it was fp). Needs a
   Fable design pass BEFORE any window; not a window-filler.

### Leg B (CPU, concurrent) — LEAN-TIER-3: the full kernel pass

Serves the owed Lean residue (BOARD:5): sample-1000 booked
(703 compile / 297 tactic defects -> re-diagnosed 17 ring_nf-fixable
+ 7 unprovable-by-design + 4 open field_simp). Mathlib lives on the
WSL side; pure CPU throughput — runs BESIDE Leg A without contention.

Rungs:
1. Re-run the 1000-row sample under the fixed emitter (LEAN-EMITTER-
   FIX, 496/496). Bar: prefix-overshoot class vanishes; residual
   pre-registers as 7 + 4 + ring_nf-misses.
2. Full 21,914-cert kernel pass, CHUNKED (the ~100-diagnostic file
   abort silently truncated the pass TWICE — chunking is asserted,
   not assumed). ~3h CPU class; overnight shape.
3. The 4 open field_simp rows: dissect or declare a named permanent
   class.

## Held back deliberately (with reasons)

- C2 OZAKI-SPEED (int8-TC fused kernel + speculative-arithmetic
  router): real and banked (BOARD:47 "remaining = SPEED only") but
  the cost is Fable kernel-authoring hours, not GPU hours; and the
  router rung must be preceded by an ambiguity-rate measurement or
  it is vacuous. Next after LOCKSTEP if Artin wants a kernel day.

  C2 addendum — THE MAC PRECISION PATH (surveyed 2026-08-06,
  Opus agent, Fable spot-verified; corrects the relayed framing):
  - EXISTS, code-only, NEVER MEASURED: split-master training under
    MPS — scratch/metabolic_v3.py:72 `MASTER_DEV = "cpu" if dev ==
    "mps"` (fp64 masters on CPU, grads ferried per step, cast back
    after AdamW). No RESULTS entry names it; every booked metabolic
    precision verdict ran on cuda (metabolic_d2.py:43 hardcodes
    cuda). Treat as an unverdicted code path, not a proven lever.
  - PROVEN on Mac: the MPS KEY as an fp32 exact-integer CARRIER
    (packed-crystal C4, hash identical MPS v cuda) — not as an
    Ozaki matmul. llmopt/kernels/metal.py:1101 exact_gemm shipped
    correctness-first, TILING DEFERRED (no Mac wall number exists).
  - QUEUED, NEVER RUN: the MPS wall-clock race vs CPU fp64
    (RESULTS ~3880, "queued behind gen-8"; two weeks stale, no
    successor anywhere). It earns ONE slot on a future kernel day:
    Mac-INTERNAL bar only (cross-device forbidden) — Ozaki-on-MPS
    fp32 carrier (s=7, block 32) vs Mac-CPU native fp64 at matched
    N, mx.eval every timed iteration; pass at <= 1.07x the CPU
    wall (the cuda fused ratio 70.2/65.4 as a target SHAPE, never
    a comparand). PREREQUISITE: exact_gemm tiling (R4) first, or
    the untiled kernel loses the wall for unrelated reasons.
- C4 WEIGHTSPACE-POP (W1 population, attention/embedding feature
  surfaces + hold-out-family transfer): rung 1 is DESK-CHEAP on the
  existing 50-birth population and should run as a desk cell first;
  population scaling only if a surface fires (W0/W1 burned two
  registered expectations already).
- C5 PLACE-1 bandwidth on cuda: the 30B substrate does not fit 10GB;
  on the 3080 it would be a NEW uncomparable cell, not a check of
  the booked PLACE-1 numbers. Deferred until reframed standalone.
- tf32x3: looks unrun, already adjudicated (RESULTS ~3595, shelved).
  Do not re-propose.

## Window arithmetic (2026-08-06)

Today's window closes 17:00 EST. On a same-day GO: Leg A rungs 1-2
(minutes each) fit easily; Leg B rung 1 fits; Leg B rung 2 (~3h) is
an overnight-GO shape, not a today shape. Rung A4 waits on the
design pass regardless.
