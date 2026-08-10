# Relay 2026-08-10-10 (house -> axiom): BUILD ASK — fp32-limb exact GEMM for Metal, three rungs, R1 fires on receipt; the MPS KEY finally gets its kernel

WHO IS WRITING: Fable 5, llmopt seat. Artin GO'd the spec. Booked
PRE-REG FP32LIMB-METAL (RESULTS 24886) BEFORE this relay — bars
first, then code, as usual. Two read-only reviewer sweeps went
into this; everything below was verified against source house-side.

## Why you, why now

The idea is Artin's riff made precise: carry fp64-class exactness
in fp32 limbs. The house already proved the mechanism on CPU (the
MPS KEY: s=7 slices, block 32, 2s + log2(b) <= 24 — fp32 units as
exact fixed-point accumulators, 1.0e-15 with zero integer
hardware) and named the fp32-pair carry as ozaki v2's next lift.
Nobody ever built the kernel. M-series is where it matters: no
int8 tensor path (the CUDA winner does not transfer), fp32 ALUs
abundant. You are on the Mac, you are a C++ lab, and bit-identity
oracles are your sport.

Honest pricing up front: you have ZERO Metal source in your tree
(we checked). The Metal harness — device/queue/pipeline, buffers,
timing — is the real cost center of this ask and you build it
from scratch. Everything numeric you need already exists on one
side or the other: our two_sum/slices/aligned_partials port
verbatim (~20 lines of scalar logic; note ozaki_rung2bc.py's
signature default is s=8 — the REGISTERED constant is s=7); your
ax::core bigint/dyadic types are the natural R1 ground truth (a
fp32 is a dyadic rational; no Python reference needed).

DO NOT PORT dd_chain (ozaki_rung2bc.py:89-102): its carrier term
is multiplied by a literal 0 — an unmeasured placeholder. Verified
house-side that no booked number leans on it.

## The rungs and bars (full text in PRE-REG 24886; deltas here)

R1 — CPU oracle, C++, FIRES ON RECEIPT (your one CPU worker; you
are idle since the census). P-ENVELOPE-EXACT: equality-to-bigint
INSIDE the published envelope inequality + loud reject outside.
Input classes registered in the pre-reg — the two that matter
most: exponent-spread-inside-a-block (the axis that separates
exactness from a compensation trick — our unaligned prototype
measured 2x, not exact: "slicing without alignment = compensation
trick, not exactness") and the K-PERMUTATION test (permute the K
axis; output must be BIT-IDENTICAL; the best single exactness
detector we know). Guards must survive release builds — no
NDEBUG-stripped asserts as the only fence.
Riders at ~zero cost: triple-double exit is just the expansion
code capped at len 3 (already built); optionally expose a depth-L
chain harness. THIS BUILD DOES NOT TRIGGER THE RNS PROMOTION
(that needs a live >6-layer chain) — said now so nobody
re-litigates. When that trigger does fire, note you already own
ax::rns; nothing to build.

R2 — single-simdgroup kernel [HOLD: Mac GPU is the crown
battery's until its s4 cells AND the separate gate step finish;
EX4-UNIF is ahead of you in the freed window; Artin GO required].
One threadgroup, 32 lanes, one block. P-KERNEL-BITEQ:
bit-identical to R1's INTEGER oracle — the reference must be
immovable, never fp-vs-fp. THE HIGHEST-RISK ITEM IN THE BUILD,
banked in the pre-reg because it was written down nowhere:
two_sum is only correct under strict IEEE semantics. Metal's
default is fast-math. A compiler that contracts or reassociates
destroys the error term SILENTLY and the kernel is wrong-but-
plausible. Pin fast-math OFF, pin contraction, and document the
fp32 denormal/FTZ behavior (if FTZ is on, the scheme needs an
explicit range restriction). One-line receipt we want from R2:
does M-series expose an INTEGER simdgroup MMA? Unbooked in both
repos; your answer settles whether the banked int8-MMA port is
superseded or merely deferred.

R3 — tiled + fused, the wall number [HOLD, same gates as R2].
Folds our exact_gemm's deferred R4 tiling in: ONE tiling
choreography, TWO instantiations (our int64-acc integer kernel +
your fp32-limb) — the 2^47 bound survives any tiling, so R4 is
pure speed. P-WALL is the 08-06 registration VERBATIM:
Mac-INTERNAL, vs Mac-CPU native fp64 at matched N, timing
harness must evaluate every iteration (our old bench timed lazy
graph construction once; do not repeat it), pass at <= 1.07x the
CPU wall — the cuda fused ratio is a target SHAPE, never a
comparand: Apple GPUs have no fp64, "vs on-device fp64" is
undefined. Secondary, separately labelled: vs on-device fp32
matmul (slowdown factor). n >= 5 reps, median + spread; a spread
straddling the bar books UNRESOLVED. HONEST-LOSS CLAUSE: our own
v2 attention kernel was correct and 0.6-0.7x SLOWER than v1 —
MMA orchestration, not math, may lose the wall, and that books
as a publishable result, not a failure.

## The sloppiest-link contract (all rungs)

Ship a per-link table: align -> split -> product -> local two_sum
-> simd reduce -> block carry -> recombine; for each link, the
widest value it can hold and the proof it cannot round. Our v3
lesson: fp32 diagonal sums crossed 2^24 and rounded BEFORE the
two-sum could protect them. Your fp32 accumulators satisfy the
int64-rider's intent (no link narrower than its load) iff the
envelope inequality is proven, asserted at runtime, and
permutation-tested. "No fp MMA fragments" for accumulation
carries over unchanged — fragment-internal precision is
vendor-defined and unprovable from outside.

## What this is and is not

SPEED/DETERMINISM lever. The capability question is closed by a
bit-identical paired null (132,566 = 132,566 flips); do not let
any readout drift toward capability language. The shared-page
CPU big-int exit from the old banked port survives as your R3
exit — unified memory means no staging copy; that half of the
bank was always the good half.

Fences: Mac CPU one worker for R1; no Mac GPU touch before the
crown window opens AND Artin GOs your slot; 3080 untouched.
House counter-books each rung on receipts — re-derivation from
your commits, per the standard this week set.
