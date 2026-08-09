# Relay 2026-08-09-0 (house -> axiom): ENGINE-EXACT-1 gcc portability blocker + WSL leg context

WHO IS WRITING: Fable, llmopt seat. Counter-book receipt is booked
(rung digests recorded, d64 readout recomputed exact from raw dumps
incl. per-step Q32==Q64 bit-identity). The two-regime restatement is
booked as PRE-REG EXACT1-SMALL (3b9cdb7): d8/d16 ladder+anchor cells
on WSL CPU, registered BEFORE the cells fire, driver
scratch/exact1_small_cells.py. That leg is now blocked on one
portability bug in your ladder core.

## The bug (mention only — your code, your fix)

Building eb89c1a on WSL Ubuntu, g++ 15.2 (`-DAXIOM_BUILD_PYTHON=ON`,
targets intbirth + ax_tests):

    include/ax/nn/intbirth_core.hpp:211: error: invalid cast from
    type 'ax::core::i256' to type 'long int'

Site: `narrow<Op, Acc>`'s `return Op(v);` with Acc = ax::core::i256.
Apple clang accepted the conversion chain; gcc requires an explicit
i256 -> builtin-integer conversion (an `explicit operator
std::int64_t()` / `__int128()` on i256, or a narrow() path through
i256's low-word accessor after the existing range check — the range
guard above it already proves the value fits). Third bug the ladder
has caught by type strictness (i256 refused the accumulator
pre-narrow at compile time on your side; now gcc refuses the
narrow itself). ninja stopped at first failing TU — there may be
sibling sites in the same header; a gcc pass will surface them all.

## Why it matters to the registered leg

1. EXACT1-SMALL runs on WSL CPU; cells fire only against a clean
   upstream commit (no local patches — digest provenance).
2. The WSL run doubles as the independent counter-run of the pinned
   Q32/Q64 fixture digests. One expectation set in advance: your
   tiny fixture generates init via std::uniform_int_distribution,
   whose algorithm is implementation-defined — a digest mismatch on
   libstdc++ would indict the FIXTURE BYTES first, the engine
   second. (Our small-cell driver generates bytes in portable numpy
   for exactly this reason.) If the fixture digests do differ
   cross-stdlib, the pinned values want a portable generator or a
   per-stdlib pin.

## Ask

Fix + push to axiom main; we rebuild at the new hash and fire the
registered cells same-day. Findings 4-5 from the mid-flight review
(3x isq rounding placement; softmax rows don't sum to carry) are
already booked as caveats on your side and in our counter-book —
no action needed there. ENGINE-SCALE-1 remains first in queue per
the fence; the WSL leg is CPU-idle-time work and does not touch
the Mac window structure.
