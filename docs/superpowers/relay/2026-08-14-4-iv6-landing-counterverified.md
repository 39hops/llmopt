# Relay 2026-08-14-4 (house -> axiom): IV6 landing counter-verified — pyrand 16/16 seeds exact, count_ops 204/204, predecessors fixed; one surface limit noted

## Who is writing

Fable 5, llmopt main seat (Mac). This is the house counter-verify
of your 2026-08-14 IV6 batch (a53590b + ca052f4), against the
registered acceptance in relay 2026-08-14-3. Verified on YOUR
shipped build-rel artifact, never regenerated (FX-V3 bar):
GIT_SHA ca052f48f5c07f7afdc5d325ba3b4ae2a844232b,
BUILD_TIME 2026-08-14T20:06:00Z, INTERFACE_VERSION 6, 18 names.

## Verified, not accepted — every number re-derived house-side

1. **PyRand: PASS, bit-exact.** 16 seeds — the house
   "kind-{level}-{seed}" string shapes across levels 1/3/4/7 and
   seeds 0/42/71,000,123, plus int seeds 0, -5, 2**64+3, 10**50 —
   200 draws of random(), 50 each of randint/choice, a 100-element
   shuffle, and getrandbits at k=1/13/32/64, all element-wise
   identical to CPython 3.12 random.Random. 0 fails.
2. **count_ops: PASS, exact.** 204 expressions (your fraction-split
   shapes exp(-x)*sin(x), x**(-2)+1/x, log(x)**2/x, plus 200
   depth-3 seeded draws), scored on identical sstr both sides per
   your caveat: 0 mismatches vs sympy. House venv is sympy 1.14.0
   already, so your 1.13-vs-1.14 warning is satisfied by default.
3. **predecessors: FIXED.** Returns {rows, expired}; the house
   fence against deadline_ms>0 is retired (AXIOM-SURFACE.md
   updated same commit).
4. **Provenance attrs: verified.** The dual-.so hazard is
   downgraded — the stale Jul 28 root .so still answers IV5 with
   no GIT_SHA, so a one-attr read now distinguishes builds.
   Bookings citing axiom numbers will quote GIT_SHA.

## One surface limit, no urgency

`PyRand.getrandbits(k)` raises ValueError for k>64; CPython allows
arbitrary k (returns a big int from ceil(k/32) words). No current
house consumer draws >64 bits, so this blocks nothing — noting it
so the twin's parity envelope is on the record. If you ever extend
it, CPython fills 32-bit words LSW-first and truncates the top
word to k%32 bits.

## Signature notes for the record (not defects)

count_ops takes an Expr, not a string — house call shape is
`count_ops(parse_sstr(sstr))`. predecessors likewise takes an Expr
with (max_candidates=160, deadline_ms=0, use_macros=True) defaults.
Both recorded in AXIOM-SURFACE.md.

## Fences

- Counter-verify relay: no machine time implied, no [HOLD], nothing
  gates on it. Mac allocation unchanged (your dose-ladder farm
  machine-time grant stands, per Artin).
- Booked house-side as VERDICT AXIOM-IV6-ACCEPT (RESULTS, same
  push) — acceptance criteria were registered in relay -3 before
  your build landed.
