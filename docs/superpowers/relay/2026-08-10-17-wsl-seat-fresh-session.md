# Relay 2026-08-10-17 (house -> axiom WSL seat, FRESH SESSION): you are the CUDA seat on the 3080 box; your prior session's rung is CLOSED and counter-booked; the no-merge gate is LIFTED — merge, tidy, and hold

WHO IS WRITING: Fable 5, llmopt seat. Your predecessor session's
work is fully counter-booked house-side (COUNTER-BOOK RNSCHAIN-C1
+ VERDICT RNSCHAIN-C2C3). This relay is your onboarding + the one
action now authorized.

## Who and where you are

You are the axiom seat ON the WSL/3080 machine (AMENDMENT
RNSCHAIN-CUDA-SEAT: Artin opened this seat; you compile and run
your own CUDA). Toolchain quirk your predecessor banked: CUDA
here is Windows-side nvcc 13.3 + MSVC 14.51, and MSVC has NO
__int128 — mulmod is a portable shift-add ladder for that reason;
keep it that way. FIRST ACTION IN ANY SESSION ON THIS BOX:
`git fetch && git status` — a stale checkout on this machine
already caused one full collision incident today (your
predecessor re-authored a module that existed on main because
this clone predated anchor-v2).

## What your predecessor shipped (all verified house-side)

Branch rnschain-cuda-2 (c870d66 + 5af1148), rebased onto main:
- C1 chain oracle: PASS all 6 classes x 5 depths, composed
  K-permutation bit-identity at depths 1 and 8; post-rebase
  digests BIT-IDENTICAL to pre-rebase (semantics-preserving port,
  and two independently-authored RNS modules agreeing to the
  digest). Collision resolved correctly: anchor-v2's rns.hpp
  survives verbatim, the duplicate died, oracle rides its API.
- C2 biteq: FIRED on the 3080 (slice kernel == naive mulmod,
  entrywise, 8 channels, N=1024).
- C3 P-BREAKEVEN: REFUTED, INSTRUMENT-SCOPED — the scalar-ladder
  shape is ~74x fp64 DEPTH-FLAT (no per-layer advantage, so no
  crossover can exist at any depth). The tensor-core 13-v-43 law
  is UNTESTED by this run. Honest loss, booked with caveats on
  both ledgers. Fused-kernel promotion correctly did not trigger.

## The one authorized action: MERGE

The relay -15 no-merge gate had three conditions: rebase onto
main, collision resolved in favor of the results-cited module,
C1 re-receipted against the survivor. ALL MET. You are cleared
to merge rnschain-cuda-2 into main (ordinary merge, push, no
force). Then delete the stale rnschain-cuda-1 remote branch —
Artin has been informed it is pre-rebase garbage; treat his
relay of this message as the GO for both actions. If the
classifier blocks the branch deletion, leave it and say so;
nothing rides on it.

## After the merge: HOLD

Nothing further is GO'd for this seat. The named follow-on (the
Montgomery/tensor-core RNS reimplementation — the real test of
the break-even law) is [HOLD]: it needs its own pre-reg and
Artin's GO; do not start it from this relay. The 3080 is
otherwise idle and stays one-worker. The Mac seat is separately
running FUNNEL-PREC (closed-loop anchor precision) and the Metal
window — not yours, do not touch.

## Standing fences (travel with every session on this box)

One worker on the 3080; long walls in the nightly window; the
llmopt checkout on this machine is a THIN EXECUTION TARGET
(write only the axiom repo); no cross-device comparisons ever
(your CUDA walls and the Mac's Metal walls are separate
instruments, permanently); commit messages carry no session
URLs; receipts stream, JSONL + shas, per the pin-3 contract.
