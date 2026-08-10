# Handoff 2026-08-10-1 — ceiling-swap closed both ways, ambiguity law replicated, two axiom seats live, crown mid-swap

Resume: BOARD header -> this file -> RESULTS tail. Prior:
2026-08-10-0 (morning; gcd wall, cofactor GO). This is the
EVENING handoff of the same date.

## LIVE RIGHT NOW (check these first post-compact)
- CROWN BATTERY: 4/6 cells done; c_s4 RUNNING (mps, 3 epochs
  ~52 min each); a WATCHER (background task, this session) fires
  on c_s4's "deployed T" line and KILLS the loop at the cell
  boundary (NOTE CROWN-WINDOW-SWAP booked — Artin GO'd
  interleaving the Metal window there). SEQUENCE ON FIRE: kill
  loop -> Artin relays Metal ping to mac-axiom (--gpu-ok; run
  order ftz -> biteq -> wall) -> relaunch m_s4 as its own rjob
  (ARM=m SEED=4 scratch/rev3_crown.py, log to
  logs/revive/crown_m_s4.log) -> after all six cells: the
  SEPARATE gate-eval step -> crown books (booking notes in
  handoff -0: tie language +-2.3/seed, n=2 fallback, single-seed
  dominance check). If the session died and the watcher with it:
  check logs/revive/crown_c_s4.log for "deployed T" and do the
  kill/relaunch by hand (pid was 11734, bash loop).
- mac-axiom seat: has TWO jobs — the armed Metal dispatch
  (relay -13/-16 precedence: ping preempts) and PRE-REG
  FUNNEL-PREC (closed-loop anchor precision, relay -16, GO'd).
- wsl-axiom seat: DONE and HOLDing — rnschain-cuda-2 MERGED to
  axiom main (16dd86e..e06ca6a), stale branch deleted, checkout
  clean. Montgomery/TC reimplementation [HOLD, needs pre-reg +
  GO]. Fresh session onboards via relay -17.

## THE DAY'S ARC (evening half; RESULTS 24379 -> ~25400)
1. Cofactor rung CLOSED: gate NOT-APPLICABLE (24379) + census
   |r|-refutation (24452). Open problem standing: bound c
   without reduction.
2. CEILING-SWAP program (Artin's theory) run to ground in one
   day, all CPU: 0A hardness-null (rho +0.18), 0B depth-null
   (FORK-ONLY, chains die in 1-3 plies), 0C P-AMBIGUITY FIRES
   (rho -0.9) + 0C-R replication (d128 rho -1.0). THE AMBIGUITY
   LAW [REPLICATED, math crystals]: margin is an inverse readout
   of the data's local branching factor; zero tie events at
   n_succ=1 (d256). Theory row waits on a SECOND DOMAIN.
   Rung B (2x2) recommended NO-FIRE (9.65% choice-free = rations
   violation). Training-side survivors: BRIDGE-1, exact-mode
   slot (RESULTS 7226), gradient-spectrum reading.
3. ATTRACTOR-0 (Artin's reddit/Collatz riff): single basin —
   198/198 absorb to ANSWER-FORM, median 1 step, absorbing
   margin 2.25 v 0.85 transit. X=>X identity attractor ABSENT
   free-running -> located in the search scaffolding. Named
   unrun arms: iterate-through-answer-form; correctness rider.
4. FP32LIMB (Artin's register riff -> banked -> pre-reg 24886):
   R1 counter-booked (envelope exact, LSB-not-spread contract,
   _f24), R2/R3 BUILT + interlocked (MMA answered by compile
   probe: NO integer simdgroup MMA on M-series — int8-MMA bank
   superseded), dispatch awaits crown window.
5. RNSCHAIN-CUDA: C1 pass (after stale-checkout collision -> 
   anchor-v2 rns.hpp survives, digests bit-identical across
   rebase), C2 biteq FIRES on 3080, C3 P-BREAKEVEN REFUTED
   INSTRUMENT-SCOPED (scalar ladder 74x depth-flat; TC law
   untested; promotion correctly no-fire). MSVC no-__int128
   hazard banked.
6. FUNNEL-PREC pre-reg'd + GO'd (the typo rung): closed-loop
   precision, P-CLOSED-INVARIANT = path-invariance by
   construction; mac-axiom builds.
7. Charter reaffirmed post-safeguards-article: NO bio/protein
   engines ever; landscape MATHEMATICS banked as methods
   (funnel x3 meanings, frustration-metric rung sketched).
8. Process: /relay + /counterbook skills shipped; reviewer agent
   opus-5[1m] high cap 5; hookify rules (pipe-gated commit
   BLOCK, unguarded scratch import WARN); doctor run (7 dup
   plugins disabled); seat taxonomy in memory
   (machines-and-seats).

## DEFECTS CAUGHT TODAY (both directions, keep the streak)
House: pipe-gated commit slipped once (rule now a hook); str-v-
int recount false alarm (fit-the-artifact includes types);
farm_dist_rows import ran the farm (inline + hookify warn).
Cross-lab: stale checkout -> module collision (counter-corrected
their premise correction; frozen-evidence rule held the merge).

## QUEUED / HOLDS
[HOLD] Montgomery/TC RNS (wsl-axiom, needs pre-reg).
[HOLD] DATA-CEIL rung B (recommended no-fire as designed).
EX4-UNIF behind the Metal window. ENGINE-EXACT-2 pre-reg still
unwritten. Ambiguity-law second domain (ZX needs seed knob or
physics enumerator). Attractor follow-ups (through-answer-form;
correctness rider). Frustration metric (desk, banked).
BRIDGE-1. Seat tabulation. 12 [UNVERIFIED] specs-INDEX rows.

## STATE
483+ tests green at every commit (rc-gated). FINDINGS ratchet
under 320 after three same-commit curations. Everything pushed
through 9decad8. Untracked: axiom artifact drops only. 3080:
idle post-merge, wsl-axiom holds. Receipts under logs/data_ceil/
committed (seedslad pattern, booked per entry).
