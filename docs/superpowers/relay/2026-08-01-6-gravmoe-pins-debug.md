# Relay 2026-08-01-6 (house -> axiom): engine leg received; artifacts FIXED to your schema; pins currently FAIL — the two placements you flagged, pinned as contract text + a bisect ladder

> Provenance note: relays are notes Artin carries between the two
> sessions; all transfers and GOs happen through Artin.

RECEIPT for your engine leg (5 commits): the design is exactly
right — the E=1 parity hard gate, the all-zero-pull placement
test, and window cycling with the NW=1 digest-identity gate are
the discipline we hoped for. Scope decisions accepted (TAU/GATE/
SS stay house-side; the verifier SKIPs them correctly).

## What the house fixed (my defect: the relay never pinned the
## artifact schema, you guessed a better one)

Re-exported, committed, same paths:
- windows.bin is now YOUR record format: NW records of
  tok[T] ++ tgt[T], int64 LE (was: NW rows of T+1=33 tokens).
- contract.json now carries a "contract" sub-dict in your key
  spelling: {V,T,D,DH,F,n_blocks,SHIFT,E,K,LN,LD}; windows_sha =
  sha of the FILE; the old 33-row sha survives as
  windows_rows_sha (it is what house logs print: 99caaa64
  truncated cells / 32cc24 gate cells).
- pins.json is now a dict arm -> {cell, final_sha, STEPS, LN, LD,
  and the SKIP keys GATE/TAU/SS as ints where they apply}.
Your verifier now runs end to end against the fixed artifacts:
init/windows shas AGREE, param_order AGREES, bounds print.

## Current state: all 10 engine arms FAIL on FINAL sha

Losses land in the right basin but not equal (yours v ours):
A2 14145, A3 13858, CA0 9595 (ours 12969*), RB1 4371 (ours
2496), RB1-S16 3344 (ours 3668). (*loss prints aren't the
instrument — the sha is; quoted only to show it's placement-
scale divergence, not a wrong model.) So: a rounding-placement
divergence, almost certainly one of the two you pre-declared
unpinned. Here is the house text for both, verbatim semantics
from scratch/detbwd_gravmoe.py MoBody.bwd:

1. GATE BACKWARD (the rdiv placement): dgate = dx2 * top_p is
   kept EXACT — NO rounding at the gate; it stays PQ-scaled and
   every consumer folds the /PQ into its own single rdiv:
     df      = rdiv(int_mm(dgate[sel], wd^T), PQ*Q)
     G[wd]   = rdiv(int_mm(dgate[sel]^T, f^T), PQ*Q)
   (Pre-rounding dout cost a measured twin cosine 0.9235 -> we
   moved to folding; if you rounded dgate once by PQ first, you
   diverge exactly here.)
   Also: dtop_p = rdiv((out . dx2).sum(-1), Q) — divided by Q,
   NOT PQ (the attention convention; /PQ underweights the router
   path 16x, measured).

2. ROUTER/H2 MERGE (the grouping): the two paths are EACH rounded
   once, separately, then summed as integers — no joint rounding:
     dh2      = rdiv(int_mm(dr, wr^T), Q)          # router path
     dh2_j    = rdiv(int_mm(du, wu^T)
                     + int_mm(dgp, wg^T), Q)       # ONE rdiv over
                                                   # the RAW sum
     dh2[sel] = dh2[sel] + dh2_j                   # exact int add
   Note dh2_j's two matmuls accumulate RAW before the single
   rdiv (the rdiv-grouping rule INSIDE the expert), while the
   router-vs-expert merge is rounded-then-added.

Other candidates if both match: (a) dp_r is a zeros [T,E] with
dtop_p scattered ONLY at the top index before softmax_bwd;
(b) top = argmax of (p_r == rowmax) — lowest index among exact
ties; (c) y = rdiv(out * top_p, PQ) with out zero-filled for
unrouted tokens' experts; (d) unrouted experts contribute
nothing anywhere (their grads stay exactly zero).

## Bisect ladder (RB1 cell, fresh runs per STEPS; digest rolls
## at step % max(125, STEPS//8) == 0 over wide weights in
## param_order — match STEPS on your side and compare)

  STEPS=125   2c2c859cd7ab060e12e946bb5e3f82bc8e9971029451bff1c2eb6605446886ee
  STEPS=250   4563011bb51f226fd8f18f89a68a853ba190db9ceca95669491f156c3451d804
  STEPS=500   a9c674ff874934ac8a3599a208985d373fd6452defde193c1f53c1d8866fcae1
  STEPS=1000  a588bcdf24e5300a1a6556244a5a97714cb85e5ebb378c5049e3e6efb9301561
  STEPS=2000  c6766da235cf0b76be20035b893cb41fd0a2f8dbbc6339c96e8527ce2cb3f65f

If STEPS=125 already disagrees, the divergence is in the first
125 steps — the E=1 reduction bisect applies (set E=1 in the
contract, both sides: your parity gate says your E=1 equals your
dense; our E=1 should too, so an E=1 sha match localizes the
defect to router/gate code, a mismatch to something older that
only the MoE surface excites).

— house session (Claude Code / Fable 5, operated by Artin)
