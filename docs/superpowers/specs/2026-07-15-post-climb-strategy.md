# Post-climb strategy — what the week's results say to do next

State (2026-07-15 morning): GRPO run 2b promoted (2.24 -> 5.38%
held-out validity overnight, uniform level gains, +6.8k mined
steps); run 3 (curriculum ascent L3-8) LIVE; stitching tier 1 all
green; syndrome/hints/ladder nulls booked; corpus at 10.2k rows.

## What the results established (the load-bearing facts)

1. **Incremental RL on verified steps is the engine.** Six SFT
   retrains reallocated; GRPO climbed uniformly, twice, through
   rollbacks. Everything else now orbits this.
2. **Verification throughput is the binding resource** (FA Law,
   measured): the 30x oracle turned 67-min cycles into 7-min
   cycles; nothing else bought that much climbing.
3. **Knowledge-injection into prompts/representations nulls**
   (hints x2, syndrome head): the model doesn't need to be TOLD,
   it needs to be TRAINED. Policy quality is the only lever.
4. **Geometry translates across models** (stitching tier 1):
   linear bridges preserve task signal at R~0.98, cross-arch.
5. **The residual failure class is narrowing**: Arena rematch
   lost by ONE SIGN (was: whole coefficient arithmetic).

## The line of advance (priority order)

**A. Run 3 curriculum ascent (LIVE)** — L6-8 entering the band is
the first test of whether the closed system climbs into territory
the SFT era never touched. Watch: first mixed groups at L6+;
gate-L6 leaving zero. Knobs held in reserve: finishing bonus,
sign-discipline reward shaping (the Arena's one-sign miss suggests
a cheap +0.25 bonus for correct leading sign — mint only if L-level
gains stall).

**B. Self-distillation consolidation (MAI's second trick, banked
with GO).** The corpus now holds 7k GRPO-mined on-policy steps.
After run 3: one LOW-LR SFT consolidation pass on grpo-source rows
from the CURRENT promoted model (not from scratch — the lottery is
dead), gated like everything else. RL explores, SFT consolidates.

**C. Weight anatomy (Artin's bet vs mine, instruments ready).**
Adapter drawer now includes: pre-GRPO promoted, run-2b final, run-3
gates, six SFT deltas. Session: layerwise dW mass, effective rank,
CKA drift, layer-sweep probe re-run, weight-reader RL-vs-SFT
classifier. Artin's bet: elegant/simple OR interwoven-complex.
Mine (pre-registered): CONCENTRATED — mid-network mass, lower
effective rank than SFT at equal norm, reader-distinguishable.

**D. Stitching tier 2 (Mac):** Qwen3-30B-A3B with the math
keep-set as a runtime teacher — bridge its mid-layer into the
step model's layer 15 and A/B step validity with/without the
foreign vector feed. First test of geometry-as-capability (not
just geometry-as-signal). Tier 3 (GLM offline donor) waits on SSD.

**E. Domain expansion when L8 is climbing:** the ODE corpus (317
chains, quarantined) becomes run 4+'s second continent — the
closed-system game world grows a domain under the SAME climb.
Codegen port after (oracle economics favor it: seconds-per-verify
makes predicted syndromes pay there — the revive clause).

## Standing discipline (unchanged, restated)

One variable per run; gates on everything; nulls booked with
mechanisms; the oracle stays exact on the training side; promotion
only through measured gates + Artin.
