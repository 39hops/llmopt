# Handoff 2026-08-03-2: the F2 instrument pair

## What ran (after the F1 close, same day)
- PRE-REG V4-F2a -> VERDICT: expert recall 0.035 — BELOW the 0.0625
  random floor. Mechanism: deployed selection is topk(scores + bias),
  so most-NEGATIVE-bias residents are the most-PENALIZED experts.
  F1c's direction pin corrected. Bake-off vs measured demand:
  oracle-16 = 0.758, pos-bias = 0.069, random = 0.062, neg-bias
  (deployed) = 0.035. Offload doctrine (noaux_tc kills static load
  signals) CONFIRMED at the frontier. Mask mechanism verified;
  echo loop is amputation-consistent.
- PRE-REG V4-F2b -> VERDICT: profile-then-swap with oracle residents.
  Recall 0.345 (prediction >= 0.5 FAILED honestly): DEMAND IS
  TRAJECTORY-DEPENDENT — installing the oracle changed the text
  (hash misses 14 -> 72), which invalidated the profile that chose it.
  The attractor CHANGED (prompt-echo -> "of the" cycle, distinct-4
  count 9 -> 5): routing causally selects the text; degeneracy is
  over-determined at 6.25% residency.

## Standing instruments gained
- RECALL mode in scratch/v4flash_f1d.py: unmasked-demand logging +
  expert recall, equivalence-fenced. The standard readout for ANY
  residency rule, judged on the trajectory it CREATES.
- Demand ground truth (jsonl rows 6-7) for the from-disk keep-set
  bake-off (bias/norms/SVD/FFT predictors — Artin's weight-plane
  question, banked in RIFF-LEDGER with killers).

## Open, pre-reg-gated
K-sweep on fp8-dense base (coverage-of-many-trajectories); dynamic
score-layer fetch-on-miss; prompt-lookup spec decode (speed, in-tree);
VQ/B-tree codebook cell (Artin's riff, grounded by the pooled table);
rung 13; tilelang leg needs py3.11.

## Cautions
Disk ~3 GB free (F1 cache ~24 GB; Artin decides keep-vs-delete).
F2b tok/s (0.114) depressed by disk pressure — not comparable.
Reviewer seats self-reported Opus 4.5 twice today.

main @ 4f7b3e5 + this commit; 451 tests; no jobs running.
