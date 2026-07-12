# Step-Level Expert Iteration Loop — Design

Approved by Artin 2026-07-12 ("approved and go"). Decisions made in
brainstorming: standalone solver (not engine proposer), frontier-
adaptive curriculum, autonomous with tripwires, cumulative retrain
from base (approach A) with a 50% engine-chain cap per round.

## Goal

A 0.5B model that, paired only with the sympy oracle (no engine at
inference), solves integrals by emitting one verified rewrite step
per call — trained entirely on verifier-approved chains, difficulty
rising at its measured frontier. Baseline (2026-07-12, RESULTS
"Step-tokens"): base model 5/30 on L2/L3 at 5% step validity vs
one-shot 0/30.

## Loop anatomy (one round = evaluate, mine, train, gate)

All phases fork-isolated (sympy pathology rules #7/#8/#10: fork is
the only real timebox), all resumable from disk.

1. **Evaluate**: oracle-gated chain solver (bench_step_tokens
   machinery, current promoted adapter) on 40 fresh problems per
   level, L2 upward; stop at the first level with solve rate <20%.
   Frontier F = highest level in the 20-80% solve band.
2. **Mine** at level F plus F-1 at 25% weight (retention): verified
   chains from (a) the model's own solved traces during evaluation
   plus extra sampling — on-policy; (b) engine replay chains
   (expert_iter_steps._chain_worker) at those levels, capped at 50%
   of the round's corpus additions. Every (cur, nxt) pair is
   oracle-verified before entering the corpus; dedupe on the pair.
   Only steps from SOLVED traces are mined (no-op steps that pass
   equivalence but make no progress cannot enter via failed stalls).
3. **Train**: LoRA from BASE weights on the full cumulative corpus.
   train_calculus recipe verbatim: r=16, all seven proj targets,
   alpha=32, loss on step tokens only, length-bucketed batches with
   per-epoch order shuffle, 3 epochs, lr 2e-4. Save
   checkpoints/step_lora_r{N}.pt (adapter-dict convention).
4. **Gate** on a held-out seed block, all levels <= F: PROMOTE iff
   no level regresses by more than 2 solves AND (frontier solves
   improve OR step validity improves >= 2 points). Rollback keeps
   the prior adapter; the corpus block stays on disk flagged
   `gate: failed` for post-mortem (never train on it again unless
   un-flagged manually).

## Tripwires (the autonomy contract)

- Two consecutive failed gates -> HALT, report.
- Mining step validity < 1% -> HALT (format death).
- Round wall > 2x budget (budget ~90 min) -> HALT.
- Every round appends one line to docs/LOOP-LOG.md: round, frontier,
  corpus size, validity %, per-level solves, gate verdict.

## Data & state

- Corpus: data/step_chains.jsonl, append-only; rows tagged
  {round, source: engine|model, level, gate}.
- Adapters: checkpoints/step_lora_r{N}.pt; promoted one copied to
  checkpoints/step_lora.pt.
- Seeds: three disjoint blocks — mining 8.0M+, evaluation 8.2M+,
  held-out gate 8.4M+ (contamination scar tissue: disjoint BLOCKS,
  never offsets within a block).

## Failure modes designed against

- Verifier gaming / degenerate no-op steps: mine only from solved
  traces; max chain length 12 at inference kills stall loops.
- Teacher swamping: 50% engine cap per round (the model must
  increasingly imitate ITSELF winning; engine steps from
  i_linear_basis one-plies aren't in the model's reachable step
  space at depth anyway).
- Forgetting: F-1 retention mix + gate's no-regression clause.
- Format drift: validity tripwire.

## Measurements (per round, into RESULTS.md at halts/milestones)

1. Step validity curve over rounds (baseline 5%).
2. Frontier level over rounds.
3. Solves vs the ENGINE at each level — the honest ceiling: where
   does verified self-training stall relative to the searcher that
   seeded it?

## Testing

- Unit: chain extraction (smoke-tested 2026-07-12), gate logic with
  synthetic scoreboards.
- One full MANUAL round before arming the autonomous loop.

## Non-goals (YAGNI)

- No engine-proposer track this spec (separate rung if wanted).
- No replay-buffer weighting (approach C) unless A plateaus.
- No model scale-up; 0.5B until the loop itself is proven.
