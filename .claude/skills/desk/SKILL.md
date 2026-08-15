---
name: desk
description: Price a proposed rung with a zero-cost census before spending a machine on it - name the threshold first, count deterministically, book the number as an OBSERVATION whether it kills or promotes the rung. Use for "is this worth running", "how much is even there", "census", "what fraction of rows", or whenever a proposal's value depends on a countable property of data already on disk.
---

# The desk census (pricing a rung with arithmetic)

Origin: 2026-08-15, TREE-CENSUS-0. An external seat proposed
shared-prefix (tree) training on the strength of a paper reporting
3.9x. Before writing any of it, a census counted the duplicated
prefix mass in the actual diet: **4.48% of linear FLOPs, 1.91% of
attention**. The rung died the same afternoon for zero training
cost, and the idea was parked with a named condition for revival.

`/probe` moves a question from a training run down to existing
checkpoints. `/desk` is the step below that: questions answerable by
COUNTING what is already on disk, with no model in the loop at all.

## When this applies

The tell is a proposal whose value rests on a countable property:
"most rows are redundant", "these prefixes repeat", "half the shard
is one rule family", "the levels are imbalanced". Every one of those
is a number, and the number usually exists before the idea does.

## The ritual

1. **Name the threshold BEFORE counting.** Write down what result
   kills the rung and what result promotes it. TREE-CENSUS-0's was
   "under ~10% and the machinery is not worth it" — recorded first,
   so the 4.48% decided the rung instead of being argued about
   after. A threshold chosen after seeing the number is not a
   threshold.
2. **Count the thing that actually costs.** Rows are rarely the
   unit that matters. Weight the census by what the proposal spends:
   FLOPs, tokens, wall-clock, memory. TREE-CENSUS-0's first draft
   counted duplicated TOKENS; the refinement that made it decisive
   was splitting linear-FLOP from attention-FLOP reuse, because
   attention is quadratic and the two answers differ by 2.3x.
3. **Derive from the real artifact, never a summary.** Load the
   frozen shard, the excised diet, the actual encoded sequences.
   Desk arithmetic on a remembered number is how a pin drifts: the
   soft-collapse desk figure (143,391) and the instrument's measured
   one (143,571) differ because encoding drops rows the desk model
   did not know about.
4. **Book it as an OBSERVATION**, kill or promote. A census that
   kills a rung is a result and belongs in RESULTS.md with its
   fences (what the FLOP model ignores, what regime it covers) and a
   FINDINGS bullet. The ratchet counts it, so budget the bullet.
5. **Name the revival condition when it dies.** TREE-CENSUS-0 books
   "parked for tree-shaped data (search trajectories, rollouts)
   where the shared prefix dominates the sequence" — the idea stays
   alive with a stated trigger instead of being silently dropped.

## Fences that travel with a census

- State the model behind any derived quantity. "FLOPs" means a
  specific cost model (token-linear plus L^2 attention, kernel and
  batching constants ignored) — write it down, because a reader will
  otherwise assume measured wall-clock.
- A census is deterministic, so it needs no seed fence — but it is
  scoped to the exact artifact counted. Name the file and its row
  count.
- Desk numbers are PREDICTIONS about an instrument's behavior, not
  measurements of it. When the instrument later reports its own
  count, book the instrument's number and note the delta.

## Cheap and decisive beats thorough

The value here is the ratio: a census costs minutes and can retire
a multi-day implementation. Run it before the argument about whether
to implement, not after.
