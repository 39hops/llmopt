# Engine optimization trio — design (bench parallelism, policy distill, sampled verification)

Date: 2026-07-07. Status: approved (standing authorization; Artin
directed order: #3 first — "speed unlocks development speed").
Three related specs in one doc; each gets its own plan when its turn
comes. Implementation order: O3 → O1 → O2.

## O3. Bench parallelism (FIRST — meta-speed)

Every bench is embarrassingly parallel across problems. One shared
helper, `llmopt/search/parallel.py::pmap`:

- `pmap(worker, items, jobs) -> list[result]` preserving item order;
  `jobs=1` bypasses multiprocessing entirely (serial fallback, same
  code path benches use today).
- multiprocessing with the **fork** context (macOS default is spawn;
  fork keeps sympy state and avoids pickling closures' modules —
  workers are pure sympy/CPU).
- Per-item wall-clock timeouts stay INSIDE the worker (SIGALRM works
  per-process in children); a worker returns a value, never raises.
- Model-backed benches (proposer race) are NOT parallelized — torch
  MPS/CUDA contexts don't fork safely; those keep `jobs=1`. CPU-only
  benches (bench_derivation, bench_ksweep, calibrate_hce) gain
  `--jobs N` (default: `os.cpu_count() - 2`).
- Determinism guard: workers receive (seed-derived) problem specs, not
  shared RNG state; results must be identical to serial runs — tested.
- Success metric: measured speedup on a fixed table (target ≥5x on
  8+ cores); recorded in the commit.

## O1. Policy distillation into features (SECOND)

The proposer buys +2/+3 solves per hard cell at 1190s-vs-16s wall
clock. Hypothesis (NNUE story, one level up): cheap structural
features can carry most of its ranking signal.

- Train `PolicyMLP`: featurize(child.expr) [+ a rule-type one-hot from
  the move label] -> score; trained to rank like the proposer
  (KL/logistic on the proposer's scores over each state's children —
  the 1429-row data re-scored by the tuned model, plus frontier rows
  when they exist).
- Slots into `beam_search(proposer=...)` unchanged — swap, never blend.
- Race: full vs rand3 vs prop3(LLM) vs prop3(distilled), solve rate
  AND wall clock. Success: ≥half the LLM's edge over random at <5% of
  its wall clock. Null pre-registered: if ranking quality collapses,
  the LLM's edge lives in semantics features can't see — that itself
  motivates rung-3 features.

## O2. Sampled edge verification (THIRD — touches soundness, so last)

Per-edge diff+simplify is the engine's metabolic cost. The rules are
property-tested; the per-edge oracle is a redundant belt.

- `verify_mode` on successors/beam_search: `"all"` (default,
  unchanged), `"sampled"` (verify each edge with probability p=0.1 +
  ALWAYS verify terminal states and every edge on the final winning
  path before reporting solved — so a reported solution is still
  fully verified end-to-end), `"terminal"` (only the winning path).
- Honest-accounting counter: corrupted-derivation detections (an edge
  that fails post-hoc winning-path verification) must be reported by
  every bench; if it's ever nonzero, the mode's risk is measured, not
  hypothetical.
- Success metric: nodes/sec speedup at equal solve-rate with zero
  winning-path verification failures across the full bench suite.
- Soundness invariant kept: no solution is ever REPORTED without its
  entire path passing the full oracle.

## Out of scope (all three)

GPU batching of sympy, process pools for model inference, caching
layers beyond what exists, async/await rewrites.
