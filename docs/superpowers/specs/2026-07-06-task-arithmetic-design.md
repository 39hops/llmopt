# Task arithmetic on LoRA task vectors

**Date:** 2026-07-06
**Status:** Approved

## Goal

Treat a LoRA adapter as a task vector (ΔW = (α/r)·B·A, "the calculus
skill as a weight delta") and test the four classic task-arithmetic
operations with this repo's oracle metrics. The adapter from
scripts/train_calculus.py (checkpoints/calculus_lora.pt) is the raw
material; no experiment below trains anything new on the Mac.

## Core utility — `llmopt/train/task_vector.py`

- `load_adapter(path) -> dict[str, (A, B)]`: read the a/b tensors saved
  by train_calculus.py, keyed by module path.
- `apply_task_vector(model, adapter, scale: float, r=16, alpha=32) -> undo`:
  merge `scale * (alpha/r) * B @ A` into each target Linear's weight
  in-place; returns an `undo()` that subtracts it again (exact, since
  the delta is stored). Composition = two apply calls.
- Property tests: apply(+1) then undo restores bit-identical weights;
  apply(scale=0) is a no-op; apply is additive
  (apply(a); apply(b) == apply(a+b) on the same adapter).

## Experiments — `scripts/task_arithmetic.py`

All evals: mathgen symbolic accuracy (N=300, seed 99 — the same eval
set train_calculus.py used, so numbers are directly comparable with
the trained 65.7%-class result). General-ability control: perplexity
on a fixed small general-prose file checked at every scale — arithmetic
that buys math accuracy by lobotomizing the language model is a loss
and gets reported as one.

1. **Scaling sweep** — λ ∈ {0, 0.5, 1.0, 1.5, 2.0} on the Instruct
   model the adapter was trained on. Hypothesis on record: accuracy
   peaks at λ ≠ 1.0 (task-arithmetic folklore says slightly >1 or <1;
   if the peak is exactly the trained 1.0, say so).
2. **Negation** — λ = −1. Expect calculus accuracy well below the
   untrained baseline (targeted unlearning) with general perplexity
   moving < 10% relative. If perplexity blows up, negation here is a
   lobotomy, not surgery — report honestly.
3. **Cross-model transfer** — apply the adapter (trained on
   Qwen2.5-0.5B-Instruct) to Qwen2.5-0.5B **base** at λ ∈ {0.5, 1.0}.
   Baseline: base model's untouched accuracy. Any statistically visible
   lift is the headline; zero transfer is plausible and publishable.
4. **Composition** (Windows prerequisite) — two single-skill adapters
   (differentiate-only, integrate-only; recipe identical to
   train_calculus.py but KINDS restricted, run on the 3080). Then on
   one model: each adapter alone, both added. Success = both-added
   retains ≥ 90% of each single-adapter skill accuracy. Interference
   below that is the finding.

## Non-goals

- Full-weight task vectors (needs two full fine-tunes; LoRA deltas are
  the affordable equivalent).
- Cross-architecture transfer (GLM-scale dreams; no method exists).
- SLERP/TIES merging variants — later, only if plain addition shows
  interference in experiment 4.
