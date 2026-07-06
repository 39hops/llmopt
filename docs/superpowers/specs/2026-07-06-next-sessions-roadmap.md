# Next-sessions roadmap (drafted 2026-07-06, end of Mac session)

Status: DRAFT backlog — each item needs its own brainstorm→spec pass
before implementation. Ordered by (my) expected value. Origin: session
discussion after the weight-reader result and the task-arithmetic
build-out.

## 1. Derivation search engine with hand-crafted evaluation ("Stockfish for math")

The step-level-search goal from CLAUDE.md's expert-iteration thread,
made concrete. Chess-engine mapping: position = derivation state (a
sympy expression + goal); move = rewrite (factor, substitute,
integrate-by-parts, u-sub, simplify-subtree); move generator = the
fine-tuned small model proposing candidate rewrites; **HCE = hand-
crafted evaluation of a state**, e.g. expression-tree node count and
its trend, distance-to-solved-form heuristics, count of unreduced
subexpressions, penalty for revisited states; search = beam or MCTS
over the rewrite tree. Oracle: every edge sympy-verified (illegal
moves are impossible, unlike RLHF-style setups); terminal states
checked exactly.

- First rung: differentiation/integration states only, 5-10 rewrite
  move types, beam width 8, HCE v0 = tree-size + solved-form distance.
- Calibrate the HCE like chess did: does eval(state) correlate with
  eventual solve rate from that state? (We can measure this — chess
  couldn't for decades.)
- Success metric: solve rate on problems the greedy model fails alone;
  the interesting chart is solve-rate vs search budget (nodes).
- Later: NNUE moment — replace HCE with a small learned eval trained
  on search outcomes, compare.

## 2. mathgen frontier curriculum (grow the data, find the limit)

- Wire the already-landed but untrained kinds into the recipe:
  mathgen/odes.py, mathgen/linalg.py.
- New families with traced reasoning: multi-step chain rule, Taylor
  series, definite integrals with substitution traces, multivariable
  partials. Traces follow the make_limit_traced pattern (steps in the
  answer format, metric scores only the final line).
- Frontier generation: instead of fixed levels, generate at the
  difficulty where the current model scores ~50%, train on verified
  solutions, re-measure, raise. The deliverable is the *curve*:
  attainable difficulty vs training rounds (and vs model size, with
  the 3B cached for comparison).
- Guard rails from CLAUDE.md apply doubly at higher difficulty: string
  seeds, exclude= splits, widen generator space before trusting it.

## 3. Rotate-then-quantize (numerically better weight arrangements)

The measurable version of "is there a mathematically better
arrangement of the same weights": orthogonal rotations (QuaRot-style
incoherence processing) change every numeric value, preserve the
function exactly, and should reduce quantization error. Repo has
quantize/ already.

- Experiment: quantize W vs quantize (Q W) with random/Hadamard Q at
  int8/int4; score by (a) per-layer reconstruction error, (b) decode
  accuracy on mathgen with the quantized model. Oracle-checked, cheap,
  CPU-friendly for (a).
- Honest kill-switch: if int8 shows no measurable accuracy gap to
  fp16 anywhere (likely), int4/int3 is where the experiment lives.

## 4. Weight-space rungs 3-4 (generator; LoRA reader)

- Rung 3 (generator): predict subject-MLP weights from function
  description; score by *running the generated net* against the oracle
  function (MSE), not by weight distance — weight distance is the
  wrong metric per the permutation result. Tonight's finding
  (augmentation > canonicalization, 88.4%) says don't canonicalize
  targets; train with permutation-augmented supervision or a
  set-structured loss.
- Rung 4 (LoRA reader): classify domain / predict eval accuracy of
  r=16 adapters from their weights. We now manufacture adapters as a
  byproduct (calculus, diff-only, int-only + future curriculum runs) —
  collect them; ~10s of adapters won't train a reader but sets up the
  harness; generating hundreds of cheap adapters (tiny rank, short
  training, varied kinds/seeds) is the data plan.

## 5. Carry-overs (small, unblocked)

- Physical expert slicing for the MoE if the masked-pruning chart says
  quality holds (depends on tonight's queued runs).
- moe/offload.py ExpertCache seeded with domain keep-sets
  ("domain-aware expert prefetch").
- Flash prefill attention port to Metal (kernels item; needs quiet GPU
  for benches).
- Metal learning track (Artin): tree-reduction exercise in
  ~/practice.py, then read _ATTN_PARTIAL_SRC end to end.
