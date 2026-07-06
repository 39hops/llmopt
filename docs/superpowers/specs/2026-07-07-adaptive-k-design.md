# Adaptive branching: confidence-gated depth vs breadth — design

Date: 2026-07-07. Status: approved design (standing authorization).
Origin: Artin's synthesis after the k-sweep + proposer race — "there
needs to be something else determining whether to continue depth or
breadth." Evidence base: k1x3 restarts dominate where any deep line
works (diff L2@50: 18/20 vs full 7/20); full breadth wins where the
RIGHT move must be kept (int L3@50: full 13/20 vs k3 10/20); the
trained proposer's biggest edge over random is exactly in that
semantic-choice regime (int L3@25: +3).

## Question

Does per-state adaptive k — prune hard when the ranker is confident,
widen when it is uncertain — beat every fixed k AND full enumeration
at equal node budget?

## Mechanism

- Ranker emits scores s_i over a state's children (proposer scores at
  first; the distilled PolicyMLP from spec O1 once it exists).
- Normalized entropy H = entropy(softmax(s)) / log(n_children) in
  [0, 1].
- k(state) = k_min + round(H * (k_max - k_min)), defaults k_min=1,
  k_max=6. Peaked ranking -> single deep line; flat ranking -> hedge.
- Implemented as a `propose_k` generalization: `propose_k` may be an
  int (current behavior) or a callable (state, ranked_children,
  scores) -> int. Engine change is ~5 lines; the entropy policy lives
  beside the proposer in `llmopt/search/proposer.py`.
- Temperature of the softmax is ONE tunable; calibrate on the
  train-split states, never on race problems.

## Measurement

Same race protocol (held-out `proposer-race-*`, budgets 25/50/100/200,
both kinds): adaptive-k vs best-fixed-k vs full vs k1x3-restarts.
Success = strictly best aggregate at >=2 budgets, with the win
concentrated where the sweep says each pure strategy fails (adaptive
should match k1x3 on diff L2-3 AND match full on int L3). Honest null
pre-registered: if entropy doesn't localize uncertainty (flat
everywhere or peaked everywhere), adaptive collapses to a fixed k and
we report that the confidence signal, not the mechanism, is the gap.

## Out of scope

Full PUCT/MCTS, learned k-policies (entropy first — simplest thing
that could work), restart scheduling (separate composition, later),
budget reallocation across problems.
