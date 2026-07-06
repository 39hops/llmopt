# NNUE rung: learned eval for the derivation search — design

Date: 2026-07-07 (morning session, autonomous per standing
authorization; "go"). Parent: roadmap #1 "NNUE moment"; baselines from
the calibration work: rho(diff)=+0.685, rho(int)=+0.712.

## Question

Can a small net over cheap structural features order states better
than HCE v0 — measured by search performance, not by loss?

## Decisions

1. **Interface: `eval_fn` parameter.** `beam_search(..., eval_fn=hce)`
   and `successors` stay eval-free; the eval is swapped, never blended
   (locked rung-1 decision). Default unchanged so all existing
   callers/tests behave identically.
2. **Features: hand-coded structural vector (~20 dims)** in
   `llmopt/search/features.py`: counts of Integral / Derivative / Subs
   / Add / Mul / Pow / sin / cos / tan / exp / log atoms; count_ops of
   the whole expression; count_ops of the largest and the summed
   unsolved subtrees; number of unsolved atoms; max Integral nesting
   (limits length); tree depth; free-symbol count. `State.plies` is
   deliberately EXCLUDED: probes restart fresh, so history cannot
   affect solvability — including it would leak the label. Rejected:
   GNN/token encoders (10^3-state data scale; NNUE lesson is cheap
   features + tiny net).
3. **Labels: log2(nodes-to-solve)** from probes (beam, width 8,
   max_plies 20) capped at 200 nodes and 60 s wall (BaseException
   alarm, per the calibration guards); failures and wall-hits labeled
   log2(400). MSE regression. Rejected: pairwise rank loss (complexity
   without expected gain at this scale).
4. **Data**: on-policy states via the trace hook, both kinds, levels
   1-3. Train seeds `nnue-train-{kind}-{level}-{i}`, eval seeds
   `nnue-eval-{kind}-{level}-{i}`; explicit exclude= guard — drop eval
   problems whose root srepr appears in the train root set (repo
   contamination lesson: never trust seed disjointness alone). Target
   ~1500 train states, ~300 eval states after dedup.
5. **Model**: torch MLP 20→64→64→1, ReLU, Adam, feature
   standardization stored in the checkpoint
   (`checkpoints/nnue_eval.pt`); CPU-only, seconds to train.
6. **The race (scripts/bench_nnue.py)**: (a) Spearman rho of the net's
   prediction vs measured nodes-to-solve on held-out states, reported
   next to HCE's rho on the same states; (b) decisive heat: solve rate
   on held-out problems at node budgets 25/50/100/200, both kinds,
   eval_fn=hce vs eval_fn=nnue. Score by running the eval inside the
   search — never by training loss (weight-reader lesson, CLAUDE.md).
7. **Pre-registered risk**: a null (net ≤ HCE) is a finding — it says
   HCE v0 is near-ceiling for structural features and the next eval
   rung needs semantic features. Report it plainly.

## Out of scope

Model-as-move-proposer, GNN/transformer evals, HCE weight tuning,
self-play data loops, GPU.
