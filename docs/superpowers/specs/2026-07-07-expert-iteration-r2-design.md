# Expert iteration round 2: the loop closes — design

Date: 2026-07-07 (afternoon). Status: approved design (standing
autonomous authorization). Parent: CLAUDE.md expert-iteration thread;
roadmap #1 (engine complete: rules + NNUE + proposer) and #2 (frontier
curriculum). Prereq results: NNUE rho +0.937 / race 346-331; proposer
top3 99.7%, +2/+3 over random-k in hard cells.

## Question

Does retraining the proposer (and NNUE) on derivations that only the
CURRENT full engine can find — verifier-approved solutions to frontier
problems — measurably raise the next engine's frontier solve rate?
Deliverable: the first point of the curve (frontier solve rate vs
iteration round).

## Decisions

1. **mathgen level 4** (new, in `mathgen/problems.py::_expression`):
   depth-3 compositions h(g(k(x))), 3-factor products with composed
   factors, integrand families requiring chained u-sub. GUARD (repo
   scar tissue): measure the L4 expression-space collision rate over
   1k draws per kind before trusting any train/eval split; widen
   further if >1% duplicate sreprs. String seeds throughout.
2. **Frontier set:** problems where baseline (full enumeration + HCE,
   budget 200) FAILS. Harvest set: frontier problems the full engine
   (prop3 + NNUE, budget 400, 3 restarts on failure) SOLVES. Those
   solved-only-by-the-full-engine paths are the distillation target.
3. **Round-1 data:** existing 1429 rows + harvested frontier rows
   (winning-path triples, same schema); NNUE gets frontier states with
   probe labels appended. Retrain both with unchanged recipes; bump
   only data.
4. **Measurement:** held-out frontier problems (exclude-guarded),
   solve rate at budgets 100/200/400 for engine-r1 vs engine-r2. Plus
   regression guard: L1-3 race must not degrade (catastrophic
   forgetting check).
5. **One round only** until the curve point is positive (YAGNI).
6. **Pre-registered null:** near-zero frontier yield means guidance
   is no longer the binding constraint — new REWRITE RULES are
   (trig-sub, partial fractions = rung 3). Report plainly either way.

## Out of scope

Round N>1 automation, MCTS, new rule families (unless the null fires,
they become the next spec), definite integrals, GPU inference.
