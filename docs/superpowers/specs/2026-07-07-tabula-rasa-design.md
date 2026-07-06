# Tabula rasa: the AlphaZero-way ablation — design

Date: 2026-07-07 (evening). Status: approved design (standing
authorization; Artin: "we can spec it now"). Origin: Artin's taxonomy
question — we trained the Stockfish way (learned components
bootstrapped from HCE-guided search) and are mid-first-turn of the
bootstrapped self-improvement loop; the untested paradigm is
AlphaZero's: NO hand-crafted bootstrap at all.

## Question

Does a from-scratch loop — random policy, no HCE labels, learning
purely from verified outcomes of its own random search — converge to
the bootstrapped engine's capability, and at what round/compute cost?
Deliverable: the bootstrapped-vs-tabula-rasa curve (frontier solve
rate per round, both lineages on the same held-out sets).

## Decisions

1. **Round 0 engine: pure random.** Search = k1-random dives with
   restarts (measured traction: sweep showed random depth solves most
   of L1-2 alone), eval = count_ops only (tie-break, no HCE unsolved
   weighting — that's hand-crafted knowledge), no proposer. The
   verifier stays (sympy IS the game rules; AlphaZero kept the rules).
2. **Curriculum emerges, not imposed:** each round samples problems
   from ALL levels 1-4; whatever the current engine solves becomes
   training data (winning-path rows + probe-labeled states). No
   frontier definition needed — early rounds naturally harvest easy
   problems, later rounds harder ones. This mirrors AlphaZero's
   implicit curriculum from self-play strength.
3. **Per round:** harvest (fixed problem budget, e.g. 240 problems) ->
   train proposer-tr and nnue-tr from scratch on ALL rows accumulated
   so far (no warm start; retrain-from-accumulated is the AZ recipe)
   -> next round searches with adaptive-k(T=0.1) using the new nets.
4. **Measurement:** after each round, solve rate on the SAME held-out
   sets the bootstrapped lineage uses (L1-3 race seeds + eir2-eval L4
   seeds), so the two curves share axes. Also track rows-harvested per
   round (the df/dn estimate from the limit discussion).
5. **Stop rule:** 3 rounds or until round-over-round held-out delta
   < 2 problems (whichever first) — bounded compute, honest asymptote
   estimate.
6. **Pre-registered outcomes:** (a) tabula rasa converges to
   bootstrapped-class capability in ≤3 rounds -> hand-crafted bootstrap
   was a shortcut, not a ceiling; (b) it stalls below -> the bootstrap
   carried knowledge random search cannot recover at this compute
   scale; (c) it EXCEEDS -> HCE bias was limiting the bootstrapped
   lineage. All three are findings.

## Out of scope

Value-network MCTS (beam stays), multi-agent play, L5+ problems,
compute beyond 3 rounds.
