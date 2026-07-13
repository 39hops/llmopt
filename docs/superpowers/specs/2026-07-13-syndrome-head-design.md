# Syndrome head on the step model (multi-task step + hints)

Provenance: the predicted-syndromes arc (RESULTS 2026-07-13, four
rounds in one day). Round 3/4 proved the 0.5B's representation
already contains the rule-fire syndrome (89.0% exact / 0.978
micro-F1 from frozen embeddings + orbital sketch). The step model IS
a 0.5B reading the same expressions — so stop computing hints
outside it.

## Design

Add a BCE syndrome head over the step model's hidden state at the
end of the `Current: {cur}` span (before the Hints line). Train
multi-task on the existing corpus:

    L = L_step (CE on answer tokens) + lam * L_syndrome (BCE, 16 bits)

Labels: `data/pred_syndrome_labels.jsonl` recipe (fork-isolated
rule-fire bits) applied to every corpus `cur`; ~2s/state amortized
once, streamed. lam swept over {0.1, 0.3, 1.0} — small, one knob.

At inference the model fills its OWN Hints line from the head
(threshold 0), then continues generation. No sympy in the prompt
path at all.

## Three payoffs, separately measurable

1. Hints go free: ~200ms fork -> 0 (the head rides the forward pass
   that was happening anyway).
2. Self-consistency: hint and step come from one representation.
3. THE BET: the auxiliary task improves step validity itself — a
   model forced to know which rules fire writes steps inside the
   span those rules define. (Auxiliary-task representation shaping;
   same family as the weight-reader's permutation lesson: teach the
   structure, don't impose it.)

## Pre-registered bars

- Head quality: >= round-4 probe (89% exact) on held-out states —
  it sees MORE than the probe (task gradient), so matching frozen
  is the floor, beating it is the expectation.
- Payoff 3 A/B (the one that decides adoption): three arms at equal
  budget on fresh problems — (a) oracle hints (today's production),
  (b) self-hints from the head, (c) multi-task model with NO hints
  line. Score: step validity + one-shot solves. Ship if (b) >= (a)
  - noise; the dream row is (c) > baseline-no-hints, which would
  mean the head improved the model even silently.
- Honest kill switch: if multi-task training REGRESSES step
  validity vs single-task at every lam (task interference at 0.5B
  scale), record the null and keep the external probe.

## Sequencing

After the 3080's coeff round lands (don't change training recipe
mid-attribution). Trains on either machine; the A/B needs the
verify oracle (fork pattern as usual). Estimated: one session.

## Relation to banked threads

- Predicted-hints A/B (BOARD): payoff-3 arm (b) IS that A/B with a
  better hint source; supersedes it if this ships.
- Syndrome dynamics (child-syndrome prediction, the world-model
  ply): same head, shifted target — natural rung 2.
- Magic-estimator revival on embeddings: same recipe, hardness
  target — rung 3 if the economics still want it.
