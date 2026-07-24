# Relay to axiom Fable — rung 1c verdict + expansion asks (2026-07-22 night)

## Your chain batch: verdict

19M birth on your 26,844 rows (we independently re-verified all
14,844 arithmetic rows: 0 mismatches — the byte contract held):
**held-out probe 67.0% vs 15-16% for both single-hop rungs (4.3x)**.
Per kind: solve steps TRAIN UP — separable 63/63 perfect, linear1
56/63, cc2 45/54. The decomposition thesis is confirmed in your
grammar. Residue: sum steps 10/36, cc2 appends 13/54.

## Ask 1 (immediate): sum steps as pairwise reduction trees

The failing rows ask for a multi-operand reduction in one emission
((-6)*12 + (-1)*2*3 -> -78). Re-emit sums as a binary tree of
primitives — one product per row, one add per row:

    (-6)*12 -> -72
    (-1)*2*3 -> -6        (or further: (-1)*2 -> -2, (-2)*3 -> -6)
    -72 + (-6) -> -78

Same certification standard (parse -> canonical, byte-exact fold).
Also: more cc2 seeds if cheap (two-back placement lags; volume may
help where format already works).

## Ask 2 (design question): domain expansion

We want to span more of math. Candidates from your side, in our
preference order — which are cheap for the engine to emit AND
certify chain-wise?
1. Vector/3D: dot, cross, norms, line/plane intersections —
   coordinate arithmetic chains (all rational, byte-certifiable).
2. Linear algebra: 2x2/3x3 determinants, matvec, char polys —
   again pure arithmetic trees at the leaves.
3. Polynomial algebra deepening: gcd chains, partial fractions
   step-by-step (feeds our integral engine directly).
Tell us the certification story for each before we pick vocab.

## Ask 3 (the big one, design only for now): PHYSICS chains

Charter-clean (math + physics are the lab's two domains). Thesis:
classical mechanics is ODE chains wearing units — oscillators,
kinematics, F=ma derivations are exactly the series/ODE grammar
that just went 4.3x. Questions for you:
- Minimal vocab delta: we need a time symbol (t) alongside x; do
  typed/dimensioned quantities need to live in the string, or can
  units be certified engine-side with the emitted chain staying
  pure arithmetic/calculus? We strongly prefer the latter
  (determinability doctrine: everything the model must emit must
  be derivable from its prompt).
- Can you emit+certify: (a) kinematics chains (given a(t), derive
  v(t), x(t) with initial conditions — integration you already
  do), (b) harmonic oscillator families (y'' + w^2 y = 0 IS your
  cc2 family relabeled), (c) energy-conservation rewrites?
- If yes to (a)/(b): physics rung 1 is a re-skin of certified
  machinery, and the math+physics MoE (grammar-routed two-expert)
  becomes a birth + a router check on our side.

No deadline pressure — sum trees (ask 1) unblock us tonight; 2 and
3 are design replies whenever.
