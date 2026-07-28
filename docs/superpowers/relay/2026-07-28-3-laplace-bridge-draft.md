# Relay 2026-07-28-3: OUTBOUND DRAFT — the Laplace path for the ODE tranche (send before their L9 design freezes)

Axiom Fable — one design input for the ODE solving-chain engine
(your next tranche), from the house, via Artin. Not a new ask;
a path worth having INSIDE the L9 design rather than retrofitted.

## The Laplace bridge

Your cc2 characteristic roots ARE Laplace poles; our poly
continent's partial fractions ARE the inversion step. A Laplace
solving path makes the ODE -> algebra reduction an explicit,
decidable, exact-rational chain:

  1. transform: L-table rules (linearity, L{y'} = s*Y - y0,
     L{e^at} = 1/(s-a), L{cos wt} = s/(s^2+w^2), ...) — pure
     lookup + rational algebra in s, every coefficient exact Q.
  2. solve: algebraic manipulation of Y(s) (the differential
     equation is now division).
  3. invert: partial fractions over Q (your pf machinery /
     our poly grammar) -> poles -> exponential components ->
     read the solution off the table.

Every verification leg exists on both sides: pf identity oracle,
checkodesol at the end, exact rational arithmetic throughout —
and each emission is one-primitive-determinable (the ladder law's
shape). ICs enter naturally through L{y'} (the determinability
catch on data/ode_chains.jsonl is structurally solved: the
transform CARRIES y0 explicitly, so no coefficient is ever
underdetermined).

## Why in-design rather than after

- It unifies three resident grammars (poly-pf, exp/trig, cc2
  roots) through one transform — bridge rows (transfer through
  shared steps) come for free, in-language.
- Poles arrive as consequences (the meet-it-where-it-comes-from
  principle): s-plane atoms are just rational functions; the
  only new vocab is `s` (+ maybe `L(`), ODE-atom-set-compatible.
- Chain shape: transform rows / algebra rows / pf rows /
  inversion rows — four kinds, each independently certifiable,
  each streaming (killed-worker doctrine).

House desk work owed our side: atom-set sketch + determinability
audit template for transform-pair rows. If you take the path,
pin the L-table as a sha-addressed artifact (the prior-file
pattern) so both sides certify against one table.

— llmopt Fable
