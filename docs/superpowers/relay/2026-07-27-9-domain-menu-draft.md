# Relay 2026-07-27-9 (DRAFT, ready to send): new-domain menu for axiom — multi-step math engines

Context for axiom: the house's born-rational birth beat its fp32
control tonight (+5 at the bar, weights exactly s*p/q); the L4
anatomy + thin union tail say the binding need is MULTI-STEP
DEPTH across MORE CONCEPTS, not width. Artin's direction: all of
the below make the cut; ODEs + number theory FIRST. Each item:
chain structure / oracle / house thread it feeds.

## Priority 1 (start here)

1. **ODE solving chains.** Separation, integrating factor,
   substitution, reduction of order — 3-8 step derivations.
   Oracle: substitute solution into the ODE, symbolic zero-check
   (verification much cheaper than solving). House side has a
   never-consumed generator (llmopt/mathgen/odes.py) for row
   cross-checks; the 19M probe died VOID-BY-VOCAB, so rows must
   ship with their atom list (VOCAB_EXTRA contract). Feeds: the
   physics rung, L4/L7-class depth.
2. **Number-theory chains.** Euclid GCD / extended Euclid,
   continued-fraction expansions, CRT reconstruction, modular
   inverse and exponentiation chains. Oracle: exact integer
   arithmetic (free). DOUBLE PAYOFF: CRT/modular = the RNS exact-
   GEMM machinery; continued fractions = the best-rational snap —
   axiom's engine work here IS infrastructure for the exact
   stack. Native fit for exact C++.

## Priority 2 (all approved, order by convenience)

3. **Exact rational linear algebra.** Row-reduction chains,
   determinant expansion, exact solves over Q; every
   intermediate a rational matrix. Oracle: exact recomputation.
   Feeds: the integer-twin world as a training domain.
4. **Recurrences + series closed forms.** Characteristic
   polynomials, telescoping, partial fractions, generating-
   function steps. Oracle: symbolic equivalence + term checks.
   Series continent already probes 99.2% — the derivation layer
   is what's missing.
5. **Mechanics derivation chains (physics).** Lagrangian ->
   Euler-Lagrange -> EOM; energy-method chains. Oracle: symbolic
   differentiation checks. Energy continent gates 100%; again
   only the multi-step layer is missing.
6. **Quaternion rotation chains (3D/4D geometry).** Compose
   rotations as unit quaternions with exact rational (Lipschitz/
   Hurwitz-integer) components; verify by applying to rational
   vectors. THE exact home of the rotational/distance instinct —
   whole-number rotations, no transcendentals. Oracle: exact
   arithmetic. Feeds: lattice-regularization program, geometry
   sector of THE EQUATION.
7. **Exact coordinate geometry.** Orientation predicates,
   intersection constructions, convex-position chains over
   rational coordinates; constructible-number field towers
   (iterated quadratic extensions) as the exactness showcase.
   Oracle: exact rational predicates.
8. **Linear-optics interference chains ("photon" physics).**
   Mach-Zehnder / beamsplitter networks: states = exact complex
   amplitude vectors in Z[zeta_8, 1/sqrt2], elements = exact
   matrices, chains = circuit evolution + interference readout.
   Oracle: exact linear algebra. On-charter quantum-physics MATH
   (no capability outside math/physics). Feeds: the cyclotomic/
   qubit-weights line.
9. **Stabilizer tableau chains.** Clifford circuit evolution by
   Gottesman-Knill tableau updates — exact GF(2) arithmetic,
   long verifiable chains, trivial oracle. Gateway to the
   Clifford+T / T-count synthesis engine (Z[zeta_8] exact —
   the shared long line with house ZX).

## Fences (non-negotiable, all domains)

- Rows STREAM out incrementally (killed workers must leave their
  rows — the checkpoint selection-effect, bit three times).
- Oracle calls fork-timeboxed (solve_isolated pattern); expiry
  is CENSORED, never False (expired-flag doctrine transports).
- Stable STRING seeds; train/eval split by exclude= prompt sets;
  widen the generator space before trusting a split.
- Determinability audit before training (underdetermined rows
  train hallucination); verified AND distinct at every layer.
- Every domain ships its atom list + serialization spec (the
  VOCAB_EXTRA contract; ODE void-by-vocab is the cautionary
  tale); diet exposure rations for resident grammars.
- Depth-first: prefer fewer concepts at L4-L7 depth over many at
  L1-L2 (the whole point of the ask).

## Logistics

3080-box CPU is open while the GPU runs house batteries; Artin
boots axiom there on his schedule. Rows land as jsonl per the
file-handoff convention; sha256 + row count + config named per
the substrate fence.
