# Measured history

This appendix preserves earlier mathematics and physics measurements that were
unique to the old README and still have exact support in the living verdict
ledger. It is deliberately curated rather than a copy of that chronology.

For citation, name the exact repository commit SHA and the exact verdict entry
linked beside the number. Current closed-system history is not repeated here;
use [`FINDINGS.md`](FINDINGS.md) for the maturity- and scope-tagged catalog and
[`RESULTS.md`](RESULTS.md) for amendments, nulls, and full receipts.

## Quantum-circuit search

- **`T-count engine, day one (rungs 0-2, ZX/pyzx)`** is the exact ledger
  entry supporting this retained arc: the registered random-circuit search
  first returned `0` wins, `30/30` exact ties, and `0` verification failures;
  an independent-device check again returned `30/30` ties. After
  extraction-aware scoring and a phase-polynomial macro were added, the
  search reached `9/30` verified wins against greedy and `7/30` against the
  bare pipeline on the registered rung. See
  [`docs/RESULTS.md` verdict: T-count engine, day one (rungs 0-2, ZX/pyzx)](RESULTS.md#t-count-engine-day-one-rungs-0-2-zxpyzx).

- The same named verdict records the honest boundary: unextractable diagrams
  were not counted as reduced circuits, and the first extraction-scored race
  produced `0` wins, `9` ties, and `11` timeout losses. Those losses are part
  of the result, not discarded rows. See
  [`docs/RESULTS.md` verdict: T-count engine, day one (rungs 0-2, ZX/pyzx)](RESULTS.md#t-count-engine-day-one-rungs-0-2-zxpyzx).

## Model-Hamiltonian ground states

- **`Variational ground-state engine, rung 1: the referee is a theorem`** is
  the exact ledger entry supporting the retained TFIM result. On the declared
  `n=10` model Hamiltonian, the hardware-efficient ansatz missed the `<1%`
  critical-point bar at `1.314%`; the Hamiltonian-variational ansatz reached
  `0.69%` with `3` layers and `6` parameters. Scope is statevector simulation,
  the registered optimizer, and model Hamiltonians only. See
  [`docs/RESULTS.md` verdict: Variational ground-state engine, rung 1: the referee is a theorem](RESULTS.md#variational-ground-state-engine-rung-1-the-referee-is-a-theorem-2026-07-12).

- **`Ansatz-structure search, rung 2: greedy loses to hand design`** is the
  exact ledger entry supporting the follow-on negative. At criticality the
  greedy search returned `1.518%` relative error against the hand-designed
  arm's `0.694%`; both figures carry the booked `n=1` inner-optimization
  fence. A second search method also failed its quantitative bar.
  The ledger retains the qualitative phase-reading observation but closes the
  engineering search after those registered failures. See
  [`docs/RESULTS.md` verdict: Ansatz-structure search, rung 2: greedy loses to hand design](RESULTS.md#ansatz-structure-search-rung-2-greedy-loses-to-hand-design-2026-07-12).

## Differential-equation composition

- **`ODE engine, rung 1: an engine made of engines`** is the exact ledger
  entry supporting the retained composition result. The registered families
  scored `75/75` for both the house engine and `sympy.dsolve`. The ledger books
  no device/runtime environment for its timing readout, so that comparison is
  not republished here. This is solve parity on that generated family, not
  superiority over differential-equation solvers in general. See
  [`docs/RESULTS.md` verdict: ODE engine, rung 1: an engine made of engines](RESULTS.md#ode-engine-rung-1-an-engine-made-of-engines-2026-07-12).

## Routed rather than retained

- The derivation-engine chronology, closed-system model births, crystal laws,
  precision studies, diet laws, and packed-crystal program duplicate the
  maintained catalog and now route to [`FINDINGS.md`](FINDINGS.md).
- Material outside the mathematics and physics presentation charter is not
  republished here.
- Old numerical prose without an exact named verdict in [`RESULTS.md`](RESULTS.md)
  is omitted. Its former presence in the README is not treated as evidence.
