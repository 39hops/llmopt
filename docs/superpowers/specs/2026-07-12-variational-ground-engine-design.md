# Variational Ground-State Engine (physics rung) — Design

Approved by Artin 2026-07-12 ("we follow the field's benchmark
culture and separate the tail risk — let's do it as a rung").

## Charter compliance (methods not molecules)

Targets are MODEL HAMILTONIANS ONLY (transverse-field Ising,
Heisenberg chains) — abstract lattice physics, not molecules — plus,
if ever extended, the published quantum-chemistry benchmark canon
(H2/LiH class). Bright lines: no novel molecule proposals, no
bioactivity/toxicity-linked properties, no synthesis anything.
Accuracy is a property of the METHOD; harm is a property of the
TARGET; the benchmark canon proves the separation costs nothing.

## The physics (why this fits the house)

Ground-state search is propose/verify with the referee built into
physics: the variational principle guarantees <psi|H|psi> >= E0 for
ANY state psi — you cannot score below truth, so every improvement
is sound. At small n (<= ~14 qubits) exact diagonalization gives a
PERFECT oracle (the true E0) to measure error against. This is the
repo's structure transplanted: propose (ansatz state) -> verify
(energy) -> lower is better; magic/entanglement structure decides
hardness (product states fail exactly where entanglement matters —
at criticality).

## Rung 1 (this spec): infrastructure + variational baseline

- `llmopt/quantum/ground.py`: TFIM Hamiltonian builder (n qubits,
  field h; dense numpy for n <= 12), exact ground energy (eigh),
  a minimal statevector simulator (RY rotations + CZ-ring
  entangling layers — hardware-efficient ansatz), energy
  expectation, and parameter-shift gradients (exact for RY).
- `scripts/bench_vge.py`: race three arms on TFIM n=10 across
  h in {0.5, 1.0 (critical), 2.0}: product state (mean-field
  baseline — layers=0), shallow ansatz (2 layers), deeper (4
  layers). Adam on parameter-shift gradients.
- Tests: Hamiltonian correctness vs known TFIM limits (h=0: E0 =
  -(n-1) classical Ising chain... exact values by eigh at tiny n),
  simulator unitarity, parameter-shift vs finite difference.

**Pre-registered bar**: deeper ansatz reaches relative error < 1%
of exact E0 at the critical point h=1.0 (where mean-field is worst)
AND strictly beats the product-state baseline at every h. Honest
losses recorded per house rules.

## Rungs 2+ (banked, not this spec)

Ansatz STRUCTURE search (the engine move: propose circuit layouts,
energy oracle judges — config-estimator lessons apply: only if the
structure space has variance), Heisenberg/Hubbard targets, and the
step-chain version (LLM proposes ansatz edits, oracle-gated).
