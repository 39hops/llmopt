# T-count engine: the derivation engine pointed at quantum circuits

Status: spec (banked physics-night 1, motivation completed by the
magic-detector win; Artin's technique research 2026-07-08 mapped
below). Domain library verified on Mac: pyzx 0.10.4 — random 5q/80
circuit, T 16 -> 8 under greedy full_reduce, tensor-verified equal.

## Why this domain, one paragraph

Post-error-correction, T gates cost ~100x Clifford gates, so
"minimize magic" is the quantum industry's optimization metric —
and the problem is EXACTLY our architecture: states = ZX diagrams,
moves = local rewrite rules, oracle = diagram/tensor equivalence
(GF(2)-cheap for the Clifford part), eval = T-count. The verifier is
cheaper than sympy; the search is the same search. Tonight's magic
detector was math borrowing quantum's concept; this engine is
quantum borrowing math's chassis.

## Mapping Artin's technique list

- **ZX-calculus + phase teleportation**: THE v0 substrate. pyzx has
  the rewrite rules (spider fusion, pivot, local complementation,
  gadget cancellation), phase teleportation, and verification
  (compare_tensors for small circuits; graph-equality proofs
  internally). full_reduce is the domain's greedy "doit" — the rung-0
  lesson applies: the engine must beat the greedy oracle or admit it
  wins.
- **Markov prior (Artin's ask)**: rule-bigram over ZX rewrites mined
  from winning reduction paths — the 293-scoring dict, transplanted.
  Day-one component.
- **SCDF**: upstream (Hamiltonian factorization before circuits
  exist) — related work, out of scope.
- **Unfolded distillation**: hardware/fault-tolerance layer — below
  scope.
- **Pauli-based compilation / unitary synthesis**: alternative
  compilation stacks — related work; Pauli gadget form may become a
  v2 state representation if ZX plateaus.

## v0 design (one session)

- **State**: pyzx Graph (ZX diagram). Key = canonical graph hash
  (pyzx provides normalization; else round-trip extract+qasm string).
- **Moves**: pyzx's simplify primitives applied LOCALLY, one step at
  a time (match_* + apply single matches — not the fused full_reduce
  loop): spider fusion, pivoting, local complementation, gadget
  fusion, phase teleportation step. (rule, site) pairs = real
  branching, like (rule, node).
- **Oracle**: two tiers. Small circuits (<= 8 qubits): tensor
  equality (exact, seconds). All sizes: pyzx rewrites are
  soundness-preserving by construction — the boundary check is
  extraction round-trip + tensor spot-check, mirroring
  verify_p/replay_verify.
- **Eval**: T-count (exact, free). Plus a structural featurizer
  (spider count, phase-gadget count, connectivity stats) for the
  NNUE-analog rung.
- **Benchmarks**: random CNOT_HAD_PHASE circuits (seeded, string
  seeds per house rules) + pyzx's bundled benchmark circuits (the
  literature's standard set). Score: T-count vs full_reduce baseline
  at equal wall/node budget. Honest metric: report circuits where
  greedy wins too.
- **Ladder** (each rung measured): 0 greedy baseline -> 1 primitive
  moves + best-first on T-count -> 2 markov prior from winning paths
  -> 3 NNUE-analog eval -> 4 autopsy the stuck diagrams.

## Non-goals (v0)

General quantum simulation (complexity-walled), fault-tolerance
layout, chemistry compilation (SCDF), >12-qubit exact tensor
verification (use round-trip tier).

## Test plan

Move legality: each primitive move preserves tensor semantics on
random small diagrams (the property test). Engine: rung-1 beats or
ties full_reduce on >= 20% of seeded circuits at budget X (pre-
registered; if null, the greedy oracle wins the domain and we say
so). Determinism: string seeds. Contamination: benchmark circuits
never in any training harvest.
