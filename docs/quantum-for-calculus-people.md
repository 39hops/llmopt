# Quantum circuits math, for someone who thinks in calculus

The repo borrowed a pile of quantum-information ideas (magic, syndromes,
stabilizers, ZX). Here's what they actually are, mapped to things you
already know. No physics required — it's all linear algebra wearing
a costume.

## 1. A qubit is just a 2-vector

A qubit's state is a vector `[a, b]` with `|a|² + |b|² = 1` (complex
a, b). That's it. `[1,0]` is "0", `[0,1]` is "1", anything else is a
weighted mix ("superposition" = linear combination — the same way
`3sin(x) + 4cos(x)` is a mix of two basis functions). n qubits = a
vector with 2ⁿ entries. The exponential blowup of that vector is the
entire reason quantum computers might matter.

## 2. Gates are matrices; circuits are matrix products

A "gate" is a 2×2 (or 4×4...) unitary matrix — unitary meaning it
preserves lengths, i.e. it's a rotation in the state space. A
"circuit" is a chain of them: `U₃U₂U₁·v`. Reading a circuit diagram
left-to-right is composing matrices right-to-left. Calculus analogy:
gates are to states what operators (d/dx, ×x, substitution) are to
functions — and a circuit is a derivation chain. This is why our
engine's rewrite chains and quantum circuits felt like the same
object to you: they ARE the same shape (sequences of composable
transformations with an algebra of simplifications).

## 3. The Clifford/magic split — where "magic" comes from

Some gates (the "Clifford" set: H, S, CNOT) generate circuits that a
CLASSICAL computer can simulate efficiently — the Gottesman-Knill
theorem. Intuition: Cliffords shuffle a finite family of nicely-
structured states around; you can track the structure instead of the
2ⁿ vector. The T gate (a 45° phase rotation) breaks that structure.
Quantum advantage lives ENTIRELY in the T gates; everything else is
bookkeeping. So:

- **Magic** = how far a state is from the cheaply-simulable
  (stabilizer) family — the resource that makes quantum quantum.
  Measured by things like "stabilizer extent"; costly states are
  distilled from many noisy copies ("magic state distillation").
- **T-count** = number of T gates in a circuit = its magic budget =
  the real cost of running it fault-tolerantly.

**Repo mapping**: our "magic estimator" is this idea transplanted —
predict how much non-trivial resource (search effort) an integral
needs before spending it. An integral solvable by table lookup is
"Clifford"; one needing deep composed rewrites is "high T-count."
The ZX engine literally minimizes T-count.

## 4. Stabilizers = symmetries that pin a state

A stabilizer of state v is a matrix S with `Sv = v` — v is a +1
eigenvector; S "agrees" with v. Instead of storing the 2ⁿ vector,
store the LIST OF CHECKS it passes (n matrices suffice). Calculus
analogy: describing F(x) not by its formula but by the equations it
satisfies — "odd function, satisfies F' = f, F(0)=0." A function is
pinned by its constraints; a stabilizer state is pinned by its
checks.

## 5. Syndromes = which checks FIRE (the bit-vector of symptoms)

Error correction: encode information redundantly, then repeatedly
measure a set of parity checks. Each check returns pass/fail; the
vector of results is the **syndrome**. You never look at the data —
you look at the pattern of failed checks, and a decoder infers the
error from the pattern (like diagnosing a disease from a symptom
checklist, never seeing the pathogen). qLDPC codes = error-correcting
codes whose checks are sparse (each check touches few bits) — cheap
to measure, hard to decode, very efficient.

**Repo mapping (the big one)**: our syndrome policy asks each rewrite
rule "do you fire on this expression?" and gets a bit-vector — a
syndrome. The policy net is a DECODER: from the pattern of which
rules fire, infer which move is right, without deeply reading the
expression. Same trick, same math, different universe. It became the
production brain.

## 6. ZX calculus = circuits as rewritable graphs

ZX is a graphical language: circuits become colored graphs (green/red
"spiders" with phases), and there's a small set of rewrite rules
(fusion, phase-teleportation, ...) that transform graphs WITHOUT
changing the linear map they compute. Simplifying a circuit =
rewriting its graph to a smaller one — EXACTLY our engine's move
structure (equivalence-preserving rewrites + a cost to minimize +
search over rule orders). That's why porting the engine's recipe to
ZX (rungs 5-7) worked: same problem class, different rule set,
same oracle discipline (the graph's semantics never change).

## 7. Why quantum computers DON'T speed up LLMs (the investor note)

LLMs = huge dense linear algebra over huge DATA. Loading N numbers
into a quantum state costs ~N operations — the exponential state
space doesn't give you exponential data throughput. Known quantum
wins: factoring (Shor — breaks RSA), quadratic search (Grover — dies
under error-correction overhead), and simulating quantum systems
(chemistry/materials — the real killer app, because there the DATA
is small and the STATE SPACE is the hard part). The repo's actual
returns came from quantum MATHEMATICS on classical hardware — magic
as a hardness measure, syndromes as features, ZX as a rewrite
domain. The math is liquid; the hardware is still escrowed.

## Cheat sheet

| Quantum thing | It's just... | Repo incarnation |
|---|---|---|
| qubit state | unit vector | — |
| gate / circuit | rotation matrix / product of them | rewrite step / derivation chain |
| Clifford vs T | cheap structure vs expensive resource | table-lookup solves vs deep search |
| magic | distance from cheaply-simulable | magic estimator (hardness) |
| stabilizer | check that pins a state | constraints pinning a function |
| syndrome | bit-vector of failed checks | rule-fire bits -> policy/dispatcher |
| qLDPC | sparse checks, clever decoder | sparse rule-fires, NNUE decoder |
| ZX | graph rewriting for circuits | the ZX engine (rungs 5-7) |
| T-count | magic budget of a circuit | ZX engine's minimization target |
