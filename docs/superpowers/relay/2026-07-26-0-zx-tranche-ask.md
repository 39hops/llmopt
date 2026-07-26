# Relay -> axiom Fable (2026-07-26): the ZX row-factory tranche

## Context (one paragraph)

ZX-calculus is promoted to next-continent candidate #1 (first
GRAPH grammar — it decides whether the federation floor scales
with grammar count or class). Both desk gates closed llmopt-side
2026-07-26: serialization = boundary-anchored (BFS from the
ordered I/O vertices, internal vertex labels RANDOMIZED per
emitted sample — permutation augmentation over the true gauge;
canonical sorts are forbidden in anything model-facing, dedup
hashing exempt), and the atom set fits vocab (~10-14 new atoms;
phases are exact k*pi/4 = integers mod 8 — no floats anywhere).
Full spec: docs/superpowers/specs/2026-07-26-complex-zx-program.md.

## The ask (scoped, in order)

1. **ZX rewrite core in C++**: graph representation (spiders
   Z/X with phase in Z/8, hadamard/plain edges, ordered boundary),
   the four local moves as check/apply pairs — fuse (same-color
   adjacent spiders, phases add mod 8), identity removal
   (phase-0 degree-2), local complementation, pivot. Soundness by
   construction per move (the pyzx pattern); no extraction, no
   tensor anything — that stays llmopt-side.
2. **Serializer** per the boundary-anchor design above; emit both
   cur and nxt through it with a per-row RNG label permutation
   (string-seeded, like everything).
3. **Chain emitter**: random Clifford+T circuit -> diagram ->
   greedy/random descent by the four moves, one row per applied
   move (cur -> nxt), farm_v22-style schema with kind = move name,
   level = initial T-count bucket. Target: a 10k-row sample batch
   first (adjudication before scale, as always).
4. RIDER (cheap, if trivial): T-count and spider-count as emitted
   row metadata — the rarity/mass instruments read them later.

## Adjudication protocol (llmopt side, committed to)

Every sample row replayed against pyzx: parse our serialization
back to a pyzx graph, apply the named move at the named site,
byte-compare re-serialized nxt AND verify semantic equality on
small instances (compare_tensors at q<=6). Same dual-oracle gate
that qualified diff/equiv/ODE/series/energy — zero NOT_EQUIVALENT
tolerance, UNDECIDED taxed. Your Phase-B byte-exact discipline
applies verbatim: the serializer is the contract.

## Explicitly out of scope

Circuit extraction (gflow), T-count optimization as a goal,
anything unitary-simulation-shaped. This tranche is a ROW FACTORY
for a grammar-learning diet, not a quantum tool.

## Standing asks unchanged

fuzz-the-oracle CI, magic boards r2, Fourier tranche remain queued
behind this at your discretion; knock-4 unchanged.
