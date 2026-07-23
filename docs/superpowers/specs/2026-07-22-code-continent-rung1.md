# Code continent, rung 1 + universality point 3 (2026-07-22, Artin GO)

Purpose: (a) third closed system for the universality result
(geometry = f(feeding), grammar-independent — needs a maximally
DIFFERENT grammar); (b) open the banked code continent per its own
scoping (ASM-level rewrites; the model rewrites programs, the
machine emulates — never the model).

## The system: vm-asm (mini-ISA, exact symbolic oracle)

Not C (the old ladder's carrier — too far from closed): straight-
line integer assembly over 4 registers.

- **Terms**: programs = sequences of `mov/add/sub/mul/shl/neg`
  over r0-r3 + integer immediates. Straight-line only (no
  branches) — every program computes a POLYNOMIAL map of the
  initial register state.
- **Rules** (the rewrite table, one rule per row — the ladder's
  own law + tonight's one-primitive law applied from birth):
  constant folding, strength reduction (mul->shl, mul 1 elim),
  dead-store elimination, mov-chain collapse, add-0/sub-0 elim,
  neg-neg cancel.
- **Oracle**: EXACT symbolic equivalence — run both programs on
  symbolic registers (sympy), compare the resulting polynomial
  maps. No test vectors, no sampling; equivalence is decidable
  because straight-line programs are polynomial maps. (Real
  toolchain — clang/llvm-mc, the ladder's scorer — stays the
  port target for rung 2+, Windows box.)
- **Chains**: engine (greedy rule application, ~50 lines) reduces
  random generated programs to normal form; rows are cur->nxt
  single-rule rewrites. Determinability by construction: each nxt
  is one rule applied to cur. Exclude-guarded splits, stable
  string seeds, streaming emission — all standing doctrine.

## Vocab

New tokenizer class (like vocab-41, appended atoms don't disturb
anything): `mov `, `add `, `sub `, `mul `, `shl `, `neg `, `r0`-
`r3`, `, `, newline, digits, `-`. ~20 atoms; model class
"vocab-code". Same d384/8L/1536/6 skeleton, BIRTH_SEED=1.

## Runs (in order, all Mac-cheap)

1. **Universality leg**: 20,253-row diet (size-matched to the
   phys/math controls), 3ep, panel (kurt/nnCV/norm, same code).
   Pre-registered: panel matches ~1.9/0.023/0.59 => universality
   at three grammars (math/physics/programs). Shift => grammar
   encodes after all, at least for non-algebraic systems (also a
   result).
2. **Rung-1 probe**: held-out seeds, per-rule-kind accuracy
   (the ladder predicts: single-rule rewrites = learned mapping =
   trains up; anything requiring arithmetic beyond one fold
   lags unless tree-decomposed).
3. Later (not tonight): full-corpus birth, gate battery, real
   toolchain oracle port, the optimization game (shortest
   equivalent program = the search objective).

## Safety note

Straight-line integer arithmetic rewrites; no syscalls, no memory,
no I/O, no branches — the system cannot express anything but
polynomial register maps. TOS-clean by construction.
