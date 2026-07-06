# HCE rung 1: primitive differentiation move set — design

Date: 2026-07-06. Status: approved design, pre-implementation.
Parent: roadmap item #1 (`2026-07-06-next-sessions-roadmap.md`),
`llmopt/search/derivation.py` rung-0 chassis and its KNOWN LIMITATION note.

## Problem

Rung 0's `doit_one` delegates to sympy's complete solver, so beam search
solves everything in ~1 ply — the search is degenerate. Rung 1 replaces
that one omniscient move with primitive calculus rules so the *search*
composes the reasoning and sympy only verifies steps.

## Decisions (with rationale)

1. **Domain: differentiation only.** The diff rule set is small, complete,
   and terminating — the right substrate for calibrating the HCE. Integration
   (u-sub, by-parts — genuinely branching, can fail) is rung 2.
2. **Move shape: (rule, node) pairs.** A move is "apply rule R at
   Derivative-node N", applied via `expr.xreplace({node: rule(node)})`.
   This gives an honest branching factor and is exactly the shape a
   model-proposer later ranks. Whole-expression rules (rung-0 style) were
   rejected: two applicable sites collapse into one successor, quietly
   reintroducing degeneracy.
3. **Edge oracle: verify every edge with sympy.** After a rule fires,
   check parent ≡ child (`doit()` both sides, difference simplifies to 0).
   Sympy is demoted from mover to verifier; the roadmap's "illegal moves
   are impossible" property is preserved even though rules are now
   hand-written. Test-only verification was rejected (a rule bug on an
   untested shape would silently corrupt step chains).
4. **Algebra moves: trimmed subset kept.** `expand, factor, cancel,
   together, trigsimp, powsimp` remain as cleanup moves. `simplify` is
   removed from MOVES (an algebra mega-move that collapses plies and
   blurs step structure), and `doit_one` is removed entirely (doit stays
   available to the verifier and tests as ground truth).
5. **Minimal core, macros as measured ablation.** Rules stay minimal;
   preference among derivation paths is the evaluator's and (later) the
   proposer's job — the chess division of labor (move generator minimal,
   policy/eval rank lines; engines never add compound moves to the rules).
   Redundant macro rules reduce depth but increase branching, and
   branching competes for beam slots at every ply while depth is paid
   only on the path. Duplicate paths also burn node budget and per-edge
   verification cost before the visited-set dedup catches them.
6. **One eval function per search.** HCE v0 is the only evaluator. The
   future "NNUE moment" is a *swap* behind the same `hce(state)`
   signature, compared by A/B solve rate — never blended with HCE and
   never scoring states via speculative sub-searches.

## Core move set (6 rules)

`DiffRule = Callable[[sp.Derivative], sp.Expr | None]` — returns `None`
when it doesn't match (applicability test and rewrite in one function).
All rules take `Derivative(f, x)`: single variable, first order.

| Rule | Fires when | Rewrites to |
|---|---|---|
| `d_const` | `f` has no `x` | `0` |
| `d_x` | `f == x` | `1` |
| `d_sum` | `f` is `Add` | Add of **unevaluated** `Derivative` of each term |
| `d_product` | `f` is `Mul` | `f'·g + f·g'` with unevaluated inner Derivatives; an n-factor Mul branches over (head, rest) splits |
| `d_power` | `f == u**n`, `n` free of `x` | `n·u**(n-1)·Derivative(u, x)` |
| `d_chain_table` | `f == h(u)`, `h` in table (sin, cos, tan, exp, log, sqrt, …) | `h'(u)·Derivative(u, x)` |

Chain rule is not a standalone move — it is fused into `d_power` and
`d_chain_table` (an explicit-u chain move is u-sub territory, rung 2).
Every rule except the two base cases emits smaller unevaluated
`Derivative` nodes, so derivations are genuine multi-ply descents and
`max_plies` / beam width become meaningful.

There is no quotient rule in the core: sympy has no quotient node
(`u/v` is `u * v**(-1)`), so `d_product` + `d_power` already cover it.

## Macro rules (off by default, ablation-gated)

A separate `MACRO_RULES` list, starting with `d_quotient` (the textbook
quotient rule as one move). The bench script gains a `--macros` flag;
the ablation reports solve rate vs node budget and mean derivation
length with and without. Macros earn a slot only if they win on
solve-rate-per-node.

## Search / HCE changes

- `successors()` enumerates all firing (rule, Derivative-node) pairs
  plus whole-expression algebra cleanup moves; each edge sympy-verified.
- `beam_search` unchanged. HCE v0 unchanged initially — the
  unsolved-count term (weight 100) now decreases one rule application at
  a time instead of jumping to 0. Weight revision belongs to the
  calibration harness in the bench script, not this design.
- `State.history` entries should identify rule *and* target node (e.g.
  `"d_power@Derivative(x**2, x)"`) so step chains are legible training
  data.

## Testing

- **Per-rule property tests:** generate expressions (reusing `mathgen`
  generators), apply each rule at every firing site, assert
  sympy-equivalence of parent and child.
- **Non-degeneracy regression:** beam search on a mathgen
  differentiation set solves with `plies > 1` — the rung-0 failure mode
  asserted against explicitly.
- **End-to-end:** solved states match `sp.diff` ground truth (symbolic
  equivalence, never string match, per repo convention), and
  `SearchResult.state.history` reads as a coherent step chain.

## Out of scope (rung 2+)

Integration rules (u-sub, by-parts), explicit-u chain move, learned
eval (NNUE swap), model-as-proposer move ordering, HCE weight tuning.
