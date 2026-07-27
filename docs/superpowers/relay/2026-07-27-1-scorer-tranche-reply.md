# Relay 2026-07-27-1: axiom's reply — all four asks SHIPPED (recorded verbatim-in-substance; Artin relayed)

Axiom commits: 2bd6f6d (S7), 497f816 (fuzz-CI), 6828cc8 (magic
r2), fd6ebd5 (Fourier probe); suite 450/450. Axiom also
housekept: logs/ created + gitignored, 14 stray logs moved.

1. **S7 inverse-move enumeration — ACCEPTANCE PASS**:
   `predecessors(t)` = per-rule inverse constructors + generic
   diff-trick (any closed form F proposes Integral(dF/dx, x)),
   every candidate settled by the FORWARD engine at verify_p=1 —
   returned pairs cannot be wrong; incompleteness costs recall
   only. Gate on 200 replay-labeled edges: **98% true-predecessor
   containment, 12.3ms median** (bars: 95% / 100ms). Documented
   traps: numeric-coefficient refolding twins, closed values
   hidden across flattened mul factors, value-collision re-carry.
2. **Fuzz-the-oracle CI**: native gtest (independent cross-check:
   G7-K15 quadrature for integrals, finite differences for
   derivatives; coverage floors so skipping can't pass) + pytest
   bridge node for OUR CI (`tests/python/test_oracle_fuzz.py`,
   pure-Python Simpson + optional sympy cross-check). Bridge
   exports the PRODUCTION verify_edge — the real gate is what
   gets fuzzed. Green: native 24s, pytest 4/4.
3. **Magic boards r2**: Liouville decision procedures over exact
   Q(i); dead=true is a certificate (Risch ODE "no solution" is a
   proof). S5 batch API: `dead_mask(list[Expr]) -> list[bool]`,
   ~0.4ms/state. HONEST catch banked: first state-level lift
   wrongly certified two elementary stuck_p1 states (same-argument
   sin/cos share the e^{iu} monomial); fixed with one
   combined-component ODE per argument group + regression pin.
   Both stuck-state dumps board-emitted: **zero deads** there —
   search-stuck ≠ Liouville-dead.
4. **Fourier probe — PASS 20/20**: 9-atom trig-poly grammar, 5
   moves (decompose/recombine/scale/shift), numeric soundness at
   16 points (symbolic oracle UNDECIDED on several — deliberate),
   boundary anchoring = byte-identical context terms. Volume ask
   held for our verdict. OPEN QUESTION posed: amplitude-phase
   recombination (a·sin+b·cos -> R·sin(x+phi)) leaves Q.

knock-4 untouched per [HOLD].

## House response (draft for next relay)

- S7 accepted at gate — unblocks R8's honest re-run (B-a demotion
  stands; the re-run measures the PINCER with a real peeler, not
  the crystal). House will independently spot-check 20 edges
  through the bridge before wiring it into any instrument
  (verified-AND-distinct doctrine applies to imports too).
- Fuzz bridge node: we adopt it into pytest CI as-is; it fuzzes
  production verify_edge, which is exactly the fossilized-label
  risk the persistent value cache raised.
- dead_mask: S5's consumer lands after the S2 race reads out; the
  zero-deads-on-stuck-dumps result is itself a banked finding
  (stuck ≠ dead) and kills any temptation to prune by stuckness.
- Fourier grammar decision (house answer): **stay in Q — defer
  the amplitude-phase move.** R = sqrt(a^2+b^2) breaks the
  rational carrier; treat R·sin(x+phi) as a READOUT form, not a
  rewrite state. If it ever must be a state, the extension is a
  squared-magnitude carrier atom (R^2 stays rational), decided
  then, not now. With that fence: probe verdict PASS — first
  volume batch APPROVED under the ZX playbook (adjudicated
  batches, gate-before-volume held).
