# HCE calibration harness — design

Date: 2026-07-06 (overnight session). Status: approved design —
user pre-authorized autonomous execution ("go tonight"); decisions
below made with documented defaults per that authorization.
Parent: roadmap #1 ("Calibrate the HCE like chess did: does eval(state)
correlate with eventual solve rate from that state?") and the rung-1
spec (2026-07-06-hce-rung1-primitive-moves-design.md).

## Question

Does `hce(state)` predict how solvable a state is? Chess could never
measure this for its hand-crafted evals; we can. The result also
decides whether a learned eval (the "NNUE moment") has headroom.

## Decisions

1. **On-policy state sampling.** States come from real beam-search
   trees over mathgen differentiation problems, collected via a new
   optional `trace: list[State] | None = None` parameter on
   `beam_search` that appends every generated candidate. Rejected:
   random-walk sampling (off-policy, unrepresentative) and re-enumerating
   trees inside the script (duplicates engine logic, drifts).
2. **Cost, not just success.** Rung-1 solves ~100% of mathgen L1-3
   under a generous budget, so binary labels have no variance. From
   each sampled state we probe with a fresh `beam_search` and record:
   (a) nodes-to-solve under a generous budget (width=8, max_plies=20),
   (b) solved-within-small-budget (`max_nodes=40`) as the binary view.
3. **Metrics.** Spearman rank correlation between `hce(state)` and
   nodes-to-solve (rank correlation, because HCE only needs to *order*
   states for beam pruning; magnitude is irrelevant). Implemented
   inline (~15 lines) — no scipy dependency. Plus a per-HCE-bin table:
   bin range, n, solve rate at the small budget, mean nodes-to-solve.
4. **Honest reporting.** A weak ρ is a finding, not a failure — it
   motivates the NNUE rung. No HCE weight tuning in this work
   regardless of the result (out of scope per the rung-1 spec).

## Deliverables

- `llmopt/search/derivation.py`: `trace` parameter on `beam_search`
  (append-only hook; no behavior change when None). Test: trace
  non-empty after a search, and every traced state's `.expr.doit()`
  equals the root's.
- `scripts/calibrate_hce.py`: sample up to N unique states (dedup by
  `State.key()`, string seeds per repo convention) from searches over
  mathgen levels 1-3; probe each; print the bin table and Spearman ρ.
  Skip states that are already solved (their probe is trivial).
- Measured numbers recorded in the commit message, per repo convention.

## Out of scope

HCE weight changes, learned eval, plotting/matplotlib, integration
states (rung 2).
