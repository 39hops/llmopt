# streamwd_v2 full-run qualification tolerance (pinned pre-look)

Pinned 2026-08-16 ~20:56 EDT, BEFORE the full 256-expert v2 receipt
is read (run launched ~20:31, lands ~21:05; the 4-expert smoke was
seen — max rel delta 1.15e-3 on VQ arms, 4e-8 on scalar arms — and
this pin extrapolates from it, disclosed).

QUALIFICATION (descriptive, cross-device, never bar-bearing):
- per-arm |v1_op - v2_op| / v1_op <= 5e-3 for ALL 11 arms
  (operator_layer), and <= 1e-6 for the deterministic scalar arms
  (S1-T, S1-U4, S2).
- FAILS the qualification if any arm exceeds its bound; v2 then
  books as "disagrees at X" and does not advance toward promotion
  without diagnosis.

This is engineering qualification of the v2 prototype against the
v1 Mac receipt. It is NOT evidence for any 0S bar and NOT the
same-device instrument-promotion gate, which remains: v2 must
reproduce a v1 receipt on the SAME device before any rung uses it.

Known v2 nondeterminism sources (adopted from external review,
verified against torch docs at promotion time): CUDA index_add_ in
Lloyd accumulation and weighted bincount in histograms are
potentially nondeterministic; the promotion gate must either move
those reductions to deterministic implementations or measure v2's
same-device run-to-run spread FIRST and set the tolerance above it.
