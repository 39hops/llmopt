# Spec 2026-08-06: the IDENTITY battery (crest program, post-GT-7)

Status: DESIGN — pre-reg per rung before fire; 30B arms on Artin GO.
Provenance: Grok slate (RIFF 2026-08-06 evening), Artin adoption,
Fable verification. Binding context: VERDICT MOE-GT-7's survival map
— both aggregate lenses (recall, coverage) are DEAD at fixed recall;
no further aggregate-lens ladder may be proposed. The registered
variable of every rung here is EXPERT IDENTITY.

## Shared spine (all rungs)

- Vehicle: Qwen3-30B-A3B-4bit, mathgen 120-gate seed 1234, frozen
  moe_gt1_arm2 KEEPSET path (or lab equivalents once gate module
  lands), one-resident rule, oracle v3.2 line-server.
- Identity handle: keep-sets differing in NAMED experts only —
  recall pinned to the 0.72 band AND verbal coverage pinned to a
  fixed band, so no aggregate moves when identity does (GT-7's
  two-constraint draw machinery, scratch/gt7_draw.py, is the
  instrument; extend with swap/ablate ops).
- Fences: [FORMAT-BOUND] [REGIME-SCOPED: measured deployment
  artifacts]; one gate seed unless a delta needs the sigma fence;
  draw spread is a finding, never averaged away.

## Rungs (pre-reg each; order is the dependency order)

1. EX-ANAT-1 (identity swap): within a matched (recall, coverage)
   bin, swap k SPECIFIC experts between a high-scoring draw and a
   low-scoring draw of the same bin (e.g. c15's 53-v-15 pair, c30's
   54-v-5 pair — the battery's own receipts). Registered question:
   does capability FOLLOW the swapped identity set? Bars on solve
   delta direction across >= 2 bin pairs; a bisection ladder (k
   halves) only if the first swap moves >= 14 pooled.
   FROZEN DESIGN (Grok pressure points + Opus desk grounding,
   Fable spot-verified — c30 k=4 cell reproduced exactly; adopted
   2026-08-06 evening):
   a. SWAP IS SYMMETRIC AND CLASS-PRESERVING — per layer, the
      set-difference partitions into verbal-only and non-verbal-
      fill classes; verbal exclusives exchange ONLY with verbal
      exclusives, fill with fill. Coverage is then STRUCTURALLY
      INVARIANT (verbal-for-verbal preserves |keep & vonly|
      exactly; fill touches no vonly element) — measured cov
      0.3016/0.3016 at every k on c30. One-way transplant and
      non-class-preserving forms are REJECTED (measured: they
      move coverage, and global-K trips the recall fence — the
      baselines sit ~0.0005 from the window edge).
   b. K-SELECTION RULE (frozen): within each class rank exclusives
      by arm0 demand count (ties by ascending expert id); exchange
      top q = min(|A_class|, |B_class|, k) per layer, same k both
      classes. k = 4 (384/382/384 experts, ~40% of the diff, ~8/68
      per layer; recall margins +0.0020..+0.0054 across all three
      bins). k <= 8 stays in band everywhere; k = 16 degenerates
      to the full diff (that is the other arm, not a swap) and is
      FORBIDDEN.
   c. POST-SWAP VALIDITY: the driver RECOMPUTES recall and
      coverage on both swapped sets and ABORTS loudly if either
      leaves the GT-7 tolerance (recall +-0.01, coverage +-0.03).
      NO repair path — grounded finding: a re-draw of non-swapped
      slots would break the "same underlying draw, identity
      swapped" logic the rung depends on. Desk numbers say the
      assertion never fires at k <= 8.
   d. GROUNDED CONTEXT the bars should anticipate: the exclusive
      sets are LARGE and demand-THIN (~1000 experts per pair,
      ~9-10% of demand, top-10 exclusives carry only ~11-14% of
      exclusive mass) — if capability follows a FEW experts, k=4
      swap may miss them; the bisection ladder direction is
      therefore DOWN from a moving k=8 arm, not up from k=1.
2. EX-ANAT-2 (excluded-expert anatomy): for the crest mask's
   excluded population, measure what they compute — routing
   itinerary when forced resident, activation profile on gate
   problems, error-class shift when a named subset returns.
   Descriptive first tier feeding a registered second tier; this is
   the banked "what do the excluded experts compute" spearhead
   (D4-PHYS-B) made mandatory by GT-7.
3. R-EMISSION: separate text emission from math competence on the
   resurrection arms — degeneracy census (now collected per arm) +
   parseable-expression rate vs solve rate as the registered pair.
   Question: does the verbal population restore EMISSION (parseable
   output) or COMPETENCE (correct output)?
   FROZEN DEFINITION (Grok pressure point 4): "emission restored"
   = parseable-expression rate AND non-degenerate output (distinct-
   answer count in the healthy band), jointly — parseable alone is
   insufficient because the GT-1 text/coherence dissociation is
   already booked (gate capability and text coherence separate
   under subsetting). The healthy band is set from the GT-7 census
   (>= 46-solve arms ran 118-120 distinct of 120; sub-20 arms
   59-113) in the pre-reg, not read off the arm being judged.
4. CHURN-JUDGE-2: gated TWICE — needs (a) a named identity handle
   from rungs 1-2 and (b) the booked CHURN-JUDGE-1 revive-if
   (routing-margin features, new pre-reg only). Not before.

## Pairing note

lab/gate (instrument battery) can land before rung 1 and adopt the
arm2 gate path with the usual source-identity + acceptance guards —
the same pairing fence as the traj session: instruments accept
against FROZEN artifacts only; science reads through registered
lenses only.
