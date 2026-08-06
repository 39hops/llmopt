# Handoff 2026-08-06-2 — the identity day (traj close -> GT-7 null -> LOCKSTEP -> W1-S -> EX-ANAT-1 fire)

One session, seven booked verdicts/pre-regs. Read order for resume:
BOARD header (08-06 EVENING) -> this file -> RESULTS tail from
VERDICT LAB-TRAJ down.

## What happened, in order

1. **lab/traj module 4 CLOSED** (VERDICT LAB-TRAJ). Desk divergence/
   authority table FIRST (10 surfaces), then `patch_moe_router`
   strictly against it; synthetic parity vs the frozen instruments;
   live 3b/3a byte-identical; D0 590,736 rows — 5,808 differ ONLY in
   phase, provenance-dated to the 2c47630 amendment the artifact
   predates. NEW FENCE: keepsets DROP_TAIL is v2-artifact-BOUND
   (46,080 picks over-dropped on v3 files; fix waits on the
   dual-copy doctrine). Remaining lab modules: gate -> sink ->
   timebox.
2. **GT-7 fired on Artin GO, P-NULL FIRED** (VERDICT MOE-GT-7).
   Coverage joins recall as a dead aggregate lens. Within-bin draws
   matched on BOTH constraints differ 38-49 solves. Both GT-6
   anomalies dissolved into draw identity. Survival map BINDING.
3. **RULER/gist killed, LOCKSTEP opened** (spec
   2026-08-06-3080-lockstep-window.md), and its first two rungs
   PASSED same-day (VERDICT LOCKSTEP-A1/A2): MB-S14 + DIET-BRIDGE
   bit-identical on the 3080 — now DEVICE-FREE claims.
4. **TENET-W1-S: four-surface null** — TENET-W1 is now a
   SURFACE-EXHAUSTIVE negative at toy scale; no W2. Operational
   rule booked: wsl.sh launch log paths must exist BEFORE launch
   (the first launch died silently on its own redirect).
5. **EX-ANAT-1 FIRED AT 6.3x THE BAR** (VERDICT EX-ANAT-1):
   symmetric class-preserving k=4 demand-ranked swap between GT-7's
   matched-bin draws. Pooled D=176 (bar 28), 2/3 directional. Every
   low draw jumped to/above its bin's original high; c30 hi' hit 77
   > full model 64 — booked as its own anomaly. Capability FOLLOWS
   a named 384-expert set.

## Instruments/specs born today

- llmopt/lab/traj.py + tests/test_lab_traj.py (module 4).
- scratch/gt7_draw.py (two-constraint draws), gt7_run.py (multi-arm
  gate driver, ARMS env — reused verbatim by EX-ANAT-1),
  ex1_swap.py (class-preserving swap builder, abort-not-repair).
- scratch/tenet_w1_surfaces.py (surface ladder, frozen-bridge
  protocol).
- scratch/traj_accept.py (frozen-driver instrument-swap pattern).
- Specs: identity-battery (grounded, frozen design a-d),
  3080-lockstep-window (incl. Mac-precision C2 addendum).

## Open GOs / holds

- EX-ANAT-1 bisection (k=2, then 1): new pre-reg, Artin GO.
- EX-ANAT-2 + R-EMISSION: spec'd, pre-reg before fire.
- CHURN-JUDGE-2: identity handle exists; routing-margin gate still
  binds.
- Lean full pass (21,914 chunked, WSL CPU): overnight GO pending.
- PLATEAU-BREAK + deterministic gravmoe pair: design pass first.
- lab/gate module next in the extraction order.

## Cautions for the next session

- The GT-7/EX-ANAT keep-set jsons in checkpoints/ are now CITED
  evidence (gt7_ladder_*, ex1_*) — regenerable from string seeds
  but do not overwrite (gt7_draw/ex1_swap have no overwrite guard;
  add one before any re-run).
- DROP_TAIL fence: any keepsets read of a v3-tagged traj file needs
  DROP_TAIL=0.
- The lens authority for anything coverage-flavored remains
  scratch/gt7_coverage_rederive.py AS IT COMPUTES (incl. its
  DROP_TAIL=1-on-v3 choice — frozen means frozen).
- c30 hi' = 77 > full 64: single arm, one seed — do NOT lean on it
  before a replication rung; it is bait for a "beats-full v2"
  claim the resolution law does not yet license.
