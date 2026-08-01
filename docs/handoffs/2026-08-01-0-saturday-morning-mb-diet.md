# Handoff 2026-08-01-0 — Saturday morning: multi-block + diet bridge + engine adoption

State at write time: night31b LIVE on the 3080 (rjob job `night31b`,
launched 02:37; λ-merge phase done, lb ladder training in progress).
**3080 back to Artin by 1:00 PM EST** — if anything still runs near
then, `python scripts/rjob.py kill night31b` and book what the log
holds (gates print dicts + weights shas incrementally; nothing is
lost by a cut).

## What landed this morning (all booked + pushed)

1. **Engine + primitives adoption (axiom intbirth)**: FullBirth AND
   the primitive layer (Block/AdamW/int_gemm/rdiv) both PASS
   house-side with our own drivers (`scratch/verify_intbirth_prims.py`),
   8/8 digests + losses, ~1.7 s/birth. Receipts relay:
   `docs/superpowers/relay/2026-08-01-2-engine-primitives-receipt.md`.
2. **Multi-block reference SHIPPED** (`scratch/detbwd_mb.py` +
   `detbwd_mb_ref/`): 2-block mini-LM — emb + Body×2 (Block minus
   g3/wh) + final norm + TIED head. Emb grad = rounded head part +
   EXACT scatter-add, summed after per-part rounding (spec-pinned).
   Worst fp64 cosine 0.988 (bar 0.985 — single-block 0.9985 floor
   does NOT transport to 2× depth). Digest-identical across two
   independent house drivers. Spec relay for axiom:
   `docs/superpowers/relay/2026-08-01-3-multiblock-spec.md`.
3. **MB-S14**: the R2b "strictly falling" bar is CLOSED — SHIFT=14 +
   integer decay gives the first all-monotone integer birth (8/8
   milestones, nz steady 0.298). Honest misses booked: width ALONE
   removed the S12 blowup (granularity, not lr magnitude), and
   const-lr beats decay on final loss (6138 v 6271). Doctrine:
   SHIFT=14 default at mb scale; SCHED for stability demos, const
   for chasing loss.
4. **DIET-BRIDGE**: the integer stack trains on REAL gen-4 text
   (V=40 MathTokenizer, true next-token CE, 8 windows round-robin).
   Falls 15909 → 12518 then plateaus (~391/512 per token — far from
   crystal; 8 real windows >> one random target in difficulty).
   Deterministic (rerun sha identical); `detbwd_diet_ref/diet_init.bin`
   carries the window tokens so the untracked diet file is never a
   trajectory dependency.
5. **rjob launch-detach fix**: backgrounded `setsid …&` held the ssh
   channel for the job's life; double-fork through an exiting
   subshell fixed it (measured 0.97 s launch, pid captured).

## Relay state (Artin carries; read as files, never pasted)

- To axiom, ready: `2026-08-01-2` (receipts) + `2026-08-01-3`
  (multi-block spec; artifacts in `scratch/detbwd_mb_ref/`).
- From axiom, all processed through their `2026-08-01-3-primitives`.

## Queue (in order)

1. **night31b booking** when it lands (λ-merge pre-reg = amendment
   #6 review; R0 TF32-off lb ladder n=3 — the per-level dicts also
   serve the transport-verdict confirmation Artin queued).
2. **Deterministic gravmoe pair** — the last scale rung; NEEDS A
   DESIGN PASS first (what "gravity" means inside the integer
   battery: paired mb births + integer relaxation events, exact
   cross-device comparison legal). Don't improvise it; spec it.
3. **Plateau-break scaling** for the diet bridge on the intbirth
   engine (more windows/steps/params — the engine's 60× makes the
   capacity-vs-windows question cheap). Multi-block engine support
   is axiom's next leg once they take the spec.
4. Carried: exact-manipulation diet share for L4; diet meter; ckpt
   forensics/51GB triage [Artin GO]; FOURIER-4 force-the-clock.

## Fences to remember

- mb acceptance bar is 0.985 (booked), NOT the single-block 0.9985.
- The diet plateau is a capacity/task observation, not a bug —
  don't "fix" it without a pre-reg.
- Cross-device fp gate comparisons stay forbidden; the integer
  battery is its own instrument (pooling legal inside it).
