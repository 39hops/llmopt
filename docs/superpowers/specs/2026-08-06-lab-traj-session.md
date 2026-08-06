# Spec 2026-08-06: lab/traj — the careful session (module 4)

Opener of its OWN session (agreed Fable + Grok review, 2026-08-06):
traj fails the verbatim-adoption test that made oracle/config/keepsets
safe — it UNIFIES three divergent copies, which is design, not copy.

## The three copies to unify

1. scratch/moe_gt1.py `instrument` — free-routing TRAJ recorder
2. scratch/moe_gt1_arm2.py `instrument` — masked variant (+ closed-
   loop recall counters)
3. scripts/moe_router_stats.py — the third copy

Target: one `patch_moe_router(model, traj=..., keep=None)` in
llmopt/lab/traj.py; restore() becomes a context manager (a raising
gate must not leave the class patched — emit INSTRUMENT_NOT_RESTORED,
per the loud-failure contract).

## Certified-artifact-sensitive surfaces (the pin-diff list)

- pooled-pos vs per-prompt tpos split
- prompt_tail rule (TRAJ v3 phase tagging)
- precise-softmax flag
- H/scores rounding at write time
- row schema field names + order (verdicts cite them)

## Acceptance (in order; needs the 30B RESIDENT — one-resident rule)

1. Desk tier: source-vs-source diff of the three copies, divergences
   ENUMERATED in the module docstring before any unification.
2. Regression tier: D0-style bit-identity — re-run the certified TRAJ
   regeneration (the 4f3dc6c instrument's regression: 590,736 rows
   bit-identical) through the unified patch.
3. Live tier (SPLIT resolved 2026-08-06, Artin nod — arm2 writes no
   TRAJ rows, so the two claims separate):
   3a. masked arm: one small gate run, unified patch vs frozen
       moe_gt1_arm2 path — recall counters equal + per-problem rows
       byte-identical.
   3b. free arm: fresh free-routing TRAJ rows, unified patch vs the
       frozen moe_gt1 instrument — rows byte-identical (on top of
       tier 2's D0 regression).
   traj+keep TOGETHER is REFUSED in v1 (loud ValueError; no source
   copy ever ran the combination, no certified artifact constrains
   it — Artin nod 2026-08-06); revisit only under a registered run.
4. keepsets closes the loop for free: the always-on full acceptance
   (booked GT2 stats + dump bytes) re-runs over any regenerated rows.

## Session pairing with GT-7 (optional, Artin's call)

Load the 30B once; run the traj acceptance arm FIRST (certified path,
pin-diff attention fresh), then GT-7 under its own HOLD->GO pre-reg —
or reverse if science is the priority. NAMED CONTRACT fence: neither
arm's row format may "help" the other; traj acceptance compares
against FROZEN artifacts only, GT-7 reads through its registered lens
(gt7_coverage_rederive.py) only.

## After traj

gate (can reuse the adoption patterns from tonight: source-identity
+ synthetic battery + booked-number acceptance), then sink (extend
llmopt.runlog, terminal-record contract), then timebox (fork
run_isolated; the 47 SIGALRM scripts migrate ONLY on re-run).
