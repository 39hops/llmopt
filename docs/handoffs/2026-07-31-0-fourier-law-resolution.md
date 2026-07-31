# Handoff 2026-07-31-0 — the clock-placement law + the resolution law

**Resume path**: this handoff -> BOARD -> RESULTS tail (from
"VERDICT OVERNIGHT-31 (rung 0)") -> specs/2026-07-31-scaffold-program.md.

## The day's two laws

1. **CLOCK-PLACEMENT LAW** (FOURIER-2b + FOURIER-3, THEORY row added).
   Two-sided, pre-registered, both branches would have booked:
   - 2b: wide-Mod crystal learns exactly the digit-local moduli
     (4/5/10 = 1.00; 3/6/7/9/11 near chance) and builds Fourier
     clocks ONLY there: k=5 276/512 periodic neurons (v 11/512 on
     the incidental diet), k=7 0/512. Loss plateaued — hard moduli
     are a DIET wall, not an epochs wall.
   - 3: teach the digit-sum ALGORITHM for k=3/9 -> competence
     arrives (0.71/0.83 via multi-hop rollout; untaught k=7/11 stay
     dead) and the k=9 clock NEVER FORMS (0/512 at 0.83 acc).
   - Law: Fourier/rotational structure marks WHERE the computation
     runs (single-pass v chain), not whether the task is solved.
     Clock COUNT is exposure-sensitive (351->142 at same acc);
     presence/absence is the robust readout.
   - v1 amendment en route: uniform n starved the recursion's base
     case (90% five-digit rows; reduced forms 0.1%) -> fixed-point
     loops. NEW DOCTRINE CLAUSE: multi-step competence needs diet
     share per RECURSION DEPTH, not per input. Length-uniform fixed.

2. **RESOLUTION LAW** (methods, adopted in RESULTS). The 120-prompt
   gate has binomial sigma ~5; n=1 cells cannot resolve deltas < ~5
   solves. Rung-1's "tree win was capacity" verdict INVERTED on the
   cuda replication (capacity slope 48/46/47 flat; tree/channel +4/+5
   the other way) — both sub-sigma. RULE: gate-delta claims < 1.5
   sigma need n>=3 paired seeds before a direction is booked.

## Scaffold program end-state (spec rungs 0-3 all run)

- SURVIVING: (1) merge-free — Hebbian-pull MoE merges 4->1 at
  zero-or-positive delta, n=4 (three Mac seeds + one cuda birth;
  +2,-2,+1,+2); recipe = "birth as Hebbian MoE, ship as dense".
  (2) Mac gravmoe advantage +5-6 v lb at n=3 — device-scoped (cuda
  n=1 shows +1).
- RETRACTED TO NOISE: all tree/channel/capacity single-seed
  adjudications, both devices, both directions.
- Combos: treegrav CREATED the phylogeny (within-pair 0.9465 v
  across 0.1046 — designed signature achieved) but gate 45 = worst
  of family: correlation structure is cheap and worthless (weight-
  distance doctrine, generative side). chantree a_i pinned ~0 (4th
  pin). Channel tail never used by anything, anywhere.
- Determinism rider: cuda-born gravmoe gate reproduced EXACTLY on
  Mac (49/120, valid 45.1867816091954) — greedy gates device-stable
  on this checkpoint.
- IN FLIGHT at handoff: lambda-sweep 0.1/0.25/1.0 on 3080 (~3:40 PM
  EST; read as TREND under the resolution law, not per-cell).
  Promotion of the merge recipe to llmopt/ + axiom C++ ask: GATED on
  lambda sane + (optionally) a cuda lb seed-ladder for transport.

## Cross-lab

- FX-V2 PASS booked: axiom's torch-free/libm-free C++ twin
  reproduces both P3 digests in 0.16s (their d5e9d5a). Ladder
  complete: device -> lab -> RUNTIME. Their floor-div == trunc-div
  nonneg-numerator proof noted. rANS-unpack rider deliberately
  deferred (their call). Paper determinism section upgrade available.

## Ops/infra of the day

- friendly-fire #10: ${2:+VAR=$2} expands AFTER assignment parsing
  -> "command not found"; env(1) form fixed; chain markers must
  depend on arm exit codes (re-queued lambda chain uses &&).
- checkpoints/ gitignored (git add -A was timing out on 51GB/337
  files); confirmed anchors committed under checkpoints/confirmed/
  {crystal-math,scaffold-moe,grav-spacetime,calibration,determinism}
  via git add -f; scripts/ckpt_manifest.py writes MANIFEST.jsonl
  (sha/bytes + curated category/verdict/note, curation preserved;
  --all = forensics scan). Both determinism pins re-verified by it.
- llmopt/runlog.py shipped + tests: elapsed-wallclock logging
  ([+mm:ss.s]), timed() context, LLMOPT_LOG env; scripts extend or
  override freely. First use: fourier2b/3.
- BANKED (Artin): checkpoint forensics — sha-dedup both machines,
  autopsy name-twins, score-check v RESULTS; behind cleanup GO.
- data note: micromodel_gen4_sidecar.jsonl = 29,275 raw lines on
  both boxes (bit-identical); the booked "38,325" is load_rows()'s
  MERGED count. Not a discrepancy.

## Queue (explicit)

- [LIVE] lambda-sweep verdict (~3:40 PM) -> then merge-recipe
  promotion decision (llmopt/ + axiom C++ backend ask).
- [BANKED] cuda lb seed-2/3 ladder (gravmoe transport adjudication).
- [BANKED] FOURIER-4 candidates: (a) clock-formation dynamics
  (probe during training — when do clocks crystallize v acc?);
  (b) force-the-clock arm (single-pass k=9 diet: no algorithm rows,
  masses of small-n practice — does a k=9 clock form at all at d64?).
- [BANKED] checkpoint forensics + 51GB triage (Artin GO).
- [HOLD] paper (banked at prose-v1; FX-V2 upgrade available).
- Standing: consistent-corruption DK arm; mass candidates; old-null
  revival sweep; HCE/NNUE shelf v2; P/NP pincer-scaling.
