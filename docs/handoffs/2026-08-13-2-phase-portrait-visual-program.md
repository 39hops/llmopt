# Handoff 2026-08-13-2 — visual program executed + PHASE-PORTRAIT-1 born and booked

Seat: Fable 5 (Opus 5 mid-session per Artin's /model switches), Mac.
HEAD at close: `558755a` (plus this commit). 22 commits since
`aff9247` (handoff -1). 3080: untouched all session, free. Mac:
phase19m rjob DONE rc=0, no live runs. Supersedes -1 for session
state.

## What landed (chronological, all pushed)

- `aa5d9d2` three anim scenes redesigned per Sol 5.6 plan items 1-4
  (later superseded by the art-direction reset below); figures.json
  gains routing_crest `seed_pairs` (verified against RESULTS L18927);
  snake_case groundwork; metallicity jsonl + .no-amnesia gitignored.
- Art-direction reset (Artin/Sol brief): crystal_rotation prototype
  `756e954` -> revision `870b463` -> final `35f0b9c`; house animation
  grammar (field/actors/memory/geometry/text/receipt) banked in the
  storyboard doc.
- `3220fa6` RIFF: pendulum momentum-space frame (Artin, from 2swap's
  double-pendulum video); residue = birth-energy divergence ladder +
  fixed-basis trajectory rendering.
- `0f53ba4` RIFF: static-figure composition diagnosis + four named
  repairs (render driver, anim guards, paired-slope, palette ruling).
- Flagship pivot: measured the grpo3 12-cycle ladder (total
  displacement ~0.3% of row norm, max/median 2.0) and KILLED the
  dense-checkpoint phase-portrait concept on amplitude; expert-atlas
  concept approved instead. Animatic `add7f15^` (e295ba9); control
  pairing certified byte-identical via VERIFY_ONLY ex3_build replay.
- Sol's routing-canyon rewrite reviewed line-by-line and landed with
  three data-integrity fixes `0001793`: truncated-axis outcome frame
  (origin 170 inflated 1.29x to 3.89x visually) replaced by
  zero-anchored rails vs 360 (atlas_visuals.rail_fraction + test);
  hero path followed the WEAKEST expert (scores ascending, index 0);
  stride=2 silently dropped half the experts in half the frame. Plus:
  manimpango registration of vendored fonts in house_theme — every
  earlier scene had silently fallen back from Inter to a default face.
- `db779ef` routing-canyon language propagated to training_morph +
  crest_race; docs/assets/anim renamed to snake_case (git mv).
- `88c4646` tooling repairs 1+2 of the banked four:
  scripts/render_anim.py (recorded invocation, scene-declared poster
  times) + tests/test_anim_assets.py (npz checkpoint-sha re-hash,
  ledger-pinned atlas arms, triplet completeness — fired on first run:
  expert_atlas had no GIF).
- Three Opus reviewer briefs (research arc, instruments, today's
  thread), each verified before adoption. Adopted: crest_race receipt
  fence truncation BLOCKER + five doc drifts `3fee8eb`. Rejected
  after verification: "data/anim untracked" claim (all four npz
  tracked).
- `1821684` crystal_rotation frozen to archive (Artin ruling):
  assets -> docs/assets/archive/2026-08-13/, scene kept as record.
- PHASE-PORTRAIT-1 (Artin GO): driver `31c392f`
  (scratch/birth19m_phase.py, tee {model, opt, step} every 900
  steps), smoke-verified, ran on Mac MPS ~49 min, 18 milestones with
  Adam state (first optimizer-state artifacts in the lab).
  Precompute `fba4443` -> data/anim/phase.npz. Scene `28dd0c7`
  (phase_portrait: x=angle fixed basis, y=log per-step speed).
  BOOKED `947d4ee`: OBSERVATION PHASE-PORTRAIT-1 (RESULTS L27800) —
  speed 1.06e-4 -> 1.0e-6 monotone, collective settling, exp_avg a
  distinct velocity; FINDINGS bullet same commit; `558755a`
  code_commit set.

## Conditions that bite next session

- Curation ratchet back at cap==backlog after this booking — the
  NEXT booking needs its FINDINGS bullet same-commit again.
- Reviewer findings NOT yet acted on: BOARD LIVE-table staleness
  (test count 483 vs 869; code-quality row missing phases 6-7);
  results-index null-date class (40 rows, incl. today's
  VERIFY-FAST verdict — date-first headings break the extractor);
  EX4-UNIF still waiting on a "free Mac window" since 08-09 (the
  Mac is NOW free).
- checkpoints/phase19m/ = 4.1GB untracked milestones; the booked
  observation cites them. Logs-doctrine class; deletion is
  Artin-GO only.
- data/metallicity + .no-amnesia now gitignored (both verified
  benign earlier).

## Next session, in order

1. This handoff -> BOARD -> RESULTS tail (resume protocol).
2. Capability-vs-trajectory pre-reg ("optimal theta-dot"): gate all
   18 phase19m milestones — the cheap rung the booking fence names.
   Needs pre-reg BEFORE any gate fires.
3. Watch-it-think flagship (Artin-endorsed): record a 19M model
   solving one integral, every token's internals; build the small
   numpy->PIL->ffmpeg frame renderer as part of it.
4. Static figures 5-12 (unpaused once Artin says so) + remaining
   banked repairs (paired-slope form, palette ruling).
5. Reviewer-flagged BOARD/index hygiene (cheap, one commit).

## Open Artin decisions

1. Phase C frozen-paths ruling (181 files; shim/symlink v
   spec-amendment) — still the reorg blocker, unchanged.
2. Rebirth MEDIUM substitutions per family (unchanged from -0).
3. Static-figure queue unpause + palette ruling (recommendation
   stands: categorical=arm identity in statics, inferno=magnitude
   everywhere, one sentence in assets README).
4. phase19m milestones (4.1GB): keep on disk (feeds the
   capability-vs-trajectory rung) or delete after that rung books.
5. Divergence-ladder rung (pendulum riff residue): GO when wanted —
   twin births at epsilon-separated inits, measured divergence rate.
