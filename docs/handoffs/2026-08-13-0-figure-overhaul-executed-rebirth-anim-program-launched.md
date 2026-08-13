# Handoff 2026-08-13-0 — figure overhaul executed; rebirth+anim program launched; bar retired

Seat: Fable 5, Mac, overnight session (Artin asleep; standing GO for
autonomy). HEAD at close: `5110b22` (plus this handoff's commit).
pytest at close: 847 passed / 11 skipped rc=0. ruff clean. 3080 idle.

## What landed

### Figure overhaul (plan 2026-08-12, ALL 7 TASKS)
- `11de1bf..93ab5d8`: taxonomy mirrored on disk (hero/gallery/
  archive/2026-08-12, all moves 100% git similarity), link guard
  tests/test_asset_links.py (live docs only),
  figstyle.continuous("magnitude", mode) inferno slices FROZEN in
  test (dark 0.12-1.0, light 0.0-0.82), hero re-rendered (5-word
  title + ramp cue + fence, ~0.45MB), plot_neurons gallery set
  RECLASSIFIED [R]->[H] (no recorded invocation = no reproducible
  pair), both crest figures restyled to figstyle light+dark pairs,
  web/ gained SVG <title> tooltips + one-line scopes (figures.json
  edits, numbers untouched) + fence wrapping. CI green end to end at
  `93ab5d8` (one red on the way: stale INDEX, fixed same night).

### VERDICT VERIFY-FAST-BAR-RETIRED (`5d34bbb`)
- Parity bench: 31.5x (600.4s -> 19.1s), 10 accept flips ALL
  old-oracle false negatives on corpus-true rows (probed directly),
  0 unsound-direction flips. Ship bar retired with cause; soundness
  stays on the vendored 167/167 replay. FINDINGS bullet same commit
  ([MECHANISM-CONFIRMED] [REGIME-SCOPED: calculus search]).

### Archive-rebirth + animation + housekeeping program (spec
`docs/superpowers/specs/2026-08-13-archive-rebirth-animation-program-design.md`,
Artin-approved shape: GitHub+LinkedIn targets, manim engine, curated
~8 rebirth, three scenes, full housekeeping pass)
- Phase A partial: llmopt/figures/export.py (readme 1600w/linkedin
  1200x627/source profiles, size-pinned test), scripts/
  render_gallery.py driver (GALLERY list = recorded invocation),
  FIRST REBIRTH shipped: qwen-vs-19m-polar (the survey's one HIGH
  pair; eyeballed against the frozen ancestor, structure matches).
- Phase B COMPLETE: manim CE 0.21.0 in .venv-anim (pip-only,
  python3.13; scout receipt in scratch/manim_feasibility_*.md),
  three data-true scenes in scripts/anim/ (crystal_rotation,
  training_morph, crest_race) fed by scripts/anim_precompute.py
  npz artifacts; docs/assets/anim/ ships GIF+MP4+poster triplets
  (GIFs 0.2-1.2MB after 720w/12fps palette encode). Defect caught
  at the eyeball gate: joint normalization flattened polar to a
  line; per-axis fix rendered and verified.
- Phase C NOT STARTED (blocked on Artin ruling, below).

### Surveys (Opus fleet, proposal-grade, committed to scratch/)
- taxonomy_survey_2026-08-13.md: 201 UNCITED = 146 keep / 55
  archive-candidate / 0 dead; 304 sibling-import couplings; 283
  entry points; HEADLINE: 181 frozen evidence files carry hardcoded
  data//checkpoints/ paths — the approved taxonomy spec's
  "update every reference" step conflicts with frozen-files
  doctrine.
- rebirth_pairs_2026-08-13.md: 1 HIGH / 4 MEDIUM / three-minds LOW
  (V3 expert tensor never on disk; ternary ep1 overwritten).
- manim_feasibility_2026-08-13.md: install plan (executed), npz
  interface (adopted), fallback assessment.

### Riffs banked
- `1d8879f`: engine+NNUE-pattern AGI frame (Artin, Grok break
  folded in; union-ranker residue).

## Open decisions for Artin (morning)

1. **Phase C ruling (BLOCKS housekeeping moves)**: 181 frozen files
   hardcode paths the taxonomy would move. Options: path-resolver
   shim / symlinks (frozen files untouched — house recommendation),
   or amend the taxonomy spec. No moves until ruled.
2. **Rebirth MEDIUM pairs**: ternary-vs-fp32 needs
   mathnative_45m_ternary_3ep.pt substitute (different claim);
   19m-zoom is actually the Qwen RL whisper; qwen-vs-19m pca is
   ad-hoc-reconstructable. GO per family, or leave archived.
3. **README animation embed**: which GIF (if any) goes in the front
   door. Candidates rendered: crystal-rotation (1.2MB) is the
   showpiece. Currently NOTHING embedded.
4. **Permissions allow-list**: classifier correctly blocked my
   self-edit of settings; add via /permissions if still wanted.
5. Standing: branch protection (rules + names given 08-13 ~01:00),
   house-crystal KEEP recommendation, THEORY L4 next work item.

## Conditions that bite next session

- Ratchet: FINDINGS backlog moved +1 with the booking (cap
  auto-follows? NO — cap==backlog held because the ratchet counts
  curated; verify headroom before next booking).
- gen_codemap/gen_index regen after ANY scratch/scripts add (bit
  twice more tonight: INDEX after crest restyle red CI; CODEMAP at
  close).
- data/anim/*.npz are untracked regenerables; anim scenes need
  precompute run first (scripts/anim_precompute.py --scene X).
- .venv-anim is python3.13 + manim 0.21.0; NO Tex scenes (dvisvgm
  missing); main suite independent of it.

## Next session

Start: this handoff -> BOARD -> open decisions above. Then: THEORY
L4 quote (Artin queued it), Phase A remaining rebirths (on GO),
Phase C (on ruling).
