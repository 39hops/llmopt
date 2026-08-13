# Handoff 2026-08-12-4 — Phases 6-7 executed, CI green end to end, figure overhaul spec'd + planned

Seat: Fable 5, Mac. HEAD at close: `da6ade0` (plus this handoff's
commit). Both checkouts in lockstep through `1a8371c` fixes; sync
this commit after push. 3080 idle; nothing queued. pytest at close:
842 passed / 11 skipped rc=0 (fresh run cited below). ruff exit 0.
CI: ALL FOUR JOBS GREEN on `1a8371c` — first fully green run.

## What landed (engineering only — nothing to book)

### Code-quality program Phases 6-7 (plan `7a8e58c`, executed inline)

- `cc400d1` 6.1: tests/test_zx_engine.py — tcount pin, best_first
  reduces + boundary-oracle verifies, moves() children tensor-equal.
- `d5a78a7` 6.2/6.3/6.5: conftest `artifact_or_skip` — LLMOPT_FULL=1
  turns missing-artifact skips into FAILURES (verified 10/10 on the
  Mac with AXIOM_CHECKOUT); absolute axiom path dropped;
  figures.json message fixed.
- `7b1519c` 6.4: scratch/seed_audit.py — 281 RNG hits, ZERO tuple
  seeds, 4 UNSEEDED all false positives (multi-line chained
  manual_seed / samplers.py optional). No defects.
- `35701f1` 6.6: per-subpackage coverage table in CI, no threshold.
- `a8bd9a9` 7a: llmopt/common/repo.repo_root() — figures.json/fonts
  stay repo-level (no dual-copy); reproduce/figstyle/figsvg resolve
  through it and fail with the honest message in a wheel. CI wheel
  job (clean-venv install, asserts the failure message) + core-deps
  job. Wheel job passed FIRST TRY.
- `aa0a224` 7c: API stability tiers in the package docstring;
  CONTRIBUTING.md; runs/runlog.py -> runs/receipts.py with alias
  shims both at runs/runlog and lab/runlog (rename retargeted to
  runs/ because Phase 5 landed after the spec was written).
- `c698fab` Phase 7 harness: scratch/lib/driver.sh (pipefail strict
  mode, llmopt_cd, cuda_preamble, mark_done-on-success-only,
  wait_for with launcher-must-not-match); smoked incl. false|true
  aborts. /rung scaffold step points at it. Frozen drivers untouched.
- `e412343` README drift: lab/ paragraph corrected; runs/, figures/,
  common/ entries added. 7b needed no work — gen_readme.py + CI
  --check pre-existed and counts were in sync (217).

### CI was RED and nobody knew (fixed, now green)

Runs had been failing since before this session; local suites green
(Mac has pyzx; Apple silicon bit-matched). Three fix commits:
- `5589e16` pyzx joins test_public_imports OPTIONAL; shell_graft
  compares copied rows byte-equal.
- `096c748` zero-column contribution asserted directly.
- `1a8371c` the numeric compares use torch.testing.assert_close fp32
  defaults — hand-picked tolerances failed TWICE on the x86 runner
  (x @ W.T at N=16 vs N=8 re-pairs summands; diverges past 1e-6 rel).
  Lesson: measure the runner's delta or use the standard tolerance;
  never invent a bound.
- Watcher gotcha: `gh run list --commit <sha>` returns EMPTY on this
  repo silently. Working pattern: poll `gh run list --json
  headSha,...` and filter `startswith(sha)`, bounded loop.

### Figure/asset overhaul (Artin priority: images are the front door)

- `75da85c` riff banked: volatility-drag/data-quality frame (Artin),
  two-sigma split (error-term vs support variance) + curation fence.
- `c7abcc1` spec `docs/superpowers/specs/2026-08-12-figure-overhaul-design.md`
  after a Grok+GPT cross-review round: taxonomy stays, directories
  mirror it (web/hero/gallery/archive/2026-08-12), figs/ separate,
  NO seaborn, magnitude ramp = INFERNO BY EVIDENCE
  (scratch/crystal_recreate_test.py:80), hero re-render from
  checkpoints/gallery19m_s1.pt (EXISTS, verified), text budget ≤~6
  word title + one fence line, two passes (mechanical moves, then
  pixels), [H] never regenerated.
- `da6ade0` plan `docs/superpowers/plans/2026-08-12-figure-overhaul.md`,
  7 tasks. EXECUTION IS POST-COMPACT: Task 1 classification manifest
  first.

## Conditions that bite next session

1. **Ratchet cap==backlog (300/300)** — unchanged; first booking
   carries its FINDINGS bullet same-commit.
2. **gen_codemap regen-after-add** when a commit adds scratch/scripts
   files (bit twice on 08-12).
3. Figure overhaul Task 4 tunes the inferno slice endpoints, then
   FREEZES them in the Task 3 test — do not skip the freeze.
4. `bench_ladder.py` and friends have no argparse — never smoke-test
   scripts with `--help` without checking.

## Next session

Start: this handoff -> BOARD -> plan
`docs/superpowers/plans/2026-08-12-figure-overhaul.md` (spec beside
it). Execute Tasks 1-7 (both passes post-compact per Artin). Also
open: spec Phases 6-7 REMAINDER none — program complete; REPRODUCE
front-facing pass rode Phase 5-7 partially (README done; REPRODUCE
itself had no stale paths).

## Open decisions for Artin

1. `house-crystal` rename or keep (unchanged).
2. THEORY L4 paraphrase vs verbatim quote (unchanged).
3. bench_verify_fast ship bar: retire or re-pin to vendored 167 rows
   (unchanged).
4. Branch protection on main (CI-must-pass) — needs repo admin,
   flagged in the Phase 6-7 plan Task 7 Step 5; not attempted.

## Also standing

- Relay 2026-08-11-1 DRAFT/UNSENT.
- Banked/unqueued unchanged: BASIN-CENSUS-1, Q9 births ladder,
  ignition-mass cell B, GROW-DECOMP n=3, MPS decay probes,
  excluded-experts anatomy, GT-7 candidate; v4anat unbooked.
- Animations/GIFs (manim-style, 2swap-style): explicitly OUT of the
  figure spec; future spec candidate — animated SVG via figsvg
  (GitHub renders animated SVG in READMEs, zero new deps) was the
  direction discussed; phase-space portrait grid from mathgen
  mechanics as the candidate first piece.
