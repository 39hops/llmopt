---
name: anim
description: Ship an animation the house way - render through the driver, run the guards, eyeball the poster AND the receipt frame, commit assets with the scene. Use for "render the animation", "ship the scene", "re-render X", or after editing anything in scripts/anim/.
---

# Shipping an animation (the house ritual)

Born 2026-08-13, the day a zsh word-split silently rendered NOTHING
while reporting success, and a shipped receipt turned out to truncate
its own scope fence mid-word. Every step below exists because
skipping it shipped a defect.

## 1. Never render by hand — use the driver

```bash
.venv/bin/python scripts/render_anim.py <scene_stem>
```

The driver is the recorded invocation: 1080p60 master (dark + light
where the registry says so), poster at the SCENE-DECLARED time, GIF
derived from the MP4. New scene? Add it to `SCENES` in
`scripts/render_anim.py` first — the registry is the shipping list
and `tests/test_anim_assets.py` enforces it.

## 2. Guards, on a real exit code

```bash
.venv/bin/python -m pytest -q tests/test_anim_assets.py tests/test_atlas_visuals.py
```

These re-hash npz checkpoint citations, pin ledger numbers, and check
triplet completeness. They have fired on real gaps (missing GIF,
first run).

## 3. Eyeball gate — poster AND receipt frame

Extract and LOOK at two frames minimum:

```bash
ffmpeg -y -ss <poster_t> -i docs/assets/anim/<stem>.mp4 -frames:v 1 /tmp/poster.png
ffmpeg -y -sseof -1.0 -i docs/assets/anim/<stem>.mp4 -frames:v 1 /tmp/receipt.png
```

- Poster: clean composition, NEVER the receipt card, no text over
  live data.
- Receipt: the FULL fence, word-wrapped — a truncated fence is a
  BLOCKER, not a nit (the crest_race incident). Every scope
  (device, vehicle, seed-pairing) must be present.

## 4. Honesty checklist (the house grammar's hard rules)

- Axes zero-anchored via `atlas_visuals.rail_fraction` — never an
  origin near the data (the 3.89x-vs-1.29x incident).
- Every on-screen number traces to RESULTS/figures.json; per-set
  causal numbers appear only with the complete set on screen.
- Interpolated motion disclosed in the scene docstring
  (correspondence guides v true endpoints).
- Glow only from real point density; discrete data stays discrete.
- Trace/subset selection by a stated deterministic rule.

## 5. Commit

Scene + assets + (if registry changed) render_anim.py in ONE commit.
If a NEW scratch/scripts file rode along: commit, then regen
gen_codemap.py + gen_index.py, second commit (tracked-files-only
gotcha).
