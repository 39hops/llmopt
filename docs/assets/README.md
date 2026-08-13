# docs/assets — figure classes and where each comes from

Every image here belongs to exactly one class, and since 2026-08-13
the directory layout mirrors the classes. Before touching a file,
find its row; before adding one, pick its class. This is the
figure-side counterpart of docs/CODEMAP.md's frozen/adopted split.

```
docs/assets/
├── README.md
├── web/                  # [PUBLISHED] — figures.json-sourced pairs
├── hero/                 # [HERO] — neurons-19m-{light,dark}.png
├── gallery/              # [R] — regenerable renders
├── anim/                 # [ANIM] — GIF/MP4/poster triplets
└── archive/2026-08-12/   # [H] — frozen pixels, moved as-is
```

`figs/<date>/` at repo root stays separate: run exhaust, never a
publication surface. Historical docs (handoffs, RESULTS) keep their
pre-move paths as evidence; only live surfaces (top README, paper,
this file) are link-checked (`tests/test_asset_links.py`).

## Classes

**[PUBLISHED]** — carries booked numbers, cited by README/paper.
Single source of truth is `docs/figures.json`; renderers are
`llmopt/figures/figsvg.py` (SVG) + `scripts/gen_figures_web.py`
(SVG -> PNG via headless Chrome, Mac-only). Regenerate, never edit
pixels. Files: `web/*.{svg,png}` (light + `-dark` pairs).

**[HERO]** — the README front-door render. `scripts/render_hero_neurons.py`
on a catalogued checkpoint; provenance footer (ckpt sha + repo HEAD)
is part of the image. Files: `hero/neurons-19m-light.png`,
`hero/neurons-19m-dark.png`. 256-color quantized on install (Pillow
median-cut; the renderer writes full RGB to scratch first).

**[R] reproducible gallery** — figures regenerable with NO
reconstruction: the script alone reproduces the file (data inline or
a recorded invocation). `scripts/plot_gt1_crest.py`,
`scripts/plot_identity_crest.py`. Files live under `gallery/`:
`gt1-crest-small-multiples.png`, `identity-crest-fresh-seeds.png`
(light + `-dark` pairs since the 2026-08-13 restyle).
`scripts/plot_neurons.py` is the tool for NEW gallery renders — a
render is only [R] if its exact invocation is recorded (in the script
or REPRODUCE); otherwise it freezes to [H] on landing. The 2026-08-13
overhaul reclassified the old plot_neurons set to [H] on exactly that
test: checkpoints survive, the invocations were never recorded.

**[ANIM] animations** — data-true manim scenes; each ships a
GIF (README-embeddable, ≤10MB) + MP4 (1080p, LinkedIn) + poster PNG
triplet under `anim/`. Scene data is precomputed from real artifacts
(`scripts/anim_precompute.py`: checkpoints -> npz, figures.json ->
npz) and scenes (`scripts/anim/`) render numpy-only in `.venv-anim`
(manim CE; see scratch/manim_feasibility_2026-08-13.md). Regenerable:
precompute + `manim -qh` + the ffmpeg palette encode. Provenance
outro frame is part of every scene. Scenes ship a `-light` triplet (`ANIM_MODE=light`) where the
render_anim registry says so (expert_atlas is dark-only); posters are a
mid-scene frame, the reduced-motion path is poster + MP4 link.

**[H] historic / frozen pixels** — no surviving script+checkpoint
pair, or cited by a booked entry as-is. Evidence record: never
regenerate, never delete without an Artin GO, never point new docs
at them. Files live under `archive/<date>/`. `archive/2026-08-13/` holds the
retired crystal_rotation triplets (animation retired by Artin ruling
2026-08-13 — a projection morph of one frozen checkpoint has no
genuine dynamics; the scene script stays in scripts/anim/ as record).
`archive/2026-08-12/` includes
`neurons-19m.png` (the old hero — cited by
`scratch/crystal_recreate_test.py`; its caption text is superseded,
which is WHY it is frozen rather than shipped), `neurons-19m-zoom.png`,
`three-minds-*`, crystal-era renders.

## Rules

- New figures land as [PUBLISHED] (numbers) or [HERO]/[R] (weight
  renders). [H] only grows by freezing, never by authoring.
- Style for anything new: `llmopt/figures/figstyle.py` (validated
  palette, vendored fonts, light+dark CHROME). The old dark-github
  `#0d1117` look in [R]/[H] files is historical; do not imitate it.
- `figs/<date>/` at repo root is run exhaust from gallery batteries
  (logs-doctrine class), not a publication surface.
- Big-model anatomy sources: `checkpoints/v4flash_f1/` (DeepSeek
  V4-Flash expert shards, on disk) and the `k3_expert_*` byte-range
  pulls (`scratch/k3_expert_demo.py`). Dot views only for models
  with no public base pair.
