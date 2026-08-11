# docs/assets — figure classes and where each comes from

Every image here belongs to exactly one class. Before touching a
file, find its row; before adding one, pick its class. This is the
figure-side counterpart of docs/CODEMAP.md's frozen/adopted split.

## Classes

**[PUBLISHED]** — carries booked numbers, cited by README/paper.
Single source of truth is `docs/figures.json`; renderers are
`llmopt/lab/figsvg.py` (SVG) + `scripts/gen_figures_web.py`
(SVG -> PNG via headless Chrome, Mac-only). Regenerate, never edit
pixels. Files: `web/*.{svg,png}` (light + `-dark` pairs).

**[HERO]** — the README front-door render. `scripts/render_hero_neurons.py`
on a catalogued checkpoint; provenance footer (ckpt sha + repo HEAD)
is part of the image. Files: `neurons-19m-light.png`,
`neurons-19m-dark.png`. 256-color quantized on install (Pillow
median-cut; the renderer writes full RGB to scratch first).

**[R] reproducible gallery** — weight-space renders whose script and
checkpoint both still exist; regenerable at will.
`scripts/plot_neurons.py` (pca/sphere/polar/displace),
`scripts/plot_gt1_crest.py`, `scripts/plot_identity_crest.py`.
Files include `gt1-crest-small-multiples.png`,
`identity-crest-fresh-seeds.png`, `neurons-gen6-*`, `neurons-pca-*`,
the four-diets set, `neurons-113m-growth-*` (gallery pair).

**[H] historic / frozen pixels** — no surviving script+checkpoint
pair, or cited by a booked entry as-is. Evidence record: never
regenerate, never delete without an Artin GO, never point new docs
at them. Files include `neurons-19m.png` (the old hero — cited by
`scratch/crystal_recreate_test.py`; its caption text is superseded,
which is WHY it is frozen rather than shipped), `neurons-19m-zoom.png`,
`three-minds-*`, crystal-era renders.

## Rules

- New figures land as [PUBLISHED] (numbers) or [HERO]/[R] (weight
  renders). [H] only grows by freezing, never by authoring.
- Style for anything new: `llmopt/lab/figstyle.py` (validated
  palette, vendored fonts, light+dark CHROME). The old dark-github
  `#0d1117` look in [R]/[H] files is historical; do not imitate it.
- `figs/<date>/` at repo root is run exhaust from gallery batteries
  (logs-doctrine class), not a publication surface.
- Big-model anatomy sources: `checkpoints/v4flash_f1/` (DeepSeek
  V4-Flash expert shards, on disk) and the `k3_expert_*` byte-range
  pulls (`scratch/k3_expert_demo.py`). Dot views only for models
  with no public base pair.
