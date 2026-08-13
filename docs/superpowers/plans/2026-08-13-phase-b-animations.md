# Phase B: Manim Animations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three data-true manim scenes (CrystalRotation, TrainingMorph,
CrestRace) shipping GIF (README) + MP4 (LinkedIn) + poster PNG each.

**Architecture:** Lab venv precomputes projections to `.npz`
(torch stays out of the anim env); `.venv-anim` (manim 0.21.0,
python3.13, installed + smoked 2026-08-13) renders numpy-only scenes
from `scripts/anim/`, themed by a figstyle bridge.

**Tech Stack:** manim CE 0.21.0, numpy, ffmpeg 8.1.2 (present).
No Tex/MathTex (dvisvgm missing) — Text/MarkupText only.

## Global Constraints

- Scene data comes from real artifacts (checkpoints via precomputed
  npz; figures.json numbers read at precompute time) — never
  hardcoded in a scene.
- Provenance outro frame on every scene (ckpt sha or figures.json +
  repo HEAD, mono text).
- House look via the bridge only: surfaces #fcfcfb/#0f0f0e, inferno
  slice ramp sampled from figstyle.continuous, no colors named in
  scenes.
- Outputs: `docs/assets/anim/<scene>.gif` (<=10MB, 880w-class),
  `<scene>.mp4` (1080p H.264), `<scene>-poster.png`. media/ exhaust
  stays gitignored.
- Main suite must stay green with .venv-anim absent (scenes are not
  imported by lab code or tests).
- Commit rules unchanged (rc captured, ruff on lab-venv files,
  Co-Authored-By, no session URLs).

---

### Task 1: precompute + theme bridge

**Files:**
- Create: `scripts/anim_precompute.py` (lab venv; writes
  `data/anim/<name>.npz` — gitignored via data/ rules, regenerable)
- Create: `scripts/anim/house_theme.py` (anim venv; reads a
  JSON palette exported by the precompute step so figstyle itself
  never imports into the anim venv)

**Interfaces:**
- `anim_precompute.py --scene crystal|morph|crest` writes npz with
  arrays per scene + a `meta` JSON string (provenance, labels,
  ramp hexes sampled from figstyle.continuous at 16 stops).
- `house_theme.py`: `load_scene(name) -> (dict of arrays, meta)`;
  `ramp(t: float, mode) -> hex` interpolating the 16 stops.

- [ ] Steps: write precompute for the crystal scene (gallery19m_s1
  projections: pca/sphere/polar xy + rank order), run it, verify npz
  arrays' shapes; write house_theme loader; commit.

### Task 2: CrystalRotation scene

**Files:** `scripts/anim/crystal_rotation.py`

- [ ] Dots at pca coords colored by ramp(order), fade in; morph to
  sphere coords; morph to polar; hold; provenance outro. Render -ql
  GIF first (eyeball via poster/frames), then -qh MP4 + GIF + poster
  into docs/assets/anim/. Commit.

### Task 3: TrainingMorph scene

- [ ] Precompute ep0/ep1/final projections from
  gallery19m_s1_ep0/ep1 + final checkpoints (survey: exist); scene
  interpolates positions between epochs with labels. Same export
  set. Commit.

### Task 4: CrestRace scene

- [ ] Precompute routing_crest arms from docs/figures.json; rails
  grow to their values (zeros stay flat, labeled honestly); same
  export set. Commit.

### Task 5: close

- [ ] assets README gains the [ANIM] class row; README embeds ONE
  GIF max (decide with Artin which — default: none until he picks);
  suite + ruff; push; 3080 sync + hash; bounded CI watcher; BOARD.
