# Routing Canyon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat expert-atlas animatic with a cinematic,
evidence-faithful routing-canyon flagship.

**Architecture:** Pure NumPy helpers own normalization, discrete geometry,
and deterministic selection. The Manim scene consumes those transforms and
owns only composition, timing, and camera choreography. The existing frozen
`data/anim/atlas.npz` remains the sole evidence input.

**Tech Stack:** Python 3.11, NumPy, Manim Community 0.21, pytest, ffmpeg.

**Spec:** `docs/superpowers/specs/2026-08-13-routing-canyon-design.md`

## Global Constraints

- Do not edit `data/anim/atlas.npz` or `scratch/atlas_precompute.py`.
- Keep prefill/decode normalization and pooled-rank ordering fixed.
- Use snake_case for every new public asset.
- Do not generate a README GIF before visual approval.

---

### Task 1: Evidence-preserving visual transforms

**Files:**
- Create: `llmopt/figures/atlas_visuals.py`
- Create: `tests/test_atlas_visuals.py`

**Interfaces:**
- Produces: `phase_luminance`, `canyon_points`,
  `perspective_projection`, `deterministic_focus_pair`, and `block_field`.

- [x] Write failing tests for shared phase scale, discrete cell geometry,
  stable pairing, dark gutters, and perspective depth.
- [x] Run the focused tests and confirm missing-interface failures.
- [x] Implement the smallest pure NumPy transforms.
- [x] Run the focused tests and confirm all contracts pass.

### Task 2: Cinematic scene

**Files:**
- Modify: `scripts/anim/expert_atlas.py`

**Interfaces:**
- Consumes: `data/anim/atlas.npz` and Task 1 transforms.
- Produces: Manim scene `ExpertAtlas`.

- [x] Replace the flat opening with a perspective routing canyon.
- [x] Accumulate the eight real prefill snapshots into terrain profiles.
- [x] Flatten the same values into the discrete atlas.
- [x] Add the phase change, filled carrier constellation, matched-pair
  construction, set-level outcome, and isolated provenance card.
- [x] Render a low-resolution MP4 and inspect sampled beat frames.
- [x] Iterate until the visual hierarchy survives the contact sheet.

### Task 3: Verification and checkpoint export

**Files:**
- Modify: `scripts/INDEX.md`
- Create: `docs/assets/anim/expert_atlas.mp4`
- Create: `docs/assets/anim/expert_atlas_poster.png`

**Interfaces:**
- Consumes: approved scene from Task 2.
- Produces: snake_case review assets and a regenerated script index.

- [x] Run focused tests, ruff, and the repository suite.
- [x] Render the final review master and extract the clean constellation
  poster frame.
- [x] Regenerate `scripts/INDEX.md` and verify asset dimensions/duration.
