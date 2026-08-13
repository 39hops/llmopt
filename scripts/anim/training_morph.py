"""TrainingMorph: one model crossing training, in one fixed frame.
House grammar (storyboard doc 2026-08-13): Field = 2,900 neurons in
the FINAL checkpoint's whitened-PCA angle basis x absolute row norm;
Actors = 16 tracked neurons (deterministic stratified rule); Memory =
their true three-checkpoint trajectories; Geometry = quiet norm axis
+ the measured mean row norm; Text = one claim + stage labels;
Receipt = isolated end card.

Truth-language: checkpoint POSITIONS (ep0/ep1/final) and the tracked
identities are data (data/anim/morph.npz, fixed final-basis
projection); the motion BETWEEN checkpoints is linear interpolation,
a rendering choice - training did not move in straight lines. The
polyline Memory layer connects true endpoints only. Mean row norm
per stage is measured (meta), not styled.

Render (from repo root; ANIM_MODE=light for the light variant):
  .venv-anim/bin/manim -qh --format=mp4 scripts/anim/training_morph.py TrainingMorph
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, UR, Dot, FadeIn, FadeOut, MovingCameraScene,
                   RoundedRectangle, Text, TracedPath, VGroup, VMobject,
                   config)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_theme import chrome, load_scene, ramp  # noqa: E402

ARR, META = load_scene("morph")
MODE = os.environ.get("ANIM_MODE", "dark")
C = chrome(META, MODE)
config.background_color = C["surface"]

N_BG = 2900
X_SPAN = 11.6
Y_SPAN = 5.6
LAYERS = 8
ROWS_PER_LAYER = len(ARR["final_mag"]) // LAYERS
BG_OPACITY = 0.22

STAGES = ["ep0", "ep1", "final"]
STAGE_TEXT = {"ep0": "epoch 0 — born", "ep1": "epoch 1",
              "final": "final checkpoint"}
_all_mag = np.concatenate([ARR[f"{s}_mag"] for s in STAGES])
MAG_LO, MAG_HI = np.percentile(_all_mag, [1, 99])


def _pos(stage: str, idx: np.ndarray) -> np.ndarray:
    """Fixed frame: x = angle in the final checkpoint's whitened-PCA
    basis, y = absolute row norm on one shared scale."""
    x = ARR[f"{stage}_angle"][idx] / np.pi * (X_SPAN / 2)
    m = ARR[f"{stage}_mag"][idx]
    y = (np.clip(m, MAG_LO, MAG_HI) - MAG_LO) / (MAG_HI - MAG_LO)
    return np.stack([x, (y - 0.5) * Y_SPAN - 0.2], axis=1)


def _tracked_indices() -> np.ndarray:
    """Deterministic stratified sample: per layer, the neurons at the
    30th and 70th percentile of within-layer FINAL magnitude rank."""
    picks = []
    for layer in range(LAYERS):
        lo, hi = layer * ROWS_PER_LAYER, (layer + 1) * ROWS_PER_LAYER
        srt = np.argsort(ARR["final_mag"][lo:hi])
        for q in (0.30, 0.70):
            picks.append(lo + srt[int(q * (len(srt) - 1))])
    return np.array(picks)


class TrainingMorph(MovingCameraScene):
    def construct(self):
        tr_idx = _tracked_indices()
        rest = np.setdiff1d(np.arange(len(ARR["final_mag"])), tr_idx)
        bg_idx = np.random.default_rng(0).permutation(rest)[:N_BG]

        def col(stage, idx):
            m = ARR[f"{stage}_mag"][idx]
            return (np.clip(m, MAG_LO, MAG_HI) - MAG_LO) / (MAG_HI - MAG_LO)

        glow = VGroup(*[
            Dot([x, y, 0], radius=0.11,
                color=ramp(META, t, MODE), fill_opacity=0.045)
            for (x, y), t in zip(_pos("ep0", bg_idx),
                                 col("ep0", bg_idx))])
        bg = VGroup(*[
            Dot([x, y, 0], radius=0.028,
                color=ramp(META, t, MODE), fill_opacity=0.60)
            for (x, y), t in zip(_pos("ep0", bg_idx),
                                 col("ep0", bg_idx))])
        tracked = VGroup(*[
            Dot([x, y, 0], radius=0.055,
                color=ramp(META, t, MODE), fill_opacity=1.0,
                stroke_color=C["primary"], stroke_width=1.6)
            for (x, y), t in zip(_pos("ep0", tr_idx),
                                 col("ep0", tr_idx))])

        # Quiet geometry: norm ticks on the left edge.
        def y_of(v):
            return ((v - MAG_LO) / (MAG_HI - MAG_LO) - 0.5) * Y_SPAN - 0.2
        ticks = VGroup(*[
            Text(f"‖row‖ {v:.2f}", font="JetBrains Mono", font_size=14,
                 color=C["muted"]).move_to([-6.6, y_of(v), 0])
            for v in (0.55, 0.65, 0.75)])

        def stage_chrome(stage):
            lab = Text(STAGE_TEXT[stage], font="Inter", font_size=24,
                       color=C["secondary"]).to_edge(DOWN, buff=0.4)
            st = Text(f"mean ‖row‖ {META['mean_norm'][stage]:.3f}",
                      font="JetBrains Mono", font_size=18,
                      color=C["primary"]).to_corner(UR, buff=0.4)
            return lab, st

        # Beat 1 (0-3): field arrives at ep0; claim on scrim, out.
        self.add(glow, bg, tracked)
        claim = Text("Training, in one fixed frame.", font="Inter",
                     weight="LIGHT", font_size=34, color=C["primary"])
        scrim = RoundedRectangle(
            width=claim.width * 1.25, height=claim.height * 2.6,
            corner_radius=claim.height * 0.6, fill_color=C["surface"],
            fill_opacity=0.72, stroke_width=0).move_to(claim)
        self.play(FadeIn(scrim, run_time=0.7), FadeIn(claim, run_time=0.7))
        self.wait(1.1)
        lab, st = stage_chrome("ep0")
        self.play(FadeOut(claim, run_time=0.6), FadeOut(scrim, run_time=0.6),
                  FadeIn(ticks, run_time=0.6), FadeIn(lab), FadeIn(st))
        self.wait(0.8)
        # Memory: decaying tails during motion + true-endpoint polylines.
        trails = VGroup(*[
            TracedPath(d.get_center, stroke_width=2.0,
                       stroke_color=d.get_color(),
                       dissipating_time=0.6, stroke_opacity=[0.0, 0.5])
            for d in tracked])
        self.add(trails)
        prev = {i: p for i, p in zip(tr_idx, _pos("ep0", tr_idx))}
        # Beats 2-3: ep0 -> ep1 -> final, clean hold after each.
        for stage in ("ep1", "final"):
            new_lab, new_st = stage_chrome(stage)
            self.play(
                *[d.animate.move_to([x, y, 0])
                  for d, (x, y) in zip(glow, _pos(stage, bg_idx))],
                *[d.animate.move_to([x, y, 0])
                  for d, (x, y) in zip(bg, _pos(stage, bg_idx))],
                *[d.animate.move_to([x, y, 0])
                          .set_color(ramp(META, t, MODE))
                  for d, (x, y), t in zip(tracked, _pos(stage, tr_idx),
                                          col(stage, tr_idx))],
                FadeOut(lab), FadeIn(new_lab),
                FadeOut(st), FadeIn(new_st),
                run_time=2.6)
            # Memory layer: connect the TRUE endpoints just traversed.
            segs = VGroup()
            for i, (x, y) in zip(tr_idx, _pos(stage, tr_idx)):
                seg = VMobject(stroke_width=1.4, stroke_opacity=0.4,
                               stroke_color=C["secondary"])
                seg.set_points_as_corners(
                    [np.array([*prev[i], 0]), np.array([x, y, 0])])
                segs.add(seg)
                prev[i] = np.array([x, y])
            self.play(FadeIn(segs, run_time=0.4))
            self.wait(1.1)
            lab, st = new_lab, new_st
        self.remove(trails)
        self.wait(1.6)               # clean final hold - the poster frame
        # Receipt: dedicated end card, never over live data.
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
        fence = Text(f"{META['provenance']} · @ {META['head']}",
                     font="JetBrains Mono", font_size=11,
                     color=C["muted"]).move_to([0, 0, 0])
        self.play(FadeIn(fence, run_time=0.4))
        self.wait(0.9)
