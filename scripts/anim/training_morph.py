"""TrainingMorph: the crystal forming — polar view (angle in the
FINAL checkpoint's fixed whitened-PCA basis, y = absolute row norm)
drifting from a newborn model (ep0) through ep1 to the final
checkpoint. One coordinate system across all stages, the measured
mean row norm on screen, and 220 tracked neurons over a faint
density background. Data-true: data/anim/morph.npz
(scripts/anim_precompute.py --scene morph).

Render (from repo root; ANIM_MODE=light for the light variant):
  .venv-anim/bin/manim -qh --format=mp4 scripts/anim/training_morph.py TrainingMorph
  .venv-anim/bin/manim -ql --format=gif scripts/anim/training_morph.py TrainingMorph
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, RIGHT, UP, Dot, FadeIn, FadeOut, Scene, Text,
                   VGroup, config)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_theme import chrome, load_scene, ramp  # noqa: E402

ARR, META = load_scene("morph")
MODE = os.environ.get("ANIM_MODE", "dark")
C = chrome(META, MODE)
config.background_color = C["surface"]

N_BG = 2600
N_TRACK = 220
X_SPAN = 5.6      # angle [-pi, pi] maps to this width
Y_SPAN = 3.4      # magnitude [global p1, global p99] maps to this height

STAGES = ["ep0", "ep1", "final"]
# One scale for every stage: the whole point of the redesign.
_all_mag = np.concatenate([ARR[f"{s}_mag"] for s in STAGES])
MAG_LO, MAG_HI = np.percentile(_all_mag, [1, 99])


def _pos(stage: str, idx: np.ndarray) -> np.ndarray:
    x = ARR[f"{stage}_angle"][idx] / np.pi * (X_SPAN / 2)
    m = ARR[f"{stage}_mag"][idx]
    y = (np.clip(m, MAG_LO, MAG_HI) - MAG_LO) / (MAG_HI - MAG_LO)
    return np.stack([x, (y - 0.5) * Y_SPAN - 0.25], axis=1)


def _col(stage: str, idx: np.ndarray) -> np.ndarray:
    # Absolute magnitude on the shared scale — color moves when the
    # weights actually grow, unlike the old rank coloring.
    m = ARR[f"{stage}_mag"][idx]
    return (np.clip(m, MAG_LO, MAG_HI) - MAG_LO) / (MAG_HI - MAG_LO)


class TrainingMorph(Scene):
    def construct(self):
        rng = np.random.default_rng(0)
        n = len(ARR["final_mag"])
        pick = rng.permutation(n)
        bg_idx, tr_idx = pick[:N_BG], pick[N_BG:N_BG + N_TRACK]

        title = Text(META["title"], font="Inter", weight="LIGHT",
                     font_size=34, color=C["primary"]).to_edge(UP, buff=0.35)

        bg = VGroup(*[
            Dot([x, y, 0], radius=0.016, color=C["muted"],
                fill_opacity=0.14)
            for x, y in _pos("ep0", bg_idx)])
        tracked = VGroup(*[
            Dot([x, y, 0], radius=0.034,
                color=ramp(META, t, MODE), fill_opacity=0.95)
            for (x, y), t in zip(_pos("ep0", tr_idx),
                                 _col("ep0", tr_idx))])

        def label(text):
            return Text(text, font="Inter", font_size=24,
                        color=C["secondary"]).to_edge(DOWN, buff=0.5)

        def stat(stage):
            return Text(f"mean ‖row‖ = {META['mean_norm'][stage]:.3f}",
                        font="JetBrains Mono", font_size=20,
                        color=C["primary"]).to_corner(UP + RIGHT, buff=0.45)

        lab = label("epoch 0 — a newborn model")
        st = stat("ep0")
        self.play(FadeIn(title, run_time=0.6),
                  FadeIn(bg, run_time=1.0),
                  FadeIn(tracked, lag_ratio=0.002, run_time=1.2),
                  FadeIn(lab), FadeIn(st))
        self.wait(0.4)
        for stage, text in (("ep1", "epoch 1 — structure emerging"),
                            ("final", "final — the crystal")):
            new_lab, new_st = label(text), stat(stage)
            self.play(
                *[d.animate.move_to([x, y, 0])
                  for d, (x, y) in zip(bg, _pos(stage, bg_idx))],
                *[d.animate.move_to([x, y, 0])
                          .set_color(ramp(META, t, MODE))
                  for d, (x, y), t in zip(tracked, _pos(stage, tr_idx),
                                          _col(stage, tr_idx))],
                FadeOut(lab), FadeIn(new_lab),
                FadeOut(st), FadeIn(new_st),
                run_time=1.8)
            lab, st = new_lab, new_st
            self.wait(0.4)

        fence = Text(f"{META['provenance']} · @ {META['head']}",
                     font="JetBrains Mono", font_size=13,
                     color=C["muted"]).to_edge(DOWN, buff=0.25)
        self.play(FadeOut(lab), FadeIn(fence), run_time=0.6)
        self.wait(0.8)
