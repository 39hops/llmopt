"""CrystalRotation: the 19M gate-neuron cloud morphing through its
three projections. Data-true: positions come from
data/anim/crystal.npz (scripts/anim_precompute.py --scene crystal);
colors are the house magnitude ramp; outro is the provenance fence.

Render (from repo root):
  .venv-anim/bin/manim -qh --format=mp4 scripts/anim/crystal_rotation.py CrystalRotation
  .venv-anim/bin/manim -ql --format=gif scripts/anim/crystal_rotation.py CrystalRotation
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, UP, Dot, FadeIn, FadeOut, Scene, Text,
                   VGroup, config)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_theme import chrome, load_scene, ramp  # noqa: E402

ARR, META = load_scene("crystal")
MODE = "dark"
C = chrome(META, MODE)
config.background_color = C["surface"]

N_DOTS = 3000  # subsample for tractable mobject count
SPAN = 5.2     # scene units the cloud occupies


def _norm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Center + scale a projection into scene coordinates, per axis:
    polar's angle spans -pi..pi while magnitude spans ~0.1, so a
    joint scale flattens the cloud to a line."""
    p = np.stack([x, y], axis=1).astype(np.float64)
    p -= p.mean(0)
    for a in (0, 1):
        span = np.percentile(np.abs(p[:, a]), 99) * 2 or 1.0
        p[:, a] /= span
    return p * SPAN


class CrystalRotation(Scene):
    def construct(self):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(ARR["order"]), N_DOTS, replace=False)
        order = ARR["order"][idx]
        views = {m: _norm(ARR[f"{m}_x"][idx], ARR[f"{m}_y"][idx])
                 for m in ("pca", "sphere", "polar")}

        title = Text(META["title"], font="Inter", weight="LIGHT",
                     font_size=34, color=C["primary"]).to_edge(UP, buff=0.4)
        dots = VGroup(*[
            Dot([x, y, 0], radius=0.022,
                color=ramp(META, o, MODE), fill_opacity=0.85)
            for (x, y), o in zip(views["pca"] * 0.62, order)])
        dots.shift(DOWN * 0.3)

        def label(text):
            return Text(text, font="Inter", font_size=24,
                        color=C["secondary"]).to_edge(DOWN, buff=0.55)

        lab = label("PCA — global axes")
        self.play(FadeIn(title, run_time=0.8),
                  FadeIn(dots, lag_ratio=0.002, run_time=2.2),
                  FadeIn(lab))
        self.wait(0.8)
        for method, text in (("sphere", "SPHERE — directions only"),
                             ("polar", "POLAR — phase vs magnitude")):
            new_lab = label(text)
            target = views[method] * 0.62
            self.play(
                *[d.animate.move_to([p[0], p[1] - 0.3, 0])
                  for d, p in zip(dots, target)],
                FadeOut(lab), FadeIn(new_lab),
                run_time=2.4)
            lab = new_lab
            self.wait(0.8)

        fence = Text(f"{META['provenance']} · @ {META['head']}",
                     font="JetBrains Mono", font_size=16,
                     color=C["muted"]).to_edge(DOWN, buff=0.25)
        self.play(FadeOut(lab), FadeIn(fence), run_time=0.8)
        self.wait(1.2)
