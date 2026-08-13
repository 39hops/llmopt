"""CrystalRotation: three views of the same weights — the 19M
gate-neuron cloud morphing between its three projections (a
projection morph, not a rotation). A fixed set of 60 highlighted
neurons is tracked across all three views so the transformation is
followable; the rest is a dimmed background. Data-true: positions
come from data/anim/crystal.npz (scripts/anim_precompute.py --scene
crystal); colors are the house magnitude ramp; outro is the
provenance fence.

Render (from repo root; ANIM_MODE=light for the light variant):
  .venv-anim/bin/manim -qh --format=mp4 scripts/anim/crystal_rotation.py CrystalRotation
  .venv-anim/bin/manim -ql --format=gif scripts/anim/crystal_rotation.py CrystalRotation
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, UP, Dot, FadeIn, FadeOut, Scene, Text,
                   VGroup, config)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_theme import chrome, load_scene, ramp  # noqa: E402

ARR, META = load_scene("crystal")
MODE = os.environ.get("ANIM_MODE", "dark")
C = chrome(META, MODE)
config.background_color = C["surface"]

N_BG = 2600    # dimmed background subsample
N_TRACK = 60   # highlighted neurons tracked across all three views
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
        pick = rng.permutation(len(ARR["order"]))
        bg_idx, tr_idx = pick[:N_BG], pick[N_BG:N_BG + N_TRACK]
        # Normalize each view over the union so background and tracked
        # dots share one frame per view.
        all_idx = np.concatenate([bg_idx, tr_idx])
        views = {m: _norm(ARR[f"{m}_x"][all_idx],
                          ARR[f"{m}_y"][all_idx]) * 0.62
                 for m in ("pca", "sphere", "polar")}

        title = Text(META["title"], font="Inter", weight="LIGHT",
                     font_size=34, color=C["primary"]).to_edge(UP, buff=0.4)
        bg = VGroup(*[
            Dot([x, y - 0.3, 0], radius=0.018,
                color=ramp(META, o, MODE), fill_opacity=0.28)
            for (x, y), o in zip(views["pca"][:N_BG],
                                 ARR["order"][bg_idx])])
        tracked = VGroup(*[
            Dot([x, y - 0.3, 0], radius=0.042,
                color=ramp(META, o, MODE), fill_opacity=1.0,
                stroke_color=C["primary"], stroke_width=1.2)
            for (x, y), o in zip(views["pca"][N_BG:],
                                 ARR["order"][tr_idx])])

        def label(text):
            return Text(text, font="Inter", font_size=24,
                        color=C["secondary"]).to_edge(DOWN, buff=0.55)

        lab = label("PCA — global axes")
        self.play(FadeIn(title, run_time=0.6),
                  FadeIn(bg, run_time=1.0),
                  FadeIn(tracked, lag_ratio=0.01, run_time=1.4),
                  FadeIn(lab))
        self.wait(0.4)
        for method, text in (("sphere", "SPHERE — directions only"),
                             ("polar", "POLAR — phase vs magnitude")):
            new_lab = label(text)
            target = views[method]
            self.play(
                *[d.animate.move_to([p[0], p[1] - 0.3, 0])
                  for d, p in zip(bg, target[:N_BG])],
                *[d.animate.move_to([p[0], p[1] - 0.3, 0])
                  for d, p in zip(tracked, target[N_BG:])],
                FadeOut(lab), FadeIn(new_lab),
                run_time=1.6)
            lab = new_lab
            self.wait(0.4)

        fence = Text(f"{META['provenance']} · @ {META['head']}",
                     font="JetBrains Mono", font_size=16,
                     color=C["muted"]).to_edge(DOWN, buff=0.25)
        self.play(FadeOut(lab), FadeIn(fence), run_time=0.6)
        self.wait(0.8)
