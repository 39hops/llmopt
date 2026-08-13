"""TrainingMorph: the crystal forming — polar view drifting from a
newborn model (ep0) through ep1 to the final checkpoint. Data-true:
data/anim/morph.npz (scripts/anim_precompute.py --scene morph).

Render (from repo root):
  .venv-anim/bin/manim -qh --format=mp4 scripts/anim/training_morph.py TrainingMorph
  .venv-anim/bin/manim -ql --format=gif scripts/anim/training_morph.py TrainingMorph
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, UP, Dot, FadeIn, FadeOut, Scene, Text,
                   VGroup, config)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_theme import chrome, load_scene, ramp  # noqa: E402

ARR, META = load_scene("morph")
MODE = "dark"
C = chrome(META, MODE)
config.background_color = C["surface"]

N_DOTS = 3000
SPAN = 5.2


def _norm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.stack([x, y], axis=1).astype(np.float64)
    p -= p.mean(0)
    span = np.percentile(np.abs(p), 99) * 2 or 1.0
    return p / span * SPAN


class TrainingMorph(Scene):
    def construct(self):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(ARR["final_order"]), N_DOTS, replace=False)
        stages = ["ep0", "ep1", "final"]
        pos = {s: _norm(ARR[f"{s}_polar_x"][idx],
                        ARR[f"{s}_polar_y"][idx]) * 0.62
               for s in stages}
        col = {s: ARR[f"{s}_order"][idx] for s in stages}

        title = Text(META["title"], font="Inter", weight="LIGHT",
                     font_size=34, color=C["primary"]).to_edge(UP, buff=0.4)
        dots = VGroup(*[
            Dot([x, y - 0.3, 0], radius=0.022,
                color=ramp(META, o, MODE), fill_opacity=0.85)
            for (x, y), o in zip(pos["ep0"], col["ep0"])])

        def label(text):
            return Text(text, font="Inter", font_size=24,
                        color=C["secondary"]).to_edge(DOWN, buff=0.55)

        lab = label("epoch 0 — a newborn model")
        self.play(FadeIn(title, run_time=0.8),
                  FadeIn(dots, lag_ratio=0.002, run_time=2.0),
                  FadeIn(lab))
        self.wait(0.7)
        for stage, text in (("ep1", "epoch 1 — structure emerging"),
                            ("final", "final — the crystal")):
            new_lab = label(text)
            self.play(
                *[d.animate.move_to([p[0], p[1] - 0.3, 0])
                          .set_color(ramp(META, o, MODE))
                  for d, p, o in zip(dots, pos[stage], col[stage])],
                FadeOut(lab), FadeIn(new_lab),
                run_time=2.6)
            lab = new_lab
            self.wait(0.7)

        fence = Text(f"{META['provenance']} · @ {META['head']}",
                     font="JetBrains Mono", font_size=13,
                     color=C["muted"]).to_edge(DOWN, buff=0.25)
        self.play(FadeOut(lab), FadeIn(fence), run_time=0.8)
        self.wait(1.2)
