"""CrystalRotation flagship: the same 12,288 weight vectors reveal
different structure under three mathematical views. Cinematic cut:
open inside the cloud, pull back, ignite a tracked subset over a
density field, morph PCA -> sphere -> polar with identity-preserving
trails, compose the three views, provenance outro. ~15.5s.
Storyboard: docs/superpowers/plans/2026-08-13-anim-flagship-storyboard.md.

Every layer is data-derived: positions from data/anim/crystal.npz
(scripts/anim_precompute.py --scene crystal), color = house magnitude
ramp, density field = the real remaining neurons at low opacity, dot
radius varies subtly by the neuron's true layer (8 x 1536 row order),
trails = the tracked neurons' actual interpolation paths.

Render (from repo root; ANIM_MODE=light for the light variant):
  .venv-anim/bin/manim -qh --format=mp4 scripts/anim/crystal_rotation.py CrystalRotation
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, Dot, FadeIn, FadeOut, MovingCameraScene,
                   Text, TracedPath, VGroup, config)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_theme import chrome, load_scene, ramp  # noqa: E402

ARR, META = load_scene("crystal")
MODE = os.environ.get("ANIM_MODE", "dark")
C = chrome(META, MODE)
config.background_color = C["surface"]

N_BG = 2900
N_TRACK = 60
SPAN = 9.5           # cloud fills most of the 14.2-unit frame width
LAYERS = 8           # 12288 rows = 8 layers x 1536, in row order
ROWS_PER_LAYER = len(ARR["order"]) // LAYERS


def _norm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Center + scale a projection into scene coordinates, per axis
    (polar's angle spans -pi..pi while magnitude spans ~0.1; a joint
    scale flattens the cloud to a line)."""
    p = np.stack([x, y], axis=1).astype(np.float64)
    p -= p.mean(0)
    for a in (0, 1):
        span = np.percentile(np.abs(p[:, a]), 99) * 2 or 1.0
        p[:, a] /= span
    return p * SPAN


def _radius(i: int, base: float) -> float:
    # Depth cue from true structure: later layers slightly larger.
    return base * (0.82 + 0.36 * (i // ROWS_PER_LAYER) / (LAYERS - 1))


class CrystalRotation(MovingCameraScene):
    def construct(self):
        rng = np.random.default_rng(0)
        pick = rng.permutation(len(ARR["order"]))
        bg_idx, tr_idx = pick[:N_BG], pick[N_BG:N_BG + N_TRACK]
        all_idx = np.concatenate([bg_idx, tr_idx])
        views = {m: _norm(ARR[f"{m}_x"][all_idx], ARR[f"{m}_y"][all_idx])
                 for m in ("pca", "sphere", "polar")}
        order = ARR["order"][all_idx]
        frame = self.camera.frame

        bg = VGroup(*[
            Dot([x, y, 0], radius=_radius(int(i), 0.030),
                color=ramp(META, o, MODE), fill_opacity=0.65)
            for (x, y), o, i in zip(views["pca"][:N_BG],
                                    order[:N_BG], bg_idx)])
        tracked = VGroup(*[
            Dot([x, y, 0], radius=_radius(int(i), 0.052),
                color=ramp(META, o, MODE), fill_opacity=0.9)
            for (x, y), o, i in zip(views["pca"][N_BG:],
                                    order[N_BG:], tr_idx)])

        # Beat 1 (0-2.5): inside the cloud; claim fades in and out.
        self.add(bg, tracked)
        frame.scale(0.30).move_to(bg.get_center() + np.array([0.6, 0.3, 0]))
        claim = Text("12,288 weight vectors — one model", font="Inter",
                     weight="LIGHT", font_size=30, color=C["primary"])
        claim.scale(0.30).move_to(frame.get_center())
        self.play(FadeIn(claim, run_time=0.8),
                  frame.animate(run_time=2.5).scale(1.55).move_to(
                      bg.get_center() + np.array([0.2, 0.1, 0])))
        # Beat 2 (2.5-5.0): pull back to the full geometry.
        lab_pca = Text("PCA — global variation", font="Inter",
                       font_size=26, color=C["secondary"])
        lab_pca.next_to(bg, DOWN, buff=0.5)
        self.play(FadeOut(claim, run_time=0.6),
                  frame.animate(run_time=2.5).scale(1 / 0.465).move_to(
                      [0, -0.2, 0]))
        self.play(FadeIn(lab_pca, run_time=0.6))
        # Beat 3 (5.0-6.5): tracked subset ignites; cloud recedes.
        self.play(
            bg.animate(run_time=1.5).set_fill(opacity=0.13),
            *[d.animate(run_time=1.5)
              .set_fill(opacity=1.0).set_stroke(C["primary"], width=1.4)
              for d in tracked],
            FadeOut(lab_pca))
        trails = VGroup(*[
            TracedPath(d.get_center, stroke_width=1.6,
                       stroke_color=d.get_color(), stroke_opacity=0.45)
            for d in tracked])
        self.add(trails)
        # Beats 4-5: morphs with identity-preserving trails.
        for method, text in (("sphere", "SPHERE — magnitude removed"),
                             ("polar", "POLAR — phase vs magnitude")):
            target = views[method]
            lab = Text(text, font="Inter", font_size=26,
                       color=C["secondary"]).to_edge(DOWN, buff=0.45)
            self.play(
                *[d.animate.move_to([p[0], p[1], 0])
                  for d, p in zip(bg, target[:N_BG])],
                *[d.animate.move_to([p[0], p[1], 0])
                  for d, p in zip(tracked, target[N_BG:])],
                FadeIn(lab, run_time=0.8),
                run_time=2.4)
            self.play(FadeOut(lab, run_time=0.6))
        # Beat 6 (12.5-15.5): compose the three views.
        self.remove(trails)
        echoes = VGroup()
        for k, (method, mlab) in enumerate((("pca", "PCA"),
                                            ("sphere", "SPHERE"))):
            ex = 4.55
            ey = 1.6 - k * 3.3
            pts = views[method] * 0.16
            grp = VGroup(*[
                Dot([x + ex, y + ey, 0], radius=0.012,
                    color=ramp(META, o, MODE), fill_opacity=0.30)
                for (x, y), o in zip(pts[:N_BG:3], order[:N_BG:3])])
            grp.add(*[
                Dot([x + ex, y + ey, 0], radius=0.026,
                    color=ramp(META, o, MODE), fill_opacity=1.0)
                for (x, y), o in zip(pts[N_BG:], order[N_BG:])])
            grp.add(Text(mlab, font="Inter", font_size=17,
                         color=C["muted"]).move_to([ex, ey - 1.45, 0]))
            echoes.add(grp)
        polar_lab = Text("POLAR", font="Inter", font_size=17,
                         color=C["muted"])
        cloud = VGroup(bg, tracked)
        self.play(
            cloud.animate(run_time=1.6).scale(0.62).move_to([-2.3, -0.1, 0]),
            FadeIn(echoes, run_time=1.6))
        polar_lab.move_to([-2.3, -2.6, 0])
        fence = Text(f"{META['provenance']} · @ {META['head']}",
                     font="JetBrains Mono", font_size=12,
                     color=C["muted"]).to_edge(DOWN, buff=0.18)
        self.play(FadeIn(polar_lab, run_time=0.5),
                  FadeIn(fence, run_time=0.5))
        self.wait(1.3)
