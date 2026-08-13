"""CrystalRotation flagship: the same 12,288 weight vectors reveal
different structure under three mathematical views. Cinematic cut:
open inside the cloud, pull back, ignite a tracked subset over the
density field, morph PCA -> sphere -> polar with decaying
correspondence trails, compose the three views, provenance outro.
Storyboard: docs/superpowers/plans/2026-08-13-anim-flagship-storyboard.md.

Data-derived layers: projection endpoints from data/anim/crystal.npz
(scripts/anim_precompute.py --scene crystal), color = house magnitude
ramp, density field = the real remaining neurons at low opacity.
Tracked identities are real and preserved (deterministic stratified
sample: 2 neurons per layer at the 30th/70th within-layer magnitude
quantiles); the paths BETWEEN projection endpoints are
animation-generated correspondence guides, not data.

Render (from repo root; ANIM_MODE=light for the light variant):
  .venv-anim/bin/manim -qh --format=mp4 scripts/anim/crystal_rotation.py CrystalRotation
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, Dot, FadeIn, FadeOut, MovingCameraScene,
                   RoundedRectangle, Text, TracedPath, VGroup, config)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_theme import chrome, load_scene, ramp  # noqa: E402

ARR, META = load_scene("crystal")
MODE = os.environ.get("ANIM_MODE", "dark")
C = chrome(META, MODE)
config.background_color = C["surface"]

N_BG = 2900
SPAN = 9.5           # cloud fills most of the 14.2-unit frame width
LAYERS = 8           # 12288 rows = 8 layers x 1536, in row order
ROWS_PER_LAYER = len(ARR["order"]) // LAYERS
BG_OPACITY = 0.20    # density field stays visible through the morphs


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


def _tracked_indices() -> np.ndarray:
    """Deterministic stratified sample: per layer, the neurons at the
    30th and 70th percentile of within-layer magnitude rank."""
    picks = []
    for layer in range(LAYERS):
        lo, hi = layer * ROWS_PER_LAYER, (layer + 1) * ROWS_PER_LAYER
        mags = ARR["order"][lo:hi]
        srt = np.argsort(mags)
        for q in (0.30, 0.70):
            picks.append(lo + srt[int(q * (len(srt) - 1))])
    return np.array(picks)


class CrystalRotation(MovingCameraScene):
    def construct(self):
        tr_idx = _tracked_indices()
        rest = np.setdiff1d(np.arange(len(ARR["order"])), tr_idx)
        bg_idx = np.random.default_rng(0).permutation(rest)[:N_BG]
        all_idx = np.concatenate([bg_idx, tr_idx])
        views = {m: _norm(ARR[f"{m}_x"][all_idx], ARR[f"{m}_y"][all_idx])
                 for m in ("pca", "sphere", "polar")}
        order = ARR["order"][all_idx]
        frame = self.camera.frame

        bg = VGroup(*[
            Dot([x, y, 0], radius=0.030,
                color=ramp(META, o, MODE), fill_opacity=0.65)
            for (x, y), o in zip(views["pca"][:N_BG], order[:N_BG])])
        tracked = VGroup(*[
            Dot([x, y, 0], radius=0.055,
                color=ramp(META, o, MODE), fill_opacity=0.9)
            for (x, y), o in zip(views["pca"][N_BG:], order[N_BG:])])

        def edge_label(text):
            return Text(text, font="Inter", font_size=26,
                        color=C["secondary"]).to_edge(DOWN, buff=0.45)

        # Beat 1 (0-2.5): inside the cloud; claim on a local scrim.
        self.add(bg, tracked)
        frame.scale(0.30).move_to(bg.get_center() + np.array([0.6, 0.3, 0]))
        claim = Text("12,288 weights. Three views.", font="Inter",
                     weight="LIGHT", font_size=34, color=C["primary"])
        claim.scale(0.30).move_to(frame.get_center())
        scrim = RoundedRectangle(
            width=claim.width * 1.25, height=claim.height * 2.6,
            corner_radius=claim.height * 0.6, fill_color=C["surface"],
            fill_opacity=0.72, stroke_width=0).move_to(claim)
        self.play(FadeIn(scrim, run_time=0.8), FadeIn(claim, run_time=0.8),
                  frame.animate(run_time=2.5).scale(1.55).move_to(
                      bg.get_center() + np.array([0.2, 0.1, 0])))
        # Beat 2 (2.5-5.3): pull back; PCA gets the same label treatment.
        self.play(FadeOut(claim, run_time=0.6), FadeOut(scrim, run_time=0.6),
                  frame.animate(run_time=2.5).scale(1 / 0.465).move_to(
                      [0, -0.2, 0]))
        lab = edge_label("PCA — global variation")
        self.play(FadeIn(lab, run_time=0.5))
        self.wait(0.6)
        # Beat 3 (5.3-6.8): tracked subset ignites; cloud recedes.
        self.play(
            bg.animate(run_time=1.5).set_fill(opacity=BG_OPACITY),
            *[d.animate(run_time=1.5)
              .set_fill(opacity=1.0).set_stroke(C["primary"], width=1.6)
              for d in tracked],
            FadeOut(lab))
        # Decaying comet tails: correspondence guides, not data.
        trails = VGroup(*[
            TracedPath(d.get_center, stroke_width=2.2,
                       stroke_color=d.get_color(),
                       dissipating_time=0.55, stroke_opacity=[0.0, 0.55])
            for d in tracked])
        self.add(trails)
        # Beats 4-5: morphs, then a clean readable hold on each view.
        for method, text in (("sphere", "SPHERE — magnitude removed"),
                             ("polar", "POLAR — phase vs magnitude")):
            target = views[method]
            lab = edge_label(text)
            self.play(
                *[d.animate.move_to([p[0], p[1], 0])
                  for d, p in zip(bg, target[:N_BG])],
                *[d.animate.move_to([p[0], p[1], 0])
                  for d, p in zip(tracked, target[N_BG:])],
                FadeIn(lab, run_time=0.8),
                run_time=2.2)
            self.wait(0.9)          # trails decay; geometry reads clean
            self.play(FadeOut(lab, run_time=0.5))
        self.remove(trails)
        # Beat 6: compose the three views; hold ~2s before provenance.
        echoes = VGroup()
        for k, (method, mlab) in enumerate((("pca", "PCA"),
                                            ("sphere", "SPHERE"))):
            ex, ey = 4.55, 1.6 - k * 3.3
            pts = views[method] * 0.16
            grp = VGroup(*[
                Dot([x + ex, y + ey, 0], radius=0.012,
                    color=ramp(META, o, MODE), fill_opacity=0.30)
                for (x, y), o in zip(pts[:N_BG:3], order[:N_BG:3])])
            grp.add(*[
                Dot([x + ex, y + ey, 0], radius=0.028,
                    color=ramp(META, o, MODE), fill_opacity=1.0)
                for (x, y), o in zip(pts[N_BG:], order[N_BG:])])
            grp.add(Text(mlab, font="Inter", font_size=17,
                         color=C["muted"]).move_to([ex, ey - 1.45, 0]))
            echoes.add(grp)
        polar_lab = Text("POLAR", font="Inter", font_size=17,
                         color=C["muted"]).move_to([-2.3, -2.75, 0])
        cloud = VGroup(bg, tracked)
        self.play(
            cloud.animate(run_time=1.6).scale(0.62).move_to([-2.3, -0.1, 0]),
            FadeIn(echoes, run_time=1.6),
            FadeIn(polar_lab, run_time=1.6))
        self.wait(2.0)
        fence = Text(f"{META['provenance']} · @ {META['head']}",
                     font="JetBrains Mono", font_size=12,
                     color=C["muted"]).to_edge(DOWN, buff=0.18)
        self.play(FadeIn(fence, run_time=0.5))
        self.wait(1.2)
