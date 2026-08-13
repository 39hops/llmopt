"""ExpertAtlas ANIMATIC (low-res checkpoint, storyboard 2026-08-13 rev 2).

Arc, per the approved revision: prefill trace -> prefill atlas ->
decode atlas -> carrier constellation -> intervention.

Data: data/anim/atlas.npz (scratch/atlas_precompute.py). Every field
measured. The atlas is 48 layers x 128 experts, columns ordered by
per-layer POOLED demand rank -- the same ordering scratch/ex3_build.py
uses to define each carrier's +-8 control window, so a carrier-to-
control move is literally a <=8 column hop.

HONESTY CONTRACT (enforced by construction here):
  * cells are drawn as discrete blocks with hard gutters, nearest-
    neighbour resampled; no value is interpolated between experts
  * prefill and decode are both normalized by their own routing
    OPPORTUNITIES (picks / events, uniform = 8/128) and mapped through
    ONE fixed transfer function; column order is identical across phases
  * transient gate-score FLASHES (vector overlays, fully decaying) are
    a different visual channel from accumulated selection-frequency
    RESIDUE (the field image)
  * causal numbers are SET-LEVEL. One pairing is shown at close range
    to explain the construction; the +28/+55 labels appear only when
    complete sets are on screen
  * no post-deletion routing is shown -- those trajectories do not exist
  * provenance lives on a separate end card, never over live data

Render (animatic): .venv-anim/bin/manim -ql --format=mp4 \
    scripts/anim/expert_atlas.py ExpertAtlas
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, UP, Circle, FadeIn, FadeOut, ImageMobject,
                   MovingCameraScene, Rectangle, Text, VGroup, config)
from manim.constants import RESAMPLING_ALGORITHMS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_theme import chrome, load_scene  # noqa: E402

ARR, META = load_scene("atlas")
MODE = os.environ.get("ANIM_MODE", "dark")
C = chrome(META, MODE)
config.background_color = C["surface"]

L, E = 48, 128
CELL, GUT = 12, 2          # px at master scale: 128*(12+2) = 1792 <= 1920
ATLAS_W = 13.27            # scene units; pitch = 13.27/128 -> 14px at 1080p
ATLAS_H = ATLAS_W * L / E
# ONE fixed transfer function for BOTH phases, over the global max rate.
RMAX = float(max(ARR["rate_prefill"].max(), ARR["rate_decode"].max()))
GAMMA = 1 / 2.2


def _stops():
    hexes = META["ramp"][MODE]
    return np.array([[int(h.lstrip("#")[i:i + 2], 16) / 255
                      for i in (0, 2, 4)] for h in hexes])


STOPS = _stops()


def _lum(rate: np.ndarray) -> np.ndarray:
    return np.clip(rate / RMAX, 0, 1) ** GAMMA


def field_image(rate: np.ndarray) -> np.ndarray:
    """(48,128) rate -> RGB block image with hard gutters. Nearest
    neighbour only; nothing is interpolated between experts."""
    t = _lum(rate)
    idx = t * (len(STOPS) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(STOPS) - 1)
    f = (idx - lo)[..., None]
    rgb = STOPS[lo] * (1 - f) + STOPS[hi] * f
    rgb *= t[..., None]                      # dark cells stay dark
    pitch = CELL + GUT
    img = np.zeros((L * pitch, E * pitch, 3))
    bg = np.array([int(C["surface"].lstrip("#")[i:i + 2], 16) / 255
                   for i in (0, 2, 4)])
    img[:, :] = bg
    for r in range(L):
        for c in range(E):
            y, x = r * pitch, c * pitch
            img[y:y + CELL, x:x + CELL] = rgb[r, c]
    return (img * 255).astype(np.uint8)


def atlas_mob(rate: np.ndarray) -> ImageMobject:
    m = ImageMobject(field_image(rate))
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    m.width = ATLAS_W
    m.move_to([0, -0.35, 0])
    return m


def cell_pos(layer: int, col: int) -> np.ndarray:
    """Centre of one cell in scene units."""
    px, py = ATLAS_W / E, ATLAS_H / L
    return np.array([-ATLAS_W / 2 + (col + 0.5) * px,
                     ATLAS_H / 2 - (layer + 0.5) * py - 0.35, 0])


def label(text, size=22, color=None):
    return Text(text, font="Inter", font_size=size,
                color=color or C["secondary"])


def mono(text, size=19, color=None):
    return Text(text, font="JetBrains Mono", font_size=size,
                color=color or C["primary"])


class ExpertAtlas(MovingCameraScene):
    def construct(self):
        car, ctl = ARR["carriers"], ARR["controls"]
        trace, tscore = ARR["trace"], ARR["trace_scores"]
        accum = ARR["accum"]
        frame = self.camera.frame

        # ---- Beat 1 (0-4.2): one real prefill token through 48 layers.
        dark = atlas_mob(np.zeros((L, E)))
        self.add(dark)
        claim = label("One token. 48 layers. 8 experts each.", 30,
                      C["primary"]).to_edge(UP, buff=0.35)
        self.play(FadeIn(claim, run_time=0.6))
        smax = float(tscore.max())
        for l in range(L):
            # FLASH channel: transient, gate-score brightness, decays fully.
            fl = VGroup(*[
                Rectangle(width=ATLAS_W / E * 0.9, height=ATLAS_H / L * 0.9,
                          stroke_width=0, fill_opacity=0.25 + 0.75 * s / smax,
                          fill_color=C["primary"]).move_to(cell_pos(l, c))
                for c, s in zip(trace[l], tscore[l])])
            self.add(fl)
            self.play(FadeOut(fl, run_time=0.075), run_time=0.075)
        self.wait(0.3)

        # ---- Beat 2 (4.2-8.4): RESIDUE accumulates (real partial sums).
        self.play(FadeOut(claim, run_time=0.4))
        cur = dark
        for i in range(len(accum)):
            nxt = atlas_mob(accum[i])
            self.add(nxt)
            self.remove(cur)
            cur = nxt
            self.wait(0.34)

        # ---- Beat 3 (8.4-11.4): prefill atlas complete.
        pre = atlas_mob(ARR["rate_prefill"])
        self.add(pre)
        self.remove(cur)
        ph = mono("prefill").to_edge(DOWN, buff=0.3)
        self.play(FadeIn(ph, run_time=0.5))
        self.wait(1.6)

        # ---- Beat 4 (11.4-15): same columns, same transfer fn -> decode.
        dec = atlas_mob(ARR["rate_decode"])
        dec.set_opacity(0)
        self.add(dec)
        ph2 = mono("decode").to_edge(DOWN, buff=0.3)
        self.play(dec.animate.set_opacity(1), pre.animate.set_opacity(0),
                  FadeOut(ph), FadeIn(ph2), run_time=1.4)
        self.remove(pre)
        self.wait(1.2)

        # ---- Beat 5 (15-19): the 80-carrier constellation, both phases.
        rings = VGroup(*[
            Circle(radius=ATLAS_W / E * 0.62, stroke_width=1.6,
                   stroke_color=C["primary"], fill_opacity=0)
            .move_to(cell_pos(int(l), int(c))) for l, c in car])
        tilt = label("80 carriers — 2.21x bank share in prefill, "
                     "0.84x in decode", 20).to_edge(UP, buff=0.3)
        self.play(FadeIn(rings, lag_ratio=0.01, run_time=1.2),
                  FadeIn(tilt, run_time=0.8))
        self.wait(1.3)

        # ---- Beat 6 (19-24): ONE pairing at close range (deterministic:
        # first in sorted(inv_p)); explains the +-8 window construction.
        # Clear the constellation: at 0.16 zoom its 1.6 stroke reads as a
        # thick ring and buries the focus pair under its neighbours.
        self.play(FadeOut(tilt), FadeOut(ph2), FadeOut(rings), run_time=0.5)
        l0, c0 = int(car[0][0]), int(car[0][1])
        k0 = int(ctl[0][1])
        p_car, p_ctl = cell_pos(l0, c0), cell_pos(l0, k0)
        mid = (p_car + p_ctl) / 2
        self.play(frame.animate.scale(0.16).move_to(mid), run_time=1.5)
        cr = Circle(radius=ATLAS_W / E * 0.62, stroke_width=0.3,
                    stroke_color=C["primary"], fill_opacity=0).move_to(p_car)
        ck = Circle(radius=ATLAS_W / E * 0.62, stroke_width=0.3,
                    stroke_color=C["secondary"], fill_opacity=0).move_to(p_ctl)
        n1 = Text(f"carrier  L{l0} rank {c0}", font="JetBrains Mono",
                  font_size=19, color=C["primary"]).scale(0.16)
        n1.next_to(cr, UP, buff=0.04)
        n2 = Text(f"matched control  rank {k0}  (|Δrank| = {abs(c0-k0)} ≤ 8)",
                  font="JetBrains Mono", font_size=19,
                  color=C["secondary"]).scale(0.16)
        n2.next_to(ck, DOWN, buff=0.04)
        self.play(FadeIn(cr), FadeIn(n1), run_time=0.5)
        self.play(FadeIn(ck), FadeIn(n2), run_time=0.6)
        self.wait(1.1)

        # ---- Beat 7 (24-28): pull back; COMPLETE sets, no per-pair tracking.
        self.play(FadeOut(n1), FadeOut(n2), FadeOut(cr), FadeOut(ck),
                  frame.animate.scale(1 / 0.16).move_to([0, 0, 0]),
                  run_time=1.5)
        self.play(FadeIn(rings, lag_ratio=0.01, run_time=0.7))
        crings = VGroup(*[
            Circle(radius=ATLAS_W / E * 0.62, stroke_width=1.6,
                   stroke_color=C["secondary"], fill_opacity=0)
            .move_to(cell_pos(int(l), int(c))) for l, c in ctl])
        self.play(FadeIn(crings, lag_ratio=0.01, run_time=0.9))
        setlab = label("complete sets — causal evidence is set-level, "
                       "never per expert", 19).to_edge(UP, buff=0.3)
        self.play(FadeIn(setlab, run_time=0.6))
        self.wait(0.6)

        # ---- Beat 8 (28-32.5): the intervention, in the ledger's unit.
        rows = VGroup(
            mono("full model            189", 21, C["secondary"]),
            mono("matched control       217   (+28)", 21, C["secondary"]),
            mono("named carriers        244   (+55)", 21, C["primary"]),
            label("+27 beyond demand-matched deletion", 19),
        ).arrange(DOWN, aligned_edge=1 * np.array([-1, 0, 0]), buff=0.22)
        unit = label("pooled solves over 3 paired seeds, of 360", 17,
                     C["muted"])
        VGroup(rows, unit).arrange(DOWN, buff=0.3).move_to([0, -0.35, 0])
        shade = Rectangle(width=15, height=9, stroke_width=0,
                          fill_color=C["surface"], fill_opacity=0.82)
        self.play(FadeIn(shade), FadeOut(setlab), run_time=0.6)
        self.play(FadeIn(rows[0]), run_time=0.5)
        self.play(FadeIn(rows[1]), run_time=0.6)
        self.play(FadeIn(rows[2]), run_time=0.6)
        self.play(FadeIn(rows[3]), FadeIn(unit), run_time=0.6)
        self.wait(1.3)

        # ---- Beat 9: fade everything, then the separate receipt card.
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
        self.wait(0.25)
        fence = VGroup(
            mono(META["provenance"], 13, C["muted"]),
            mono(f"trace: {META['trace_rule']}", 13, C["muted"]),
            mono(f"@ {META['head']}", 13, C["muted"]),
        ).arrange(DOWN, buff=0.22).move_to([0, 0, 0])
        self.play(FadeIn(fence, run_time=0.5))
        self.wait(2.4)
