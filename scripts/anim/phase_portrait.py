"""PhasePortrait: a model settling, drawn in true phase space.

The 2swap axes, no metaphor: x = each gate neuron's ANGLE in the
final-milestone whitened-PCA basis (position), y = its measured
per-step SPEED between adjacent milestones (velocity, log scale).
18 milestones from one real birth (PHASE-PORTRAIT-1,
scratch/birth19m_phase.py SEED=2); as training runs, the whole cloud
collapses toward the zero-velocity floor — the settling IS the
picture. House grammar: field = 2,900 neurons (PIL glow raster per
milestone, crossfaded between REAL states only), actors = 16
stratified tracked neurons, memory = decaying tails, geometry =
quiet log-speed ticks + the measured mean speed per milestone,
receipt = isolated end card.

Truth-language: milestone positions and speeds are measured; motion
between milestones is linear interpolation on the tracked actors and
crossfade on the field (no invented intermediate states). Speed is a
900-step finite difference, NOT Adam's exp_avg (that is a different
velocity, carried in the npz for a future panel). Portrait-only: no
capability claims.

Render: .venv-anim/bin/manim -qh --format=mp4 scripts/anim/phase_portrait.py PhasePortrait
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, RIGHT, UP, Dot, FadeIn, FadeOut, ImageMobject,
                   MovingCameraScene, RoundedRectangle, Text, TracedPath,
                   VGroup, config, smooth)
from manim.constants import RESAMPLING_ALGORITHMS
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_theme import chrome, load_scene  # noqa: E402

ARR, META = load_scene("phase")
MODE = os.environ.get("ANIM_MODE", "dark")
C = chrome(META, MODE)
config.background_color = C["surface"]

N_BG = 2900
LAYERS = 8
N = ARR["angle"].shape[1]
ROWS_PER_LAYER = N // LAYERS
RASTER_W, RASTER_H = 1600, 900
FRAME_W, FRAME_H = 14.22, 8.0
# speed axis: log10, measured floor/ceiling with a little air
V_LO, V_HI = -6.2, -3.2
MILES = list(range(1, ARR["angle"].shape[0]))   # vel defined from m1


def rgb(h):
    v = h.lstrip("#")
    return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))


def ramp_color(t):
    stops = META["ramp"][MODE]
    x = float(np.clip(t, 0, 1)) * (len(stops) - 1)
    lo, hi = int(x), min(int(x) + 1, len(stops) - 1)
    a, b = np.array(rgb(stops[lo])), np.array(rgb(stops[hi]))
    return tuple(int(v) for v in np.round(a + (b - a) * (x - lo)))


MAG_LO, MAG_HI = np.percentile(ARR["mag"], [1, 99])


def _xy(m, idx):
    """Phase-plane pixels: x = angle, y = log10 speed."""
    x = (ARR["angle"][m, idx] / np.pi * 0.5 + 0.5) * RASTER_W
    v = np.clip(np.log10(ARR["vel"][m, idx] + 1e-9), V_LO, V_HI)
    y = (1.0 - (v - V_LO) / (V_HI - V_LO)) * (RASTER_H * 0.86) \
        + RASTER_H * 0.07
    return np.stack([x, y], axis=1)


def _col01(m, idx):
    return (np.clip(ARR["mag"][m, idx], MAG_LO, MAG_HI) - MAG_LO) \
        / (MAG_HI - MAG_LO)


def field_image(m, idx):
    base = Image.new("RGB", (RASTER_W, RASTER_H), rgb(C["surface"]))
    glow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    crisp = Image.new("RGBA", base.size, (0, 0, 0, 0))
    gd, cd = ImageDraw.Draw(glow), ImageDraw.Draw(crisp)
    for (x, y), t in zip(_xy(m, idx), _col01(m, idx)):
        col = ramp_color(t)
        gd.ellipse((x - 6, y - 6, x + 6, y + 6),
                   fill=(*col, int(28 + 80 * t)))
        cd.ellipse((x - 1.6, y - 1.6, x + 1.6, y + 1.6),
                   fill=(*col, int(115 + 140 * t)))
    glow = glow.filter(ImageFilter.GaussianBlur(7))
    base = Image.alpha_composite(base.convert("RGBA"), glow)
    return np.asarray(Image.alpha_composite(base, crisp).convert("RGB"))


def field_mobject(m, idx):
    im = ImageMobject(field_image(m, idx))
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    im.width = FRAME_W
    im.move_to([0, 0, 0])
    return im


def scene_pos(m, i):
    x, y = _xy(m, np.array([i]))[0]
    return np.array([(x / RASTER_W - 0.5) * FRAME_W,
                     (0.5 - y / RASTER_H) * FRAME_H, 0])


def tracked_indices():
    picks = []
    final = ARR["mag"][-1]
    for layer in range(LAYERS):
        lo = layer * ROWS_PER_LAYER
        srt = np.argsort(final[lo:lo + ROWS_PER_LAYER])
        for q in (0.30, 0.70):
            picks.append(lo + srt[int(q * (len(srt) - 1))])
    return np.array(picks)


class PhasePortrait(MovingCameraScene):
    def construct(self):
        tr_idx = tracked_indices()
        rest = np.setdiff1d(np.arange(N), tr_idx)
        bg_idx = np.random.default_rng(0).permutation(rest)[:N_BG]
        steps = META["steps"]

        m0 = MILES[0]
        field = field_mobject(m0, bg_idx)
        tracked = VGroup(*[
            Dot(scene_pos(m0, int(i)), radius=0.055,
                color="#%02x%02x%02x" % ramp_color(
                    float(_col01(m0, np.array([i]))[0])),
                fill_opacity=1.0, stroke_color=C["primary"],
                stroke_width=1.4)
            for i in tr_idx])
        self.add(field, tracked)

        # Quiet geometry: log-speed ticks.
        def y_of(exp):
            return (0.5 - ((1.0 - (exp - V_LO) / (V_HI - V_LO)) * 0.86
                           + 0.07)) * FRAME_H
        ticks = VGroup(*[
            Text(f"1e{int(e)}", font="JetBrains Mono", font_size=14,
                 color=C["muted"]).move_to([-6.7, y_of(e), 0])
            for e in (-4, -5, -6)])
        axis = Text("speed  ‖ΔW‖ / step", font="Inter", font_size=15,
                    color=C["muted"]).move_to([-6.55, 3.4, 0])

        claim = Text("A model settling, in phase space.", font="Inter",
                     weight="LIGHT", font_size=34, color=C["primary"])
        scrim = RoundedRectangle(
            width=claim.width * 1.25, height=claim.height * 2.6,
            corner_radius=claim.height * 0.6, fill_color=C["surface"],
            fill_opacity=0.72, stroke_width=0).move_to(claim)
        self.play(FadeIn(scrim, run_time=0.6), FadeIn(claim, run_time=0.6))
        self.wait(1.0)
        self.play(FadeOut(claim), FadeOut(scrim),
                  FadeIn(ticks), FadeIn(axis), run_time=0.7)

        def chrome_for(m):
            lab = Text(f"step {steps[m]:,}", font="JetBrains Mono",
                       font_size=22, color=C["secondary"]
                       ).to_edge(DOWN, buff=0.4)
            mv = float(ARR["vel"][m].mean())
            st = Text(f"mean speed {mv:.1e}", font="JetBrains Mono",
                      font_size=18, color=C["primary"]
                      ).to_corner(UP + RIGHT, buff=0.4)
            return lab, st

        lab, st = chrome_for(m0)
        self.play(FadeIn(lab), FadeIn(st), run_time=0.4)
        trails = VGroup(*[
            TracedPath(d.get_center, stroke_width=2.0,
                       stroke_color=d.get_color(),
                       dissipating_time=0.55, stroke_opacity=[0.0, 0.5])
            for d in tracked])
        self.add(trails)

        cur = field
        for m in MILES[1:]:
            nxt = field_mobject(m, bg_idx)
            nxt.set_opacity(0)
            self.add(nxt)
            new_lab, new_st = chrome_for(m)
            self.play(
                nxt.animate.set_opacity(1),
                cur.animate.set_opacity(0),
                *[d.animate.move_to(scene_pos(m, int(i)))
                          .set_color("#%02x%02x%02x" % ramp_color(
                              float(_col01(m, np.array([i]))[0])))
                  for d, i in zip(tracked, tr_idx)],
                FadeOut(lab), FadeIn(new_lab),
                FadeOut(st), FadeIn(new_st),
                run_time=0.85, rate_func=smooth)
            lab, st = new_lab, new_st
        self.remove(trails)
        self.wait(2.2)               # settled cloud on the floor — poster

        # Receipt.
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.7)
        fence = VGroup(
            Text("PHASE-PORTRAIT-1 · 18 MILESTONES · 15,420 STEPS",
                 font="JetBrains Mono", font_size=17,
                 color=C["secondary"]),
            Text("x = angle, final-milestone whitened-PCA basis · "
                 "y = ‖ΔW‖/step (900-step finite difference)",
                 font="Inter", font_size=16, color=C["muted"]),
            Text("portrait-only: no capability claims",
                 font="Inter", font_size=16, color=C["muted"]),
            Text(f"{META['provenance'][:72]}", font="JetBrains Mono",
                 font_size=12, color=C["muted"]),
            Text(f"@ {META['head']}", font="JetBrains Mono",
                 font_size=12, color=C["muted"]),
        ).arrange(DOWN, buff=0.22)
        self.play(FadeIn(fence, run_time=0.5))
        self.wait(2.4)
