"""TrainingMorph in the routing-canyon language: one model crossing
training in one fixed frame, rendered as a luminous neuron field.

Field: 2,900 gate neurons as a rasterized glow field (PIL, same
treatment as expert_atlas terrain), x = angle in the FINAL
checkpoint's whitened-PCA basis, y = absolute row norm on one shared
scale. Actors: 16 tracked neurons (stratified: per layer, 30th/70th
percentile of within-layer final-magnitude rank). Memory: polylines
through their TRUE ep0/ep1/final positions. Outcome: mean row norm as
zero-anchored rails. Receipt: isolated end card.

Truth-language: checkpoint POSITIONS and tracked identities are data
(data/anim/morph.npz); tracked-dot motion between checkpoints is
linear interpolation (a rendering choice); the background field
CROSSFADES between real checkpoint states and never invents
intermediate positions. Mean row norm per stage is measured.

Render (ANIM_MODE=light for the light variant):
  .venv-anim/bin/manim -qh --format=mp4 scripts/anim/training_morph.py TrainingMorph
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, RIGHT, UP, Dot, FadeIn, FadeOut, ImageMobject,
                   LaggedStart, Line, MovingCameraScene, RoundedRectangle,
                   Text, TracedPath, VGroup, VMobject, config, smooth)
from manim.constants import RESAMPLING_ALGORITHMS
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llmopt.figures.atlas_visuals import rail_fraction

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_theme import chrome, load_scene  # noqa: E402

ARR, META = load_scene("morph")
MODE = os.environ.get("ANIM_MODE", "dark")
C = chrome(META, MODE)
config.background_color = C["surface"]

N_BG = 2900
LAYERS = 8
ROWS_PER_LAYER = len(ARR["final_mag"]) // LAYERS
RASTER_W, RASTER_H = 1600, 900
FRAME_W, FRAME_H = 14.22, 8.0

STAGES = ["ep0", "ep1", "final"]
STAGE_TEXT = {"ep0": "epoch 0 — born", "ep1": "epoch 1",
              "final": "final checkpoint"}
_all_mag = np.concatenate([ARR[f"{s}_mag"] for s in STAGES])
MAG_LO, MAG_HI = np.percentile(_all_mag, [1, 99])


def rgb(h: str) -> tuple[int, int, int]:
    v = h.lstrip("#")
    return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))


def ramp_color(t: float) -> tuple[int, int, int]:
    stops = META["ramp"][MODE]
    x = float(np.clip(t, 0, 1)) * (len(stops) - 1)
    lo = int(x)
    hi = min(lo + 1, len(stops) - 1)
    a, b = np.array(rgb(stops[lo])), np.array(rgb(stops[hi]))
    return tuple(int(v) for v in np.round(a + (b - a) * (x - lo)))


def _mag01(m: np.ndarray) -> np.ndarray:
    return (np.clip(m, MAG_LO, MAG_HI) - MAG_LO) / (MAG_HI - MAG_LO)


def _xy(stage: str, idx: np.ndarray) -> np.ndarray:
    """Pixel positions in the fixed frame: x = angle, y = |row|."""
    x = (ARR[f"{stage}_angle"][idx] / np.pi * 0.5 + 0.5)
    y = 1.0 - (0.08 + 0.84 * _mag01(ARR[f"{stage}_mag"][idx]))
    return np.stack([x * RASTER_W, y * RASTER_H], axis=1)


def field_image(stage: str, idx: np.ndarray) -> np.ndarray:
    """One real checkpoint state as a luminous point field."""
    base = Image.new("RGB", (RASTER_W, RASTER_H), rgb(C["surface"]))
    glow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    crisp = Image.new("RGBA", base.size, (0, 0, 0, 0))
    gd, cd = ImageDraw.Draw(glow), ImageDraw.Draw(crisp)
    pos = _xy(stage, idx)
    val = _mag01(ARR[f"{stage}_mag"][idx])
    for (x, y), t in zip(pos, val):
        col = ramp_color(t)
        gd.ellipse((x - 6, y - 6, x + 6, y + 6),
                   fill=(*col, int(30 + 90 * t)))
        cd.ellipse((x - 1.6, y - 1.6, x + 1.6, y + 1.6),
                   fill=(*col, int(120 + 135 * t)))
    glow = glow.filter(ImageFilter.GaussianBlur(7))
    base = Image.alpha_composite(base.convert("RGBA"), glow)
    return np.asarray(Image.alpha_composite(base, crisp).convert("RGB"))


def field_mobject(stage: str, idx: np.ndarray) -> ImageMobject:
    m = ImageMobject(field_image(stage, idx))
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    m.width = FRAME_W
    m.move_to([0, 0, 0])
    return m


def scene_pos(stage: str, i: int) -> np.ndarray:
    x, y = _xy(stage, np.array([i]))[0]
    return np.array([(x / RASTER_W - 0.5) * FRAME_W,
                     (0.5 - y / RASTER_H) * FRAME_H, 0])


def tracked_indices() -> np.ndarray:
    picks = []
    for layer in range(LAYERS):
        lo = layer * ROWS_PER_LAYER
        srt = np.argsort(ARR["final_mag"][lo:lo + ROWS_PER_LAYER])
        for q in (0.30, 0.70):
            picks.append(lo + srt[int(q * (len(srt) - 1))])
    return np.array(picks)


class TrainingMorph(MovingCameraScene):
    def construct(self):
        frame = self.camera.frame
        tr_idx = tracked_indices()
        rest = np.setdiff1d(np.arange(len(ARR["final_mag"])), tr_idx)
        bg_idx = np.random.default_rng(0).permutation(rest)[:N_BG]

        def tcol(stage, i):
            return "#%02x%02x%02x" % ramp_color(
                float(_mag01(ARR[f"{stage}_mag"][np.array([i])])[0]))

        field = field_mobject("ep0", bg_idx)
        tracked = VGroup(*[
            Dot(scene_pos("ep0", i), radius=0.055, color=tcol("ep0", i),
                fill_opacity=1.0, stroke_color=C["primary"],
                stroke_width=1.4)
            for i in tr_idx])
        self.add(field, tracked)

        # 1. Open inside the newborn field; claim on scrim; pull back.
        frame.set(width=FRAME_W * 0.42).move_to([-1.0, -0.8, 0])
        claim = Text("Training, in one fixed frame.", font="Inter",
                     weight="LIGHT", font_size=34, color=C["primary"])
        claim.scale(0.42).move_to(frame.get_center())
        scrim = RoundedRectangle(
            width=claim.width * 1.25, height=claim.height * 2.6,
            corner_radius=claim.height * 0.6, fill_color=C["surface"],
            fill_opacity=0.72, stroke_width=0).move_to(claim)
        self.play(FadeIn(scrim, run_time=0.6), FadeIn(claim, run_time=0.6))
        lab = Text(STAGE_TEXT["ep0"], font="Inter", font_size=24,
                   color=C["secondary"]).to_edge(DOWN, buff=0.4)
        self.play(FadeOut(claim), FadeOut(scrim),
                  frame.animate(run_time=2.6).set(width=FRAME_W)
                  .move_to([0, 0, 0]),
                  rate_func=smooth)
        self.play(FadeIn(lab, run_time=0.5))
        self.wait(0.5)

        # 2-3. ep0 -> ep1 -> final: field crossfades between REAL states;
        # tracked actors move with decaying tails, then their true
        # endpoints join into polyline memory.
        trails = VGroup(*[
            TracedPath(d.get_center, stroke_width=2.0,
                       stroke_color=d.get_color(),
                       dissipating_time=0.6, stroke_opacity=[0.0, 0.5])
            for d in tracked])
        self.add(trails)
        prev = {int(i): scene_pos("ep0", int(i)) for i in tr_idx}
        cur = field
        for stage in ("ep1", "final"):
            nxt = field_mobject(stage, bg_idx)
            nxt.set_opacity(0)
            self.add(nxt)
            new_lab = Text(STAGE_TEXT[stage], font="Inter", font_size=24,
                           color=C["secondary"]).to_edge(DOWN, buff=0.4)
            self.play(
                nxt.animate.set_opacity(1),
                cur.animate.set_opacity(0),
                *[d.animate.move_to(scene_pos(stage, int(i)))
                          .set_color(tcol(stage, int(i)))
                  for d, i in zip(tracked, tr_idx)],
                FadeOut(lab), FadeIn(new_lab),
                run_time=2.4, rate_func=smooth)
            self.remove(cur)
            cur = nxt
            segs = VGroup()
            for i in tr_idx:
                p = scene_pos(stage, int(i))
                seg = VMobject(stroke_width=1.4, stroke_opacity=0.4,
                               stroke_color=C["secondary"])
                seg.set_points_as_corners([prev[int(i)], p])
                segs.add(seg)
                prev[int(i)] = p
            self.play(FadeIn(segs, run_time=0.35))
            self.wait(0.9)
            lab = new_lab
        self.remove(trails)
        self.wait(1.2)

        # 4. Outcome on black: mean |row| per stage as zero-anchored rails.
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.7)
        title = Text("MEAN ROW NORM", font="JetBrains Mono", font_size=18,
                     color=C["secondary"]).to_edge(UP, buff=0.55)
        denom = MAG_HI          # shared scale ceiling, marked on screen
        rail_x0, rail_w = -4.6, 8.2
        rails, fills, labels = VGroup(), VGroup(), VGroup()
        for k, stage in enumerate(STAGES):
            y = 0.85 - k * 1.05
            v = META["mean_norm"][stage]
            rails.add(Line([rail_x0, y, 0], [rail_x0 + rail_w, y, 0],
                           color=C["grid"], stroke_width=3))
            end = rail_x0 + rail_w * rail_fraction(v, denom)
            col = C["primary"] if stage == "final" else C["secondary"]
            fills.add(Line([rail_x0, y, 0], [end, y, 0], color=col,
                           stroke_width=3))
            labels.add(VGroup(
                Text({"ep0": "EP0", "ep1": "EP1",
                      "final": "FINAL"}[stage],
                     font="JetBrains Mono", font_size=15, color=col)
                .move_to([rail_x0 - 0.15, y, 0], aligned_edge=RIGHT),
                Text(f"{v:.3f}", font="Inter", font_size=28, color=col,
                     weight="LIGHT").next_to([end, y, 0], RIGHT, buff=0.22)))
        ceiling = Text(f"scale p99 = {MAG_HI:.3f}", font="JetBrains Mono",
                       font_size=14, color=C["muted"])
        ceiling.next_to([rail_x0 + rail_w, 0.85, 0], UP, buff=0.28)
        note = Text("growth is small — the motion is angle churn",
                    font="Inter", font_size=19, color=C["secondary"]
                    ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(title, run_time=0.35))
        self.play(LaggedStart(*[FadeIn(r) for r in rails], lag_ratio=0.15),
                  run_time=0.7)
        self.play(LaggedStart(*[FadeIn(f) for f in fills], lag_ratio=0.15),
                  LaggedStart(*[FadeIn(la) for la in labels],
                              lag_ratio=0.12),
                  FadeIn(ceiling), run_time=1.0)
        self.play(FadeIn(note, run_time=0.5))
        self.wait(1.5)

        # 5. Isolated receipt.
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.7)
        fence = VGroup(
            Text("GALLERY19M_S1  ·  EP0 / EP1 / FINAL", font="JetBrains Mono",
                 font_size=17, color=C["secondary"]),
            Text("fixed final-checkpoint whitened-PCA basis · absolute row norm",
                 font="Inter", font_size=17, color=C["muted"]),
            Text(f"{META['provenance'][:76]}", font="JetBrains Mono",
                 font_size=12, color=C["muted"]),
            Text(f"@ {META['head']}", font="JetBrains Mono", font_size=12,
                 color=C["muted"]),
        ).arrange(DOWN, buff=0.22)
        self.play(FadeIn(fence, run_time=0.5))
        self.wait(2.4)
