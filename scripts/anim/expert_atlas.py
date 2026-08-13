"""ExpertAtlas flagship: routing canyon to causal constellation.

The 48 x 128 atlas is not used as an opening dashboard. Recorded routing
first appears as a token crossing receding layer planes, then as accumulated
demand terrain. The camera rises and the same values flatten into the
discrete atlas before the pooled-lens carriers and matched control explain
the set-level intervention.

Analytic: every expert position, routing flash, phase rate, carrier/control
identity, and solve count. Illustrative: perspective, bloom, trail decay,
camera motion, and interpolation between measured endpoints.

Render:
  .venv-anim/bin/manim -qh --format=mp4 \
    scripts/anim/expert_atlas.py ExpertAtlas
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    ImageMobject,
    LaggedStart,
    Line,
    MoveAlongPath,
    MovingCameraScene,
    Succession,
    Text,
    VMobject,
    VGroup,
    config,
    smooth,
)
from manim.constants import RESAMPLING_ALGORITHMS
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llmopt.figures.atlas_visuals import (
    block_field,
    deterministic_focus_pair,
    perspective_projection,
    phase_luminance,
    rail_fraction,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_theme import chrome, load_scene  # noqa: E402

ARR, META = load_scene("atlas")
MODE = os.environ.get("ANIM_MODE", "dark")
C = chrome(META, MODE)
config.background_color = C["surface"]

LAYERS, EXPERTS = 48, 128
RASTER_W, RASTER_H = 1600, 900
FRAME_W, FRAME_H = 14.22, 8.0
ATLAS_W = 12.85
ATLAS_H = ATLAS_W * LAYERS / EXPERTS
SANS = "Inter"
MONO = "JetBrains Mono"
PRE_LUM, DEC_LUM = phase_luminance(
    ARR["rate_prefill"], ARR["rate_decode"]
)
RMAX = float(max(ARR["rate_prefill"].max(), ARR["rate_decode"].max()))


def rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def mix(a: str, b: str, amount: float) -> tuple[int, int, int]:
    ar, br = np.array(rgb(a)), np.array(rgb(b))
    return tuple(
        int(value) for value in np.round(ar + (br - ar) * amount)
    )


def ramp_color(value: float) -> tuple[int, int, int]:
    stops = META["ramp"][MODE]
    position = np.clip(float(value), 0, 1) * (len(stops) - 1)
    lower = int(position)
    upper = min(lower + 1, len(stops) - 1)
    return mix(stops[lower], stops[upper], position - lower)


def terrain_image(luminance: np.ndarray, *, alpha: float = 1.0) -> np.ndarray:
    """Measured layer profiles rendered as a perspective luminous canyon."""
    background = rgb(C["surface"])
    base = Image.new("RGB", (RASTER_W, RASTER_H), background)
    glow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    crisp = Image.new("RGBA", base.size, (0, 0, 0, 0))
    gd, cd = ImageDraw.Draw(glow), ImageDraw.Draw(crisp)
    projected = perspective_projection(
        LAYERS, EXPERTS, width=RASTER_W, height=RASTER_H
    )

    for layer in range(LAYERS):
        depth = layer / (LAYERS - 1)
        lift = (18 + 132 * depth) * luminance[layer]
        points = projected[layer].copy()
        points[:, 1] -= lift
        rail = [(int(x), int(y)) for x, y in points]
        rail_alpha = int((30 + 70 * depth) * alpha)
        gd.line(rail, fill=(*rgb(META["ramp"][MODE][7]), rail_alpha),
                width=max(1, int(2 + 2 * depth)))
        cd.line(rail, fill=(*rgb(C["grid"]), int(96 + 76 * depth)), width=1)

        # Every expert at every layer: an earlier stride=2 for layers
        # <24 dropped half the measurements in half the frame, which
        # reads as "deep layers route more". Radius still grows with
        # depth -- that is perspective, applied to all cells equally.
        radius = 1 if depth < 0.45 else 2
        for expert in range(EXPERTS):
            value = float(luminance[layer, expert])
            if value < 0.025:
                continue
            x, y = points[expert]
            color = ramp_color(value)
            energy = int((45 + 190 * value) * alpha)
            gd.ellipse((x - 6, y - 6, x + 6, y + 6),
                       fill=(*color, min(150, energy)))
            cd.ellipse((x - radius, y - radius, x + radius, y + radius),
                       fill=(*color, energy))

    glow = glow.filter(ImageFilter.GaussianBlur(8))
    base = Image.alpha_composite(base.convert("RGBA"), glow)
    return np.asarray(Image.alpha_composite(base, crisp).convert("RGB"))


def terrain_mobject(luminance: np.ndarray) -> ImageMobject:
    image = ImageMobject(terrain_image(luminance))
    image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    image.width = FRAME_W
    image.move_to([0, 0, 0])
    return image


def atlas_mobject(luminance: np.ndarray) -> ImageMobject:
    image = ImageMobject(block_field(
        luminance,
        META["ramp"][MODE],
        C["surface"],
        cell=8,
        gutter=2,
    ))
    image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    image.width = ATLAS_W
    image.move_to([0, -0.15, 0])
    return image


def atlas_position(layer: int, rank: int) -> np.ndarray:
    pitch_x, pitch_y = ATLAS_W / EXPERTS, ATLAS_H / LAYERS
    return np.array([
        -ATLAS_W / 2 + (rank + 0.5) * pitch_x,
        ATLAS_H / 2 - (layer + 0.5) * pitch_y - 0.15,
        0,
    ])


def terrain_position(layer: int, rank: int, luminance: np.ndarray) -> np.ndarray:
    points = perspective_projection(
        LAYERS, EXPERTS, width=RASTER_W, height=RASTER_H
    )
    x, y = points[layer, rank]
    depth = layer / (LAYERS - 1)
    y -= (18 + 132 * depth) * luminance[layer, rank]
    return np.array([
        (x / RASTER_W - 0.5) * FRAME_W,
        (0.5 - y / RASTER_H) * FRAME_H,
        0,
    ])


def phase_label(word: str) -> Text:
    return Text(
        word.upper(), font=MONO, font_size=18,
        color=C["secondary"], weight="MEDIUM",
    ).to_corner(LEFT + DOWN, buff=0.42)


def number(text: str, size: int, color: str) -> Text:
    return Text(text, font=SANS, font_size=size, color=color,
                weight="LIGHT")


class ExpertAtlas(MovingCameraScene):
    def construct(self):
        frame = self.camera.frame
        zero = np.zeros_like(PRE_LUM)
        terrain = terrain_mobject(zero)
        self.add(terrain)

        # 1. A real token crosses 48 receding layer profiles. Recorded
        # scores are ASCENDING within topk, so the highest-scoring pick is
        # index -1; the path follows that expert and all eight real
        # selections flash at each layer as the cursor passes.
        trace = ARR["trace"]
        scores = ARR["trace_scores"]
        score_max = float(scores.max())
        path_points = [terrain_position(layer, int(trace[layer, -1]), zero)
                       for layer in range(LAYERS)]
        path = VMobject(stroke_color=C["primary"], stroke_width=1.1,
                       stroke_opacity=0.26)
        path.set_points_as_corners(path_points)
        cursor = Dot(path_points[0], radius=0.045,
                     color=C["primary"])
        cursor_glow = Dot(cursor.get_center(), radius=0.18,
                          color=C["primary"], fill_opacity=0.12,
                          stroke_width=0)
        flashes = []
        for layer in range(LAYERS):
            selected = VGroup()
            for rank, score in zip(trace[layer], scores[layer]):
                point = terrain_position(layer, int(rank), zero)
                # Channel separation: the ramp is reserved for ACCUMULATED
                # demand, so a transient gate score is carried by brightness
                # on one fixed hue. (Scoring it through the ramp landed in
                # inferno's blue-purple floor and read as a second palette.)
                strength = float(score) / score_max
                selected.add(Dot(
                    point, radius=0.025 + 0.018 * layer / (LAYERS - 1),
                    color=C["primary"], fill_opacity=0.32 + 0.63 * strength,
                    stroke_width=0,
                ))
            flashes.append(Succession(
                FadeIn(selected, scale=0.35, run_time=0.08),
                FadeOut(selected, scale=1.8, run_time=0.16),
            ))
        cursor_glow.add_updater(lambda mob: mob.move_to(cursor.get_center()))
        opening_center = np.mean(path_points[:12], axis=0)
        frame.set(width=FRAME_W * 0.48).move_to(opening_center)
        self.add(cursor_glow, cursor)
        self.play(
            Create(path),
            MoveAlongPath(cursor, path),
            LaggedStart(*flashes, lag_ratio=0.06),
            frame.animate.set(width=FRAME_W).move_to([0, 0, 0]),
            run_time=3.25,
            rate_func=smooth,
        )
        cursor_glow.clear_updaters()
        self.play(FadeOut(cursor), FadeOut(cursor_glow), FadeOut(path),
                  run_time=0.35)

        # 2. Real partial sums raise the terrain. No synthetic density.
        pre_label = phase_label("prefill")
        self.play(FadeIn(pre_label), run_time=0.35)
        current = terrain
        for snapshot in ARR["accum"]:
            luminance = np.clip(snapshot / RMAX, 0, 1) ** (1 / 2.2)
            nxt = terrain_mobject(luminance)
            nxt.set_opacity(0)
            self.add(nxt)
            self.play(
                nxt.animate.set_opacity(1),
                current.animate.set_opacity(0),
                run_time=0.32,
                rate_func=smooth,
            )
            self.remove(current)
            current = nxt
        self.wait(0.55)

        # 3. Rise overhead and flatten the exact endpoints into the atlas.
        atlas = atlas_mobject(PRE_LUM)
        self.play(
            FadeOut(current, shift=DOWN * 0.18),
            frame.animate.set(width=FRAME_W).move_to([0, 0, 0]),
            run_time=0.9,
            rate_func=smooth,
        )
        self.remove(current)
        self.play(FadeIn(atlas, shift=UP * 0.12), run_time=0.9,
                  rate_func=smooth)
        self.wait(0.45)

        # 4. Same cells, shared scale: prefill becomes decode.
        decode = atlas_mobject(DEC_LUM).set_opacity(0)
        decode_label = phase_label("decode")
        self.add(decode)
        self.play(
            decode.animate.set_opacity(1),
            atlas.animate.set_opacity(0),
            FadeOut(pre_label),
            FadeIn(decode_label),
            run_time=1.4,
        )
        self.remove(atlas)
        self.wait(0.6)

        # 5. Carriers lift as filled light, not diagram annotations.
        carriers = ARR["carriers"]
        carrier_glow = VGroup(*[
            Dot(atlas_position(int(layer), int(rank)), radius=0.075,
                color=C["primary"], fill_opacity=0.20, stroke_width=0)
            for layer, rank in carriers
        ])
        carrier_core = VGroup(*[
            Dot(atlas_position(int(layer), int(rank)), radius=0.024,
                color=C["primary"], fill_opacity=0.96, stroke_width=0)
            for layer, rank in carriers
        ])
        lens = Text(
            "POOLED-LENS CARRIERS", font=MONO, font_size=17,
            color=C["primary"], weight="MEDIUM",
        ).to_corner(LEFT + UP, buff=0.42)
        tilt = Text(
            "2.21× prefill share   →   0.84× decode share",
            font=SANS, font_size=23, color=C["secondary"],
        ).next_to(lens, DOWN, aligned_edge=LEFT, buff=0.12)
        self.play(
            LaggedStart(*[FadeIn(dot, scale=2.8) for dot in carrier_glow],
                        lag_ratio=0.004),
            LaggedStart(*[FadeIn(dot, scale=0.25) for dot in carrier_core],
                        lag_ratio=0.004),
            FadeIn(lens),
            FadeIn(tilt),
            run_time=1.5,
        )
        self.wait(0.85)

        # 6. One exact pair explains the matched-rank construction.
        (layer, carrier_rank), (_, control_rank) = deterministic_focus_pair(
            carriers, ARR["controls"]
        )
        carrier_point = atlas_position(layer, carrier_rank)
        control_point = atlas_position(layer, control_rank)
        self.play(FadeOut(lens), FadeOut(tilt), FadeOut(decode_label),
                  FadeOut(carrier_glow), run_time=0.35)
        self.play(
            frame.animate.scale(0.19).move_to(
                (carrier_point + control_point) / 2
            ),
            FadeOut(carrier_core),
            run_time=1.35,
        )
        focus = Dot(carrier_point, radius=0.028, color=C["primary"])
        destination = Dot(control_point, radius=0.028,
                          color=C["secondary"], fill_opacity=0.42)
        hop = Line(carrier_point, control_point, color=C["primary"],
                   stroke_width=0.8)
        pair_label = Text(
            f"same layer  ·  Δ demand rank {abs(control_rank-carrier_rank)}",
            font=MONO, font_size=15, color=C["secondary"],
        ).scale(0.19).next_to(hop, UP, buff=0.035)
        self.play(FadeIn(focus, scale=0.3), run_time=0.4)
        self.play(Create(hop), FadeIn(destination), FadeIn(pair_label),
                  run_time=0.85)
        self.wait(0.55)

        # 7. Pull out. The full sets appear as coordinated filled points.
        controls = ARR["controls"]
        control_core = VGroup(*[
            Dot(atlas_position(int(layer), int(rank)), radius=0.017,
                color=C["secondary"], fill_opacity=0.72, stroke_width=0)
            for layer, rank in controls
        ])
        self.play(
            FadeOut(focus), FadeOut(destination), FadeOut(hop),
            FadeOut(pair_label),
            frame.animate.scale(1 / 0.19).move_to([0, 0, 0]),
            run_time=1.2,
        )
        self.play(
            LaggedStart(*[FadeIn(dot, scale=0.3) for dot in carrier_core],
                        lag_ratio=0.003),
            LaggedStart(*[FadeIn(dot, scale=0.3) for dot in control_core],
                        lag_ratio=0.003),
            run_time=1.0,
        )
        self.wait(0.55)

        # 8. Clean causal outcome on black: three measured points, no table.
        self.play(
            FadeOut(decode), FadeOut(carrier_core), FadeOut(control_core),
            run_time=0.8,
        )
        # Rails run 0 -> the FULL denominator. An origin chosen near the
        # data inflates 244-vs-189 to a ~3.9x visual ratio against a true
        # 1.29x; rail_fraction is zero-anchored and carries a test.
        denominator = 360
        rail_x0, rail_w = -4.6, 8.2
        values = [("FULL", 189, C["muted"]),
                  ("MATCHED", 217, C["secondary"]),
                  ("CARRIERS", 244, C["primary"])]
        stems, nodes, labels = VGroup(), VGroup(), VGroup()
        for i, (name, value, color) in enumerate(values):
            y = 0.85 - i * 1.05
            stems.add(Line([rail_x0, y, 0], [rail_x0 + rail_w, y, 0],
                           color=C["grid"], stroke_width=3))
            end = rail_x0 + rail_w * rail_fraction(value, denominator)
            nodes.add(Line([rail_x0, y, 0], [end, y, 0],
                           color=color, stroke_width=3))
            labels.add(VGroup(
                Text(name, font=MONO, font_size=15, color=color)
                .move_to([rail_x0 - 0.15, y, 0], aligned_edge=RIGHT),
                number(str(value), 30, color)
                .next_to([end, y, 0], RIGHT, buff=0.22),
            ))
        ceiling = Text(f"of {denominator}", font=MONO, font_size=14,
                       color=C["muted"])
        ceiling.next_to([rail_x0 + rail_w, 0.85, 0], UP, buff=0.28)
        labels.add(ceiling)
        title = Text("DELETING 80 EXPERTS", font=MONO,
                     font_size=18, color=C["secondary"])
        title.to_edge(UP, buff=0.55)
        delta = number("+27", 50, C["primary"]).move_to([2.4, 2.35, 0])
        delta_note = Text("beyond demand-matched deletion", font=SANS,
                          font_size=19, color=C["secondary"])
        delta_note.next_to(delta, DOWN, buff=0.12)
        unit = Text(META["unit"], font=SANS, font_size=16,
                    color=C["muted"]).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(title), run_time=0.35)
        self.play(LaggedStart(*[FadeIn(stem) for stem in stems],
                              lag_ratio=0.16), run_time=0.8)
        self.play(LaggedStart(*[Create(node) for node in nodes],
                              lag_ratio=0.14),
                  LaggedStart(*[FadeIn(label) for label in labels],
                              lag_ratio=0.12), run_time=1.15)
        self.play(FadeIn(delta, scale=0.65), FadeIn(delta_note),
                  FadeIn(unit), run_time=0.65)
        self.wait(1.5)

        # 9. Readable, isolated receipt.
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.7)
        receipt = VGroup(
            Text("QWEN3-30B-A3B  ·  MATHGEN L1–3",
                 font=MONO, font_size=18,
                 color=C["secondary"]),
            Text("3 fresh paired seeds  ·  set-level masked intervention",
                 font=SANS, font_size=18, color=C["muted"]),
            Text("RESULTS  ·  EX-FRESH  ·  L22454",
                 font=MONO, font_size=15, color=C["muted"]),
        ).arrange(DOWN, buff=0.22)
        self.play(FadeIn(receipt), run_time=0.5)
        self.wait(2.4)
