"""CrestRace: the routing-crest rails growing to their booked
values; the zero arms stay flat, honestly. Data-true: the numbers
come from docs/figures.json via data/anim/crest.npz
(scripts/anim_precompute.py --scene crest) — never typed here.

Render (from repo root):
  .venv-anim/bin/manim -qh --format=mp4 scripts/anim/crest_race.py CrestRace
  .venv-anim/bin/manim -ql --format=gif scripts/anim/crest_race.py CrestRace
"""
from __future__ import annotations

import sys
from pathlib import Path

from manim import (DOWN, LEFT, RIGHT, UP, FadeIn, Rectangle, Scene,
                   Text, VGroup, config, rate_functions)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_theme import chrome, load_scene, ramp  # noqa: E402

ARR, META = load_scene("crest")
MODE = "dark"
C = chrome(META, MODE)
config.background_color = C["surface"]

RAIL_W = 7.6
RAIL_H = 0.34
GAP = 0.92


class CrestRace(Scene):
    def construct(self):
        values = ARR["values"]
        labels = [str(x) for x in ARR["labels"]]
        denom = META["denominator"]

        title = Text(META["title"], font="Inter", weight="LIGHT",
                     font_size=30, color=C["primary"]).to_edge(UP, buff=0.5)
        self.play(FadeIn(title, run_time=0.8))

        rows = VGroup()
        anims = []
        counters = []
        top_y = 1.7
        for i, (v, lab) in enumerate(zip(values, labels)):
            y = top_y - i * GAP
            name = Text(lab, font="Inter", font_size=19,
                        color=C["secondary"])
            name.move_to([-6.6, y, 0], aligned_edge=LEFT)
            base = Rectangle(width=RAIL_W, height=RAIL_H,
                             fill_color=C["grid"], fill_opacity=1,
                             stroke_width=0)
            base.move_to([0.6, y - 0.32, 0])
            rows.add(name, base)
            frac = float(v) / denom
            count = Text(f"{int(v)}/{denom}", font="JetBrains Mono",
                         font_size=19,
                         color=C["primary"] if v else C["muted"])
            count.next_to(base, RIGHT, buff=0.25)
            counters.append(count)
            if frac > 0:
                fill = Rectangle(width=0.001, height=RAIL_H,
                                 fill_color=ramp(META, frac, MODE),
                                 fill_opacity=1, stroke_width=0)
                fill.move_to(base.get_left(), aligned_edge=LEFT)
                rows.add(fill)
                anims.append(fill.animate(
                    run_time=2.2,
                    rate_func=rate_functions.ease_out_cubic,
                ).stretch_to_fit_width(RAIL_W * frac, about_edge=LEFT))
        self.play(FadeIn(rows, lag_ratio=0.05, run_time=1.0))
        self.play(*anims, *[FadeIn(c) for c in counters])
        self.wait(1.0)

        fence = Text(f"{META['provenance']} · @ {META['head']}",
                     font="JetBrains Mono", font_size=14,
                     color=C["muted"]).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(fence, run_time=0.8))
        self.wait(1.4)
