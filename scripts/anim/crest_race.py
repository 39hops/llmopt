"""CrestRace: three paired full→mask comparisons, one per seed,
appearing sequentially — the paired-seed design IS the evidence, so
the scene shows each pair as a slope (full model dot rising to the
masked dot, delta labeled). The unpaired zero controls sit in a
visually separate gray strip. Data-true: the numbers come from
docs/figures.json (seed_pairs + control arms) via data/anim/crest.npz
(scripts/anim_precompute.py --scene crest) — never typed here.

Render (from repo root; ANIM_MODE=light for the light variant):
  .venv-anim/bin/manim -qh --format=mp4 scripts/anim/crest_race.py CrestRace
  .venv-anim/bin/manim -ql --format=gif scripts/anim/crest_race.py CrestRace
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from manim import (DOWN, UP, Create, Dot, FadeIn, Line, Scene, Text,
                   VGroup, config)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_theme import chrome, load_scene, ramp  # noqa: E402

ARR, META = load_scene("crest")
MODE = os.environ.get("ANIM_MODE", "dark")
C = chrome(META, MODE)
config.background_color = C["surface"]

# Solve counts map onto this y band (60..90 of 120 — the pairs' range;
# the axis ticks say so honestly).
Y_LO, Y_HI = 60.0, 90.0
Y_BOT, Y_TOP = -0.9, 1.9
COL_X = (-3.6, -0.4, 2.8)   # one column per seed
HALF = 0.9                  # horizontal half-width of a slope


def _y(v: float) -> float:
    return Y_BOT + (float(v) - Y_LO) / (Y_HI - Y_LO) * (Y_TOP - Y_BOT)


class CrestRace(Scene):
    def construct(self):
        denom = META["denominator"]
        title = Text(META["title"], font="Inter", weight="LIGHT",
                     font_size=30, color=C["primary"]).to_edge(UP, buff=0.45)
        self.play(FadeIn(title, run_time=0.6))

        # y ticks so the zoomed band is explicit
        ticks = VGroup(*[
            Text(f"{int(v)}", font="JetBrains Mono", font_size=15,
                 color=C["muted"]).move_to([-6.4, _y(v), 0])
            for v in (60, 70, 80, 90)])
        self.play(FadeIn(ticks, run_time=0.5))

        for cx, seed, full, mask in zip(COL_X, ARR["seed"],
                                        ARR["full"], ARR["mask"]):
            d = int(mask - full)
            p_full = [cx - HALF, _y(full), 0]
            p_mask = [cx + HALF, _y(mask), 0]
            dot_f = Dot(p_full, radius=0.07, color=C["secondary"])
            dot_m = Dot(p_mask, radius=0.09,
                        color=ramp(META, float(mask) / denom, MODE))
            slope = Line(p_full, p_mask, stroke_width=3.5,
                         color=C["secondary"])
            v_f = Text(f"full {int(full)}", font="JetBrains Mono",
                       font_size=17, color=C["secondary"]
                       ).next_to(dot_f, DOWN, buff=0.18)
            v_m = Text(f"mask {int(mask)}", font="JetBrains Mono",
                       font_size=17, color=C["primary"]
                       ).next_to(dot_m, UP, buff=0.18)
            dl = Text(f"seed {int(seed)}  ({d:+d})", font="Inter",
                      font_size=18, color=C["primary"]
                      ).move_to([cx, Y_BOT - 0.55, 0])
            self.play(FadeIn(dot_f), FadeIn(v_f), run_time=0.4)
            self.play(Create(slope), FadeIn(dot_m), FadeIn(v_m),
                      FadeIn(dl), run_time=0.7)

        ctl = " · ".join(
            f"{lab} {int(v)}/{denom}"
            for lab, v in zip([str(x) for x in ARR["ctl_labels"]],
                              ARR["ctl_values"]))
        ctl_text = Text(f"unpaired controls (seed 1234):  {ctl}",
                        font="Inter", font_size=16, color=C["muted"]
                        ).move_to([0, Y_BOT - 1.25, 0])
        self.play(FadeIn(ctl_text, run_time=0.6))
        self.wait(0.5)

        fence = Text(f"{META['provenance']} · @ {META['head']}",
                     font="JetBrains Mono", font_size=13,
                     color=C["muted"]).to_edge(DOWN, buff=0.22)
        self.play(FadeIn(fence, run_time=0.5))
        self.wait(0.9)
