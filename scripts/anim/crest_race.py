"""CrestRace in the routing-canyon language: the paired-seed evidence
as zero-anchored rails on black, one seed at a time.

Data: docs/figures.json routing_crest (seed_pairs + control arms) via
data/anim/crest.npz — values derive from the npz; the +7 bar and seed-1234 scope quote the fence. Each seed's rail runs
0..120; the full-model fill draws first, then extends to the masked
value, the paired delta labelled. Unpaired zero controls sit in a
visually separate muted block. Receipt on its own end card.

Render (ANIM_MODE=light for the light variant):
  .venv-anim/bin/manim -qh --format=mp4 scripts/anim/crest_race.py CrestRace
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from manim import (DOWN, RIGHT, UP, Create, FadeIn, FadeOut, Line, MovingCameraScene, Text, VGroup, config)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llmopt.figures.atlas_visuals import rail_fraction

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_theme import chrome, load_scene  # noqa: E402

ARR, META = load_scene("crest")
MODE = os.environ.get("ANIM_MODE", "dark")
C = chrome(META, MODE)
config.background_color = C["surface"]

RAIL_X0, RAIL_W = -4.4, 7.6


class CrestRace(MovingCameraScene):
    def construct(self):
        denom = META["denominator"]
        title = Text(META["title"], font="Inter", weight="LIGHT",
                     font_size=28, color=C["primary"])
        self.play(FadeIn(title, run_time=0.7))
        self.wait(1.1)
        self.play(title.animate.scale(0.62).to_edge(UP, buff=0.45),
                  run_time=0.7)

        ceiling = Text(f"of {denom}", font="JetBrains Mono", font_size=14,
                       color=C["muted"])
        ceiling.next_to([RAIL_X0 + RAIL_W, 1.55, 0], UP, buff=0.24)
        self.play(FadeIn(ceiling, run_time=0.4))

        for k, (seed, full, mask) in enumerate(
                zip(ARR["seed"], ARR["full"], ARR["mask"])):
            y = 1.55 - k * 1.15
            rail = Line([RAIL_X0, y, 0], [RAIL_X0 + RAIL_W, y, 0],
                        color=C["grid"], stroke_width=3)
            e_full = RAIL_X0 + RAIL_W * rail_fraction(float(full), denom)
            e_mask = RAIL_X0 + RAIL_W * rail_fraction(float(mask), denom)
            fill_full = Line([RAIL_X0, y, 0], [e_full, y, 0],
                             color=C["secondary"], stroke_width=3)
            fill_ext = Line([e_full, y, 0], [e_mask, y, 0],
                            color=C["primary"], stroke_width=3)
            name = Text(f"SEED {int(seed)}", font="JetBrains Mono",
                        font_size=15, color=C["secondary"])
            name.move_to([RAIL_X0 - 0.15, y, 0], aligned_edge=RIGHT)
            vals = Text(f"{int(full)} → {int(mask)}", font="Inter",
                        font_size=24, color=C["primary"], weight="LIGHT")
            vals.next_to([e_mask, y, 0], RIGHT, buff=0.22)
            delta = Text(f"+{int(mask - full)}", font="JetBrains Mono",
                         font_size=15, color=C["primary"])
            delta.next_to(vals, RIGHT, buff=0.3)
            self.play(FadeIn(rail), FadeIn(name), run_time=0.3)
            self.play(Create(fill_full), run_time=0.5)
            self.play(Create(fill_ext), FadeIn(vals), FadeIn(delta),
                      run_time=0.6)
        self.wait(0.7)

        ctl = " · ".join(
            f"{lab} {int(v)}/{denom}"
            for lab, v in zip([str(x) for x in ARR["ctl_labels"]],
                              ARR["ctl_values"]))
        ctl_text = Text(f"unpaired controls (seed 1234):  {ctl}",
                        font="Inter", font_size=16, color=C["muted"]
                        ).move_to([0, -2.6, 0])
        mean_d = float((ARR["mask"] - ARR["full"]).mean())
        pooled = Text(f"pooled +{mean_d:.1f} v a +7 bar, three paired seeds",
                      font="Inter", font_size=20, color=C["secondary"]
                      ).move_to([0, -1.95, 0])
        self.play(FadeIn(pooled, run_time=0.5))
        self.play(FadeIn(ctl_text, run_time=0.5))
        self.wait(1.6)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.7)
        # Full fence, word-wrapped — a receipt must never truncate
        # its own scope fences.
        words, lines, cur = META["fence"].split(), [], ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if len(trial) > 72 and cur:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        lines.append(cur)
        lines.append(f"@ {META['head']}")
        fence = VGroup(*[
            Text(line, font="JetBrains Mono", font_size=13,
                 color=C["muted"]) for line in lines]).arrange(DOWN, buff=0.2)
        self.play(FadeIn(fence, run_time=0.5))
        self.wait(2.4)
