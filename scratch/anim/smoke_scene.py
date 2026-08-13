"""Manim install smoke: one dot cloud fading in, house-dark surface.
Not a shipped scene — scenes for the program live in scripts/anim/."""
import numpy as np
from manim import Scene, Dot, VGroup, config

config.background_color = "#0f0f0e"


class SmokeScene(Scene):
    def construct(self):
        rng = np.random.default_rng(0)
        pts = rng.normal(0, 1.4, size=(300, 2))
        dots = VGroup(*[Dot([x, y, 0], radius=0.03, color="#fca50a")
                        for x, y in pts])
        self.play(dots.animate.set_opacity(0.9), run_time=1)
