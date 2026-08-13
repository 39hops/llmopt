"""Render driver for the animation triplets (banked repair, 2026-08-13).

The anim build recipe used to live in README prose and shell history;
a zsh word-split once silently rendered nothing. This driver is the
recorded invocation: per scene it renders the 1080p60 master (dark +
light), cuts the poster at the SCENE-DECLARED time, and derives the
GIF from the master (MP4 is the design target; GIF is derived down).

  .venv/bin/python scripts/render_anim.py                # all scenes
  .venv/bin/python scripts/render_anim.py expert_atlas   # one scene
  .venv/bin/python scripts/render_anim.py --skip-light expert_atlas

SCENES is the shipping registry: tests/test_anim_assets.py asserts
every entry has its full asset set in docs/assets/anim/. Retired
scenes (crystal_rotation) are simply absent.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets" / "anim"
MANIM = ROOT / ".venv-anim" / "bin" / "manim"

# scene stem -> (SceneClass, poster time in seconds, has light variant)
SCENES = {
    "expert_atlas": ("ExpertAtlas", 12.7, False),
    "training_morph": ("TrainingMorph", 8.0, True),
    "crest_race": ("CrestRace", 8.5, True),
    "phase_portrait": ("PhasePortrait", 17.5, False),
}

GIF_VF = ("fps=9,scale=560:-1:flags=lanczos,"
          "split[a][b];[a]palettegen=max_colors=128[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=4")


def run(cmd: list[str], env: dict | None = None) -> None:
    import os
    full_env = {**os.environ, **(env or {})}
    r = subprocess.run(cmd, cwd=ROOT, env=full_env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(map(str, cmd))}\n{r.stderr[-2000:]}")


def build(stem: str, mode: str) -> None:
    cls, poster_t, _ = SCENES[stem]
    suffix = "" if mode == "dark" else "_light"
    run([str(MANIM), "-qh", "--format=mp4", "--disable_caching",
         f"scripts/anim/{stem}.py", cls],
        env={} if mode == "dark" else {"ANIM_MODE": "light"})
    master = ROOT / "media" / "videos" / stem / "1080p60" / f"{cls}.mp4"
    if not master.exists():
        sys.exit(f"FAILED: {master} missing after render")
    mp4 = OUT / f"{stem}{suffix}.mp4"
    mp4.write_bytes(master.read_bytes())
    run(["ffmpeg", "-y", "-ss", str(poster_t), "-i", str(mp4),
         "-frames:v", "1", "-update", "1",
         str(OUT / f"{stem}{suffix}_poster.png")])
    run(["ffmpeg", "-y", "-i", str(mp4), "-vf", GIF_VF,
         str(OUT / f"{stem}{suffix}.gif")])
    print(f"[anim] {stem}{suffix}: mp4 + poster@{poster_t}s + gif")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", choices=[*SCENES, []],
                    help="default: all")
    ap.add_argument("--skip-light", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for stem in (a.scenes or list(SCENES)):
        build(stem, "dark")
        if SCENES[stem][2] and not a.skip_light:
            build(stem, "light")


if __name__ == "__main__":
    main()
