"""House theme bridge for manim scenes (anim venv, numpy only).

Loads the .npz artifacts written by scripts/anim_precompute.py (lab
venv) and exposes the house look without importing figstyle or
matplotlib here: the ramp arrives as 16 sampled hex stops per mode
inside the npz meta, and colors are interpolated from those stops.
No scene names a color directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "anim"


def register_house_fonts() -> list[str]:
    """Make the vendored Inter / JetBrains Mono visible to Pango.

    They are not system-installed, so a scene asking for font="Inter"
    silently falls back to a default face unless the .ttf is registered
    first. Every scene imports this module, so registering here is what
    makes the house type actually apply."""
    try:
        import manimpango
    except ImportError:            # non-anim venv (tests, precompute)
        return []
    return [f.name for f in sorted((ROOT / "assets" / "fonts").glob("*.ttf"))
            if manimpango.register_font(str(f.resolve()))]


register_house_fonts()


def load_scene(name: str):
    """-> (dict of numpy arrays, meta dict)."""
    z = np.load(DATA / f"{name}.npz", allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    arrays = {k: z[k] for k in z.files if k != "meta"}
    return arrays, meta


def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)],
                    dtype=np.float64) / 255.0


def ramp(meta: dict, t: float, mode: str = "dark") -> str:
    """Interpolate the sampled house ramp at t in [0,1] -> hex."""
    stops = meta["ramp"][mode]
    x = min(max(float(t), 0.0), 1.0) * (len(stops) - 1)
    i = int(x)
    if i >= len(stops) - 1:
        return stops[-1]
    a, b = _hex_to_rgb(stops[i]), _hex_to_rgb(stops[i + 1])
    c = a + (b - a) * (x - i)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in c)


def chrome(meta: dict, mode: str = "dark") -> dict:
    return meta["chrome"][mode]
