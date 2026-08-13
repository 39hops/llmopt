"""Pure visual transforms for the expert routing atlas.

The functions in this module convert measured routing rates into discrete
rendering geometry. They deliberately know nothing about Manim, timing, or
camera motion so the evidence-preserving transforms remain easy to test.
"""
from __future__ import annotations

import numpy as np


def _rgb(hex_color: str) -> np.ndarray:
    value = hex_color.lstrip("#")
    return np.array(
        [int(value[index:index + 2], 16) for index in (0, 2, 4)],
        dtype=np.float64,
    ) / 255.0


def phase_luminance(
    prefill: np.ndarray,
    decode: np.ndarray,
    *,
    gamma: float = 1 / 2.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Map both routing phases through one shared luminance scale."""
    maximum = float(max(np.max(prefill), np.max(decode)))
    if maximum <= 0:
        return np.zeros_like(prefill, dtype=float), np.zeros_like(
            decode, dtype=float
        )

    def convert(values: np.ndarray) -> np.ndarray:
        return np.clip(values / maximum, 0, 1) ** gamma

    return convert(prefill), convert(decode)


def perspective_projection(
    layers: int,
    experts: int,
    *,
    width: int,
    height: int,
    near_span: float = 0.34,
    far_span: float = 0.92,
) -> np.ndarray:
    """Project the discrete atlas into a centered receding layer canyon."""
    depth = np.linspace(0, 1, layers)
    span = width * (near_span + (far_span - near_span) * depth)
    unit_x = np.linspace(-0.5, 0.5, experts)
    points = np.empty((layers, experts, 2), dtype=float)
    points[:, :, 0] = width / 2 + span[:, None] * unit_x[None, :]
    points[:, :, 1] = height * (0.14 + 0.74 * depth[:, None])
    return points


def rail_fraction(value: float, denominator: float) -> float:
    """Fraction of the fixed gate a value fills, anchored at ZERO.

    Capability numbers in this lab are out of a fixed denominator, so
    the only honest bar geometry starts at 0 and ends at that
    denominator. An origin chosen near the data (say 170 for values of
    189/217/244) inflates the visual ratio to ~3.9x a true 1.29x, which
    is why this is a function with a test rather than a literal in a
    scene."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return float(np.clip(value / denominator, 0.0, 1.0))


def deterministic_focus_pair(
    carriers: np.ndarray, controls: np.ndarray
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Choose the lexicographically first stored carrier/control pair."""
    if len(carriers) != len(controls) or len(carriers) == 0:
        raise ValueError("carrier and control arrays must be non-empty pairs")
    index = min(
        range(len(carriers)),
        key=lambda item: tuple(int(v) for v in carriers[item]),
    )
    carrier = tuple(int(value) for value in carriers[index])
    control = tuple(int(value) for value in controls[index])
    if carrier[0] != control[0]:
        raise ValueError("matched controls must remain in the carrier layer")
    return carrier, control


def block_field(
    luminance: np.ndarray,
    ramp: list[str],
    background: str,
    *,
    cell: int = 12,
    gutter: int = 2,
) -> np.ndarray:
    """Render discrete luminance cells with background-colored gutters."""
    values = np.clip(np.asarray(luminance, dtype=float), 0, 1)
    stops = np.array([_rgb(color) for color in ramp])
    scaled = values * (len(stops) - 1)
    lower = np.floor(scaled).astype(int)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = (scaled - lower)[..., None]
    colors = stops[lower] * (1 - fraction) + stops[upper] * fraction
    colors *= values[..., None]

    layers, experts = values.shape
    pitch = cell + gutter
    image = np.empty((layers * pitch, experts * pitch, 3), dtype=float)
    image[:] = _rgb(background)
    for layer in range(layers):
        for expert in range(experts):
            y, x = layer * pitch, expert * pitch
            image[y:y + cell, x:x + cell] = colors[layer, expert]
    return np.round(image * 255).astype(np.uint8)
