"""Guards for the house figure style.

The palette is not a taste choice — it was computed. These tests pin
the properties that were verified with the dataviz validator, so a
future edit cannot quietly reintroduce a colorblind-unsafe pairing.

Validator results at adoption (2026-08-11), both modes, adjacent
pairlist: lightness band PASS, chroma floor PASS, CVD separation
worst adjacent ΔE 9.1 light / 8.4 dark (≥8 target), normal-vision
worst adjacent 19.6 / 19.3 (≥15 floor). The previous house palette
FAILED: red↔green ΔE 6.2 under protanopia.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from llmopt.lab import figstyle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# The exact validated order. Slot ORDER is the safety mechanism: only
# some orderings clear every adjacent gate. Changing this list means
# re-running scripts/validate_palette.js from the dataviz skill and
# updating both this list and the module docstring.
EXPECTED_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
EXPECTED_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                 "#d55181", "#008300", "#9085e9", "#e66767"]


def test_palette_order_is_the_validated_one():
    assert figstyle.SERIES_LIGHT == EXPECTED_LIGHT
    assert figstyle.SERIES_DARK == EXPECTED_DARK


def test_modes_are_separately_stepped():
    """Dark must be its own selected steps, never an inverted light
    palette — only the mode-invariant green may repeat."""
    shared = set(figstyle.SERIES_LIGHT) & set(figstyle.SERIES_DARK)
    assert shared == {"#008300"}, f"unexpected shared steps: {shared}"


def test_fonts_are_vendored():
    """Figures must render identically on the Mac, the 3080, and CI."""
    have = {p.name for p in (ROOT / "assets" / "fonts").glob("*.ttf")}
    assert {"Inter-Regular.ttf", "Inter-SemiBold.ttf",
            "JetBrainsMono-Regular.ttf"} <= have
    # licenses ship with the fonts (both SIL OFL)
    licenses = {p.name for p in (ROOT / "assets" / "fonts").glob("*LICENSE*")}
    assert len(licenses) >= 2, f"missing font licenses: {licenses}"


def test_entity_colors_are_stable_and_in_palette():
    """Color follows the entity across every figure; unknown names stay
    inside the validated slots rather than inventing a hue."""
    assert figstyle.color("champion") == figstyle.color("fp32")
    assert figstyle.color("champion") != figstyle.color("ternary")
    for mode, palette in (("light", EXPECTED_LIGHT), ("dark", EXPECTED_DARK)):
        for name in ["champion", "ternary", "merged", "totally-unknown-arm"]:
            assert figstyle.color(name, 0, mode) in palette


def test_sequential_is_one_hue_ordered():
    steps = figstyle.sequential(5)
    assert len(steps) == 5
    assert steps[0] != steps[-1]
    assert all(s in figstyle.SEQUENTIAL for s in steps)


def test_diverging_midpoint_is_neutral():
    """A diverging scale needs a neutral middle, or zero reads as a
    value. Gray has no hue channel: r == g == b."""
    for key in ("mid_light", "mid_dark"):
        h = figstyle.DIVERGING[key].lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        assert max(r, g, b) - min(r, g, b) <= 8, f"{key} is not neutral"


def test_status_never_collides_with_a_series_slot():
    """Status colors are reserved — a status hue must never be able to
    impersonate series N."""
    both = set(figstyle.SERIES_LIGHT) | set(figstyle.SERIES_DARK)
    assert not (set(figstyle.STATUS.values()) & both)


def test_scatter_refuses_more_series_than_allpairs_allows(tmp_path):
    """Every pair is compared in a scatter, and the palette only clears
    the all-pairs floors for the first three slots."""
    from llmopt.lab.figures import scatter
    four = {f"s{i}": ([1, 2], [1, 2]) for i in range(4)}
    with pytest.raises(ValueError, match="caps at 3"):
        scatter("x", four, outdir=tmp_path)


def test_forms_render_both_modes(tmp_path):
    from llmopt.lab.figures import gate_bars, curves, ladder, stat
    out = gate_bars("t", {"a": (12, 120), "zero": (0, 120)},
                    title="t", source="src", outdir=tmp_path)
    assert {p.name for p in out} >= {"t.png", "t-dark.png", "t.svg"}
    curves("c", [1, 2], {"champion": [1.0, 0.5]}, xlabel="x",
           source="src", outdir=tmp_path)
    ladder("l", {"d64": 0.4, "d128": 0.3}, reference=("H", 0.19),
           outdir=tmp_path)
    stat("s", "38/120", "gate solves", detail="n=1", outdir=tmp_path)
    for stem in ("c", "l", "s"):
        assert (tmp_path / f"{stem}.png").exists()
        assert (tmp_path / f"{stem}-dark.png").exists()


# ---- published figures (lab.figsvg) -----------------------------------

def test_published_figures_render_both_modes():
    """figsvg is pure Python: SVG must build with no browser and no
    matplotlib, so CI can verify published figures."""
    import json

    from llmopt.lab import figsvg
    names = [k for k in json.loads(figsvg.DATA.read_text())
             if not k.startswith("_")]
    assert names, "docs/figures.json has no figures"
    for name in names:
        for mode in ("light", "dark"):
            svg = figsvg.render(name, mode=mode)
            assert svg.startswith("<svg"), name
            assert svg.rstrip().endswith("</svg>"), name
            assert figsvg.CHROME[mode]["surface"] in svg


def test_every_published_figure_carries_its_fence():
    """The provenance strip is chart anatomy, not decoration: a figure
    that leaves the repo must still say what backs it."""
    import json

    from llmopt.lab import figsvg
    spec = json.loads(figsvg.DATA.read_text())
    for name, fig in spec.items():
        if name.startswith("_"):
            continue
        assert fig.get("fence"), f"{name} has no provenance fence"
        assert fig.get("title"), f"{name} has no title"
        assert figsvg._esc(fig["fence"]) in figsvg.render(name)


def test_gate_tracks_never_exceed_their_denominator():
    import json

    from llmopt.lab import figsvg
    for name, fig in json.loads(figsvg.DATA.read_text()).items():
        if name.startswith("_") or fig.get("kind") != "gate_track":
            continue
        denom = fig["denominator"]
        for arm in fig["arms"]:
            assert 0 <= arm["value"] <= denom, f"{name}/{arm['label']}"


def test_nice_ticks_are_round_numbers():
    """Axis labels a reader can compare without arithmetic."""
    from llmopt.lab.figsvg import _nice_ticks
    ticks, lo, hi = _nice_ticks(0.2899, 2.2529)
    assert len(ticks) >= 3
    step = ticks[1] - ticks[0]
    mantissa = step / 10 ** __import__("math").floor(
        __import__("math").log10(step))
    assert round(mantissa, 6) in (1.0, 2.0, 2.5, 5.0), step
    assert lo <= 0.2899 and hi >= 2.2529
