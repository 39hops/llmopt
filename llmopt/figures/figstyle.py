"""lab.figstyle — the house figure style: validated palette, vendored
fonts, light and dark surfaces.

Two properties this module exists to guarantee:

1. **Reproducible rendering.** Fonts are vendored in `assets/fonts/`
   (Inter + JetBrains Mono, both SIL OFL), so a figure looks identical
   on the Mac, the 3080 box, and a CI runner. No reliance on whatever
   the host happens to have installed.

2. **Colors that were computed, not chosen.** The categorical order
   below is the dataviz reference palette, and it passes the full gate
   in BOTH modes on the adjacent pairlist: lightness band, chroma
   floor, CVD separation (worst adjacent ΔE 9.1 light / 8.4 dark
   against an ≥8 target), and normal-vision separation (19.6 / 19.3
   against a ≥15 floor). The previous house palette FAILED it — its
   red and green sat at ΔE 6.2 under protanopia, which is the classic
   confusion, and three of five slots were under 3:1 contrast.

   Slot ORDER is the safety mechanism, not decoration. Extend the list;
   never reorder it, and re-run the validator if you do.

Scatter, bubble, and small-multiple forms compare every pair rather
than adjacent ones, and the full eight cannot clear that bar. Use
`SERIES_ALLPAIRS` (the first three slots, validated all-pairs in both
modes) for those, and fold the rest into "Other" or facet.

Three light-mode slots sit below 3:1 on the light surface. That is the
documented relief case: those series carry visible direct labels, which
the house forms do by default.

    from llmopt.figures.figstyle import figure, save, color

    fig, ax = figure("Gate solves by width", subtitle="d64 ladder, n=1")
    ax.plot(xs, ys, color=color("champion"))
    save(fig, "width-ladder")          # writes light + dark PNG and SVG
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FONT_DIR = ROOT / "assets" / "fonts"

# ---------------------------------------------------------------- color

# Categorical slots, in validated order. Light and dark are the same
# eight hues stepped for their own surface — dark is SELECTED, never an
# automatic inversion of light.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"]
# forms that compare every pair (scatter, bubble, small multiples) cap
# at three: past that no ordering clears the all-pairs floors
SERIES_ALLPAIRS = 3

# Sequential = ONE hue, light to dark. Never a rainbow.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#2a78d6", "#256abf", "#184f95", "#0d366b"]
# Diverging = two poles that read as opposite + a NEUTRAL midpoint.
DIVERGING = {"low": "#2a78d6", "mid_light": "#f0efec",
             "mid_dark": "#383835", "high": "#e34948"}
# Status is reserved: never reused as "series 4", never color alone.
STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

CHROME = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b",
              "secondary": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7"},
    # Dark surface is near-black (validated 2026-08-11: all 8 series
    # pass CVD + contrast against #0f0f0e), grid/axis stepped down to
    # keep the same relief above the darker ground.
    "dark": {"surface": "#0f0f0e", "primary": "#ffffff",
             "secondary": "#c3c2b7", "muted": "#898781",
             "grid": "#232321", "axis": "#2e2e2b"},
}

# Entity -> slot INDEX. Color follows the entity across every figure in
# the lab, so a reader learns "blue is the champion line" once. A filter
# that drops a series must never repaint the survivors.
ENTITY_SLOT = {
    "champion": 0, "fp32": 0, "baseline": 0, "attention": 0, "full": 0,
    "ternary": 1, "grown": 1, "ssm": 1, "masked": 1,
    "duo": 2, "merged": 2, "shared": 2, "soup": 2,
    "crown": 3, "independent": 3,
    "random": 4, "control": 4,
    "exact": 5, "integer": 5,
    "series": 6, "probe": 6,
    "null": 7, "refuted": 7,
}

_FONTS_REGISTERED = False


def _register_fonts() -> bool:
    """Add the vendored fonts to matplotlib. Returns whether Inter is
    usable; callers fall back to the stack in rcParams if not."""
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return "Inter" in {f.name for f in font_manager.fontManager.ttflist}
    for ttf in sorted(FONT_DIR.glob("*.ttf")):
        try:
            font_manager.fontManager.addfont(str(ttf))
        except Exception:  # a missing/corrupt font must not kill a run
            pass
    _FONTS_REGISTERED = True
    return "Inter" in {f.name for f in font_manager.fontManager.ttflist}


def color(entity: str, index: int = 0, mode: str = "light") -> str:
    """Color for a named entity, stable across every house figure.

    Unknown names fall through to `index` in slot order, so ad-hoc
    series stay inside the validated palette instead of inventing hues.
    """
    slots = SERIES_DARK if mode == "dark" else SERIES_LIGHT
    key = entity.lower()
    for name, slot in ENTITY_SLOT.items():
        if name in key:
            return slots[slot]
    return slots[index % len(slots)]


def sequential(n: int) -> list[str]:
    """n evenly spread steps of the single-hue sequential ramp."""
    if n <= 1:
        return [SEQUENTIAL[4]]
    step = (len(SEQUENTIAL) - 1) / (n - 1)
    return [SEQUENTIAL[round(i * step)] for i in range(n)]


# ------------------------------------------------------------ rcParams


def rc(mode: str = "light") -> dict:
    """House rcParams. Recessive chrome, thin marks, real typography."""
    has_inter = _register_fonts()
    c = CHROME[mode]
    sans = (["Inter"] if has_inter else []) + [
        "IBM Plex Sans", "Helvetica Neue", "Helvetica", "DejaVu Sans"]
    mono = ["JetBrains Mono", "IBM Plex Mono", "Menlo", "DejaVu Sans Mono"]
    return {
        "figure.facecolor": c["surface"],
        "figure.dpi": 100,
        "savefig.facecolor": c["surface"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.34,  # room for the provenance footer
        "axes.facecolor": c["surface"],
        "axes.edgecolor": c["axis"],
        "axes.labelcolor": c["secondary"],
        "axes.titlecolor": c["primary"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(
            color=SERIES_DARK if mode == "dark" else SERIES_LIGHT),
        "grid.color": c["grid"],
        "grid.linewidth": 0.7,
        "xtick.color": c["muted"],
        "ytick.color": c["muted"],
        "xtick.labelcolor": c["secondary"],
        "ytick.labelcolor": c["secondary"],
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "text.color": c["primary"],
        "font.family": "sans-serif",
        "font.sans-serif": sans,
        "font.monospace": mono,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0,
        # Math notation set in the text face so $\|w_i\|_2$ sits in
        # the same voice as the words around it (DejaVu math clashes).
        "mathtext.fontset": "custom",
        "mathtext.rm": sans[0],
        "mathtext.it": sans[0],
        "mathtext.bf": sans[0] + ":medium",
        "svg.fonttype": "path",  # SVG renders identically without the font
    }


# -------------------------------------------------------------- figures


def figure(title: str = "", subtitle: str = "", mode: str = "light",
           figsize: tuple[float, float] = (7.2, 4.0), **kw):
    """A styled figure + axes with house title furniture.

    Title is left-aligned and carries the claim; the subtitle carries
    the scope (device, seeds, diet) — the fences travel WITH the figure,
    the same rule the ledger uses.
    """
    # rcParams are applied GLOBALLY, not through rc_context: a context
    # manager exits before the caller draws anything, so every title,
    # label, and annotation added afterwards would silently fall back
    # to matplotlib's defaults. House figures are built one at a time
    # and all want this style, so setting it globally is the honest
    # shape.
    plt.rcParams.update(rc(mode))
    fig, ax = plt.subplots(figsize=figsize, **kw)
    c = CHROME[mode]
    # Title stack sits above the axes in AXES coordinates, so it stays
    # left-aligned to the plot area rather than to the figure edge.
    y = 1.02
    if subtitle:
        ax.text(0.0, y, subtitle, ha="left", va="bottom", fontsize=9.5,
                color=c["muted"], transform=ax.transAxes)
        y += 0.075
    if title:
        ax.text(0.0, y, title, ha="left", va="bottom", fontsize=13.5,
                weight=600, color=c["primary"],
                transform=ax.transAxes)
    ax._house_mode = mode  # noqa: SLF001 — save() and footer() read it
    return fig, ax


def footer(ax, text: str) -> None:
    """Provenance line under the plot: the sha, the verdict, the fence.
    A figure that leaves the repo should still say where it came from."""
    mode = getattr(ax, "_house_mode", "light")
    # drop below the x-label when there is one, or it collides
    dy = -58 if ax.get_xlabel() else -38
    ax.annotate(text, xy=(0, 0), xycoords="axes fraction",
                xytext=(0, dy), textcoords="offset points",
                fontsize=8, color=CHROME[mode]["muted"],
                family="monospace", annotation_clip=False)


def save(fig, name: str, outdir: Path | str | None = None,
         png: bool = True, svg: bool = True, dpi: int = 220) -> list[Path]:
    """Write the figure. PNG for README/LinkedIn, SVG for the paper."""
    out = Path(outdir) if outdir else ROOT / "docs" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    if png:
        p = out / f"{name}.png"
        fig.savefig(p, dpi=dpi)
        written.append(p)
    if svg:
        p = out / f"{name}.svg"
        fig.savefig(p)
        written.append(p)
    plt.close(fig)
    for p in written:
        print(f"[fig] {p} ({p.stat().st_size // 1024} KB)")
    return written
