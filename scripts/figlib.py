"""House chart helpers (figs/ instrumentation). One style, two forms:

- grouped_bars: solve-rate comparisons across bins/levels (the rarity
  curve shape) — counts printed on bars, n= under bins.
- lines: loss/metric curves over steps/epochs (training telemetry).

Rules baked in (dataviz doctrine): one axis, series color follows the
entity across every figure (COLORS is the fixed assignment), direct
labels over legends where possible, counts shown, muted grid. Output:
SVG (crisp anywhere) + optional PNG at 2x. Figures land in
figs/<YYYY-MM-DD>/<name>.svg — date dir per session.

    from figlib import grouped_bars
    grouped_bars("rarity", ["common", "mid", "rare", "unseen"],
                 {"CHAMPION": [(65, 69), ...], "TERNARY": [...]})
"""
from __future__ import annotations

import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

# fixed entity -> color (never cycled; extend, don't reorder)
COLORS = {
    "CHAMPION": "#4269d0", "fp32": "#4269d0",
    "TERNARY": "#efb118", "ternary": "#efb118",
    "DUO": "#3ca951", "duo": "#3ca951",
    "MERGED": "#ff725c", "merged": "#ff725c",
    "SERIES": "#a463f2",
}
_FALLBACK = ["#6cc5b0", "#97bbf5", "#9c6b4e", "#9498a0"]


def _color(name: str, i: int) -> str:
    for k, v in COLORS.items():
        if k.lower() in name.lower():
            return v
    return _FALLBACK[i % len(_FALLBACK)]


def _save(fig, name: str, png: bool = False) -> Path:
    day = datetime.date.today().isoformat()
    out = ROOT / "figs" / day
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.svg"
    fig.savefig(p, bbox_inches="tight")
    if png:
        fig.savefig(p.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")
    return p


def _style(ax, title: str, ylabel: str):
    ax.set_title(title, fontsize=12, loc="left", pad=12)
    ax.set_ylabel(ylabel, fontsize=10, color="#666")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#ccc")
    ax.tick_params(colors="#666", labelsize=9)
    ax.grid(axis="y", color="#eee", linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bars(name: str, bins: list[str],
                 series: dict[str, list[tuple[int, int]]],
                 title: str = "", png: bool = False) -> Path:
    """series: label -> [(solved, total) per bin]. Percent bars,
    solved/total printed on each bar, n= under each bin."""
    fig, ax = plt.subplots(figsize=(1.9 * len(bins) + 1.5, 3.6))
    w = 0.8 / len(series)
    for i, (label, vals) in enumerate(series.items()):
        xs = [b + i * w for b in range(len(bins))]
        pct = [100 * s / max(t, 1) for s, t in vals]
        ax.bar(xs, pct, width=w * 0.94, color=_color(label, i),
               label=label)
        for x, p, (s, t) in zip(xs, pct, vals):
            ax.annotate(f"{s}/{t}", (x, p), ha="center", va="bottom",
                        fontsize=8, color="#333")
    ax.set_xticks([b + 0.4 - w / 2 for b in range(len(bins))])
    ns = [t for _, t in next(iter(series.values()))]
    ax.set_xticklabels([f"{b}\nn={n}" for b, n in zip(bins, ns)],
                       fontsize=9)
    ax.set_ylim(0, 108)
    _style(ax, title or name, "% solved")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    return _save(fig, name, png)


def lines(name: str, xs: list, series: dict[str, list[float]],
          title: str = "", xlabel: str = "", ylabel: str = "",
          png: bool = False) -> Path:
    """series: label -> y values over shared xs. Direct end-labels,
    no legend box."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (label, ys) in enumerate(series.items()):
        c = _color(label, i)
        ax.plot(xs[:len(ys)], ys, color=c, linewidth=1.8)
        ax.annotate(label, (xs[len(ys) - 1], ys[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=9, color=c, va="center")
    _style(ax, title or name, ylabel)
    ax.set_xlabel(xlabel, fontsize=10, color="#666")
    ax.margins(x=0.12)
    return _save(fig, name, png)
