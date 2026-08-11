"""lab.figures — matplotlib chart forms for ANALYSIS figures.

Division of labour, so the two renderers cannot drift:

  lab.figsvg    PUBLISHED figures — README, paper, anything with an
                audience. Reads docs/figures.json (booked numbers
                only), emits SVG directly, captures PNG through Chrome.
  lab.figures   ANALYSIS figures — this module. Ad-hoc plots while a
                result is still being worked out, and anything that
                needs real plotting machinery: thousands of scatter
                points, PCA projections, heatmaps, weight-space
                renders. Numbers come from wherever you are working.

If a figure is going in front of someone, it belongs in figures.json
and figsvg. If you are looking at data, use this.

Each form encodes the choices that are easy to get wrong, so a caller
picks the JOB and gets a correct chart rather than assembling one:

  gate_bars     capability comparison across arms (the 120 gate)
  curves        a measure over a shared x (loss, floor, k-sweep)
  ladder        one measure across an ordered axis, with the value at
                each point — the width/precision/size ladder shape
  scatter       two measures, one point per subject (predicted v
                measured); caps at three series by the all-pairs rule
  stat          a single number that IS the finding, with its scope

Shared rules, applied by construction rather than by remembering:
one y-axis ever; color follows the entity (`figstyle.ENTITY_SLOT`);
direct labels instead of a legend wherever the form allows it, which
also discharges the relief rule for the sub-3:1 light slots; counts
and n shown; grid recessive; every figure can carry a provenance
footer. Light and dark are rendered from the same call.

    from llmopt.lab.figures import gate_bars
    gate_bars("merge-space", {"s1": (12,120), "avg": (0,120)},
              title="Independent merges do not degrade — they die",
              subtitle="d64, 3080, n=1 per cell")
"""
from __future__ import annotations

from pathlib import Path

from llmopt.lab.figstyle import (CHROME, SERIES_ALLPAIRS, color, figure,
                                 footer, save)


def _both_modes(build, name: str, outdir=None) -> list[Path]:
    """Render light and dark from one description. Dark is drawn with
    its own palette steps, never an inverted light figure."""
    written = []
    for mode, suffix in (("light", ""), ("dark", "-dark")):
        fig = build(mode)
        written += save(fig, f"{name}{suffix}", outdir=outdir)
    return written


def gate_bars(name: str, arms: dict, title: str = "", subtitle: str = "",
              source: str = "", outdir=None):
    """arms: label -> (solved, total). Percent bars with solved/total
    printed on each — the count is the checksum, so it is always shown."""
    def build(mode):
        c = CHROME[mode]
        w = min(max(0.62 * len(arms) + 2.2, 5.0), 9.0)
        fig, ax = figure(title, subtitle, mode=mode, figsize=(w, 4.2))
        labels = list(arms)
        pct = [100 * arms[k][0] / max(arms[k][1], 1) for k in labels]
        cols = [color(k, i, mode) for i, k in enumerate(labels)]
        ax.bar(range(len(labels)), pct, width=0.62, color=cols)
        top = max(max(pct) * 1.22, 8)
        for i, (k, p) in enumerate(zip(labels, pct)):
            solved, total = arms[k]
            # A measured zero must not read as missing data: give it a
            # visible stub in its own series color so the eye sees a bar
            # that IS zero rather than a gap where a bar should be.
            if solved == 0:
                ax.bar([i], [top * 0.006], width=0.62, color=cols[i])
            ax.annotate(f"{solved}/{total}", (i, p), ha="center",
                        va="bottom", xytext=(0, 5),
                        textcoords="offset points", fontsize=9.5,
                        color=c["primary"], weight=600)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylabel("% of gate solved")
        ax.set_ylim(0, top)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        if source:
            footer(ax, source)
        return fig
    return _both_modes(build, name, outdir)


def curves(name: str, xs, series: dict, title: str = "", subtitle: str = "",
           xlabel: str = "", ylabel: str = "", source: str = "",
           logx: bool = False, outdir=None, annotate_last: bool = True):
    """series: label -> y values over shared xs. Direct end-labels, no
    legend box — identity never rests on color alone."""
    def build(mode):
        fig, ax = figure(title, subtitle, mode=mode)
        for i, (label, ys) in enumerate(series.items()):
            col = color(label, i, mode)
            ax.plot(xs[:len(ys)], ys, color=col, marker="o",
                    markersize=4.5, markeredgewidth=0)
        if annotate_last:
            # Nudge end labels apart when series finish close together:
            # two labels on top of each other is worse than one sitting a
            # little off its line. Place lowest-first, pushing each clear
            # of the one below it.
            lo = min(min(v) for v in series.values())
            hi = max(max(v) for v in series.values())
            gap = (hi - lo) * 0.055 or 1.0
            placed = None
            for label, ys in sorted(series.items(), key=lambda kv: kv[1][-1]):
                y = ys[-1] if placed is None else max(ys[-1], placed + gap)
                placed = y
                ax.annotate(label, (xs[len(ys) - 1], y), xytext=(9, 0),
                            textcoords="offset points", fontsize=9.5,
                            color=color(label, list(series).index(label), mode),
                            va="center", weight=600, annotation_clip=False)
        if logx:
            ax.set_xscale("log", base=2)
            ax.set_xticks(list(xs))
            ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.margins(x=0.16 if annotate_last else 0.04)
        if source:
            footer(ax, source)
        return fig
    return _both_modes(build, name, outdir)


def ladder(name: str, points: dict, title: str = "", subtitle: str = "",
           xlabel: str = "", ylabel: str = "", source: str = "",
           reference: tuple | None = None, entity: str = "series",
           fmt: str = "{:.4f}", outdir=None):
    """points: ordered x-label -> y. One series across an ordered axis,
    value printed at every point. `reference` is an optional
    (label, y) horizontal line — a frozen number the ladder is read
    against, drawn as a rule rather than a competing series."""
    def build(mode):
        c = CHROME[mode]
        fig, ax = figure(title, subtitle, mode=mode)
        labels = list(points)
        ys = [points[k] for k in labels]
        col = color(entity, 0, mode)
        if reference is not None:
            rlabel, ry = reference
            ax.axhline(ry, color=c["muted"], linewidth=1.1,
                       linestyle=(0, (4, 3)))
            ax.annotate(rlabel, (len(labels) - 1, ry), xytext=(0, 5),
                        textcoords="offset points", ha="right",
                        fontsize=8.5, color=c["muted"])
        ax.plot(range(len(labels)), ys, color=col, marker="o",
                markersize=6, markeredgewidth=0)
        for i, y in enumerate(ys):
            ax.annotate(fmt.format(y), (i, y), xytext=(0, 9),
                        textcoords="offset points", ha="center",
                        fontsize=9, color=c["primary"])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.margins(y=0.20)
        ax.grid(axis="x", visible=False)
        if source:
            footer(ax, source)
        return fig
    return _both_modes(build, name, outdir)


def scatter(name: str, series: dict, title: str = "", subtitle: str = "",
            xlabel: str = "", ylabel: str = "", source: str = "",
            identity: bool = False, outdir=None):
    """series: label -> (xs, ys). Every pair of colors is compared in a
    scatter, so this form CAPS at three series — past that the palette
    cannot clear the all-pairs floors. Fold the rest into 'Other' or
    facet into small multiples.

    identity=True draws the y=x rule (predicted-vs-measured figures).
    """
    if len(series) > SERIES_ALLPAIRS:
        raise ValueError(
            f"scatter caps at {SERIES_ALLPAIRS} series (all-pairs colour "
            f"floors); got {len(series)}. Fold into 'Other' or facet.")

    def build(mode):
        c = CHROME[mode]
        fig, ax = figure(title, subtitle, mode=mode, figsize=(5.6, 5.0))
        if identity:
            lo = min(min(v[0]) for v in series.values())
            hi = max(max(v[0]) for v in series.values())
            ax.plot([lo, hi], [lo, hi], color=c["muted"], linewidth=1.1,
                    linestyle=(0, (4, 3)), zorder=1)
            ax.annotate("y = x", (hi, hi), xytext=(-6, 6),
                        textcoords="offset points", ha="right",
                        fontsize=8.5, color=c["muted"])
        for i, (label, (xs, ys)) in enumerate(series.items()):
            ax.scatter(xs, ys, s=34, color=color(label, i, mode),
                       label=label, zorder=3, linewidths=0)
        if len(series) > 1:
            ax.legend(loc="upper left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if source:
            footer(ax, source)
        return fig
    return _both_modes(build, name, outdir)


def stat(name: str, value: str, label: str, detail: str = "",
         source: str = "", entity: str = "series", outdir=None):
    """A hero number: the finding IS the value, so no chart is drawn.
    Use when a plot would only decorate one measurement."""
    def build(mode):
        import matplotlib.pyplot as plt

        from llmopt.lab.figstyle import rc
        c = CHROME[mode]
        with plt.rc_context(rc(mode)):
            fig, ax = plt.subplots(figsize=(5.2, 2.5))
        ax.axis("off")
        ax.text(0, 0.72, value, fontsize=46, weight=600,
                color=color(entity, 0, mode), va="center", ha="left")
        ax.text(0, 0.34, label, fontsize=13, color=c["primary"],
                va="center", ha="left")
        if detail:
            ax.text(0, 0.13, detail, fontsize=9.5, color=c["muted"],
                    va="center", ha="left")
        if source:
            ax.annotate(source, xy=(0, 0), xycoords="axes fraction",
                        xytext=(0, -14), textcoords="offset points",
                        fontsize=8, color=c["muted"], family="monospace",
                        annotation_clip=False)
        return fig
    return _both_modes(build, name, outdir)
