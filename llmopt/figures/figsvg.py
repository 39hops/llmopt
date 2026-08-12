"""lab.figsvg — web-grade figures as hand-emitted SVG.

Why a second renderer beside lab/figures.py: matplotlib is the right
tool for a paper (vector, deterministic, no browser), but it fights
you on the details that make a figure read as current — rounded
data-ends, a real type scale, generous whitespace, tuned dark mode.
This module emits SVG directly, so those are free.

Both renderers read `docs/figures.json`, so a corrected number cannot
live in one and not the other. Colors come from lab.figstyle, which
holds the validated palette — nothing here invents a hue.

THE GATE TRACK. Every capability number in this lab is out of 120, a
fixed denominator. A bar chart is the wrong form for that: it draws
the part and leaves the whole implicit, so a measured zero renders as
nothing at all and reads as missing data. A track draws the whole gate
as a rail and fills the solved portion — 0/120 is then a visibly empty
rail, which is the finding, and every arm is comparable against the
same span without the eye doing arithmetic.

No dependencies. Output is a string; the caller writes it into a page
or a .svg file.
"""
from __future__ import annotations

import html
import json

from llmopt.figures.figstyle import CHROME, color

from llmopt.common.repo import repo_root

ROOT = repo_root()
DATA = ROOT / "docs" / "figures.json"

# Type scale (px at 1x). Mono is reserved for measured values and
# provenance — in a lab, numerals in monospace read as "recorded".
T_TITLE, T_SCOPE, T_LABEL, T_VALUE, T_FENCE = 19, 13, 13, 13, 11
SANS = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _nice_ticks(lo: float, hi: float, target: int = 5):
    """Ticks a reader can hold in their head: steps of 1, 2, 2.5, or 5
    times a power of ten. Axis labels like 2.45 / 1.86 / 1.27 make the
    reader do arithmetic to compare two points; round ones don't."""
    import math
    span = hi - lo
    if span <= 0:
        return [lo], lo, hi
    raw = span / max(target - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag),
                10 * mag)
    start = math.floor(lo / step) * step
    end = math.ceil(hi / step) * step
    n = int(round((end - start) / step))
    return [start + i * step for i in range(n + 1)], start, end


def _fmt(v: float, step: float) -> str:
    decimals = max(0, -int(__import__("math").floor(
        __import__("math").log10(step))) if step < 1 else 0)
    return f"{v:.{min(decimals + 1, 3)}f}" if step < 1 else f"{v:g}"


def load(name: str) -> dict:
    return json.loads(DATA.read_text())[name]


def _head(c, w, title, scope, pad_x, y):
    """Title block. Returns the y cursor below it."""
    out = []
    if title:
        # wrap by width: rough char metric is fine, this is layout not text
        limit = int((w - 2 * pad_x) / (T_TITLE * 0.52))
        words, line, lines = title.split(), "", []
        for word in words:
            trial = f"{line} {word}".strip()
            if len(trial) > limit and line:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
        for ln in lines:
            out.append(
                f'<text x="{pad_x}" y="{y}" font-family="{SANS}" '
                f'font-size="{T_TITLE}" font-weight="600" '
                f'fill="{c["primary"]}">{_esc(ln)}</text>')
            y += T_TITLE + 7
        y += 4
    if scope:
        limit = int((w - 2 * pad_x) / (T_SCOPE * 0.505))
        words, line, lines = scope.split(), "", []
        for word in words:
            trial = f"{line} {word}".strip()
            if len(trial) > limit and line:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
        for ln in lines:
            out.append(
                f'<text x="{pad_x}" y="{y}" font-family="{SANS}" '
                f'font-size="{T_SCOPE}" fill="{c["secondary"]}">'
                f'{_esc(ln)}</text>')
            y += T_SCOPE + 6
    return "\n".join(out), y + 12


def _fence(c, w, h, pad_x, text):
    """The signature element: provenance as part of the figure, not a
    caption someone can crop off."""
    if not text:
        return ""
    return (f'<line x1="{pad_x}" y1="{h - 40}" x2="{w - pad_x}" '
            f'y2="{h - 40}" stroke="{c["grid"]}" stroke-width="1"/>\n'
            f'<text x="{pad_x}" y="{h - 20}" font-family="{MONO}" '
            f'font-size="{T_FENCE}" fill="{c["muted"]}">{_esc(text)}</text>')


def gate_track(spec: dict, mode: str = "light", width: int = 880) -> str:
    """Fixed-denominator capability, drawn as filled rails."""
    c = CHROME[mode]
    arms = spec["arms"]
    denom = spec.get("denominator", 120)
    pad_x, row_h = 34, 46
    label_w = 200
    head, y = _head(c, width, spec.get("title"), spec.get("scope"), pad_x, 44)
    top = y
    height = int(top + len(arms) * row_h + 64)
    rail_x = pad_x + label_w
    rail_w = width - rail_x - pad_x - 62

    body = []
    for i, arm in enumerate(arms):
        cy = top + i * row_h
        col = color(arm.get("entity", arm["label"]), i, mode)
        frac = arm["value"] / denom
        fill_w = max(rail_w * frac, 0)
        body.append(
            f'<text x="{rail_x - 14}" y="{cy + 15}" text-anchor="end" '
            f'font-family="{SANS}" font-size="{T_LABEL}" '
            f'fill="{c["secondary"]}">{_esc(arm["label"])}</text>')
        # the whole gate, always drawn — this is what makes zero visible
        body.append(
            f'<rect x="{rail_x}" y="{cy + 3}" width="{rail_w}" height="16" '
            f'rx="8" fill="{c["grid"]}"/>')
        if fill_w > 0:
            body.append(
                f'<rect x="{rail_x}" y="{cy + 3}" width="{fill_w:.1f}" '
                f'height="16" rx="8" fill="{col}"/>')
        # value in mono, always shown: the count is the checksum
        body.append(
            f'<text x="{rail_x + rail_w + 12}" y="{cy + 15}" '
            f'font-family="{MONO}" font-size="{T_VALUE}" font-weight="500" '
            f'fill="{c["primary"] if arm["value"] else c["muted"]}">'
            f'{arm["value"]}<tspan fill="{c["muted"]}">/{denom}</tspan></text>')
    return _svg(width, height, c, head, body, spec.get("fence"), pad_x)


def curves(spec: dict, mode: str = "light", width: int = 880) -> str:
    """A measure over a shared x, with direct end labels."""
    c = CHROME[mode]
    xs, series = spec["x"], spec["series"]
    pad_x = 34
    head, top = _head(c, width, spec.get("title"), spec.get("scope"),
                      pad_x, 44)
    plot_h, height = 300, int(top + 300 + 118)
    left, right = pad_x + 54, width - pad_x - 74
    ys_all = [v for s in series for v in s["y"]]
    pad_y = (max(ys_all) - min(ys_all)) * 0.08
    ticks, lo, hi = _nice_ticks(min(ys_all) - pad_y, max(ys_all) + pad_y)
    tstep = ticks[1] - ticks[0] if len(ticks) > 1 else 1

    import math
    lx = [math.log2(x) for x in xs] if spec.get("logx") else list(map(float, xs))
    x0, x1 = min(lx), max(lx)

    def px(v):
        return left + (v - x0) / (x1 - x0) * (right - left)

    def py(v):
        return top + plot_h - (v - lo) / (hi - lo) * plot_h

    body = []
    for tv in ticks:  # horizontal rules only; x is a discrete sweep
        gy = py(tv)
        body.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{right}" '
                    f'y2="{gy:.1f}" stroke="{c["grid"]}" stroke-width="1"/>')
        body.append(f'<text x="{left - 12}" y="{gy + 4:.1f}" text-anchor="end" '
                    f'font-family="{MONO}" font-size="11" '
                    f'fill="{c["muted"]}">{_fmt(tv, tstep)}</text>')
    for i, x in enumerate(xs):
        body.append(f'<text x="{px(lx[i]):.1f}" y="{top + plot_h + 26}" '
                    f'text-anchor="middle" font-family="{MONO}" '
                    f'font-size="11" fill="{c["muted"]}">{x}</text>')

    # de-collide end labels: lowest first, each pushed clear of the last
    ends = sorted(range(len(series)), key=lambda i: series[i]["y"][-1])
    placed = None
    label_y = {}
    for i in ends:
        want = py(series[i]["y"][-1])
        yy = want if placed is None else min(want, placed - 17)
        placed = yy
        label_y[i] = yy

    for i, s in enumerate(series):
        col = color(s.get("entity", s["label"]), i, mode)
        pts = " ".join(f"{px(lx[j]):.1f},{py(v):.1f}"
                       for j, v in enumerate(s["y"]))
        body.append(f'<polyline points="{pts}" fill="none" stroke="{col}" '
                    f'stroke-width="2.5" stroke-linecap="round" '
                    f'stroke-linejoin="round"/>')
        for j, v in enumerate(s["y"]):
            body.append(f'<circle cx="{px(lx[j]):.1f}" cy="{py(v):.1f}" '
                        f'r="3.6" fill="{col}"/>')
        body.append(
            f'<text x="{right + 12}" y="{label_y[i] + 4:.1f}" '
            f'font-family="{SANS}" font-size="{T_LABEL}" font-weight="600" '
            f'fill="{col}">{_esc(s["label"])}</text>')
    if spec.get("xlabel"):
        body.append(f'<text x="{(left + right) / 2:.0f}" '
                    f'y="{top + plot_h + 52}" text-anchor="middle" '
                    f'font-family="{SANS}" font-size="12" '
                    f'fill="{c["secondary"]}">{_esc(spec["xlabel"])}</text>')
    if spec.get("ylabel"):
        body.append(f'<text transform="translate(16,{top + plot_h / 2:.0f}) '
                    f'rotate(-90)" text-anchor="middle" font-family="{SANS}" '
                    f'font-size="12" fill="{c["secondary"]}">'
                    f'{_esc(spec["ylabel"])}</text>')
    return _svg(width, height, c, head, body, spec.get("fence"), pad_x)


def ladder(spec: dict, mode: str = "light", width: int = 880) -> str:
    """One measure across an ordered axis, value printed at each point,
    with an optional frozen reference rule."""
    c = CHROME[mode]
    pts = spec["points"]
    pad_x = 34
    head, top = _head(c, width, spec.get("title"), spec.get("scope"),
                      pad_x, 44)
    plot_h, height = 280, int(top + 280 + 118)
    left, right = pad_x + 56, width - pad_x - 20
    vals = [p["value"] for p in pts]
    ref = spec.get("reference")
    v_lo = min(vals + ([ref["value"]] if ref else []))
    v_hi = max(vals)
    span = (v_hi - v_lo) or 1
    ticks, lo, hi = _nice_ticks(v_lo - span * 0.18, v_hi + span * 0.16)
    tstep = ticks[1] - ticks[0] if len(ticks) > 1 else 1

    def px(i):
        return left + i * (right - left) / max(len(pts) - 1, 1)

    def py(v):
        return top + plot_h - (v - lo) / (hi - lo) * plot_h

    body = []
    for tv in ticks:
        gy = py(tv)
        body.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{right}" '
                    f'y2="{gy:.1f}" stroke="{c["grid"]}" stroke-width="1"/>')
        body.append(f'<text x="{left - 12}" y="{gy + 4:.1f}" text-anchor="end" '
                    f'font-family="{MONO}" font-size="11" fill="{c["muted"]}">'
                    f'{_fmt(tv, tstep)}</text>')
    if ref:
        ry = py(ref["value"])
        body.append(f'<line x1="{left}" y1="{ry:.1f}" x2="{right}" '
                    f'y2="{ry:.1f}" stroke="{c["muted"]}" stroke-width="1.4" '
                    f'stroke-dasharray="5 4"/>')
        body.append(f'<text x="{right}" y="{ry - 9:.1f}" text-anchor="end" '
                    f'font-family="{MONO}" font-size="11" '
                    f'fill="{c["muted"]}">{_esc(ref["label"])}</text>')
    col = color(spec.get("entity", "series"), 0, mode)
    body.append('<polyline points="' +
                " ".join(f"{px(i):.1f},{py(p['value']):.1f}"
                         for i, p in enumerate(pts)) +
                f'" fill="none" stroke="{col}" stroke-width="2.5" '
                f'stroke-linecap="round" stroke-linejoin="round"/>')
    for i, p in enumerate(pts):
        body.append(f'<circle cx="{px(i):.1f}" cy="{py(p["value"]):.1f}" '
                    f'r="5" fill="{col}"/>')
        body.append(f'<text x="{px(i):.1f}" y="{py(p["value"]) - 15:.1f}" '
                    f'text-anchor="middle" font-family="{MONO}" '
                    f'font-size="12" fill="{c["primary"]}">{p["value"]}</text>')
        body.append(f'<text x="{px(i):.1f}" y="{top + plot_h + 26}" '
                    f'text-anchor="middle" font-family="{SANS}" '
                    f'font-size="{T_LABEL}" fill="{c["secondary"]}">'
                    f'{_esc(p["label"])}</text>')
    if spec.get("ylabel"):
        body.append(f'<text transform="translate(16,{top + plot_h / 2:.0f}) '
                    f'rotate(-90)" text-anchor="middle" font-family="{SANS}" '
                    f'font-size="12" fill="{c["secondary"]}">'
                    f'{_esc(spec["ylabel"])}</text>')
    return _svg(width, height, c, head, body, spec.get("fence"), pad_x)


def _svg(w, h, c, head, body, fence, pad_x) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img">\n'
        f'<rect width="{w}" height="{h}" fill="{c["surface"]}"/>\n'
        f'{head}\n' + "\n".join(body) + "\n" +
        _fence(c, w, h, pad_x, fence) + "\n</svg>")


def composition(spec: dict, mode: str = "light", width: int = 880) -> str:
    """One whole, split into labelled parts — a single stacked rail.

    Used for the ledger's own composition. A pie hides small slices and
    a grouped bar hides the whole; the point here is the PROPORTION of
    the record that is negative, so the whole has to stay visible and
    the small segments have to stay labelled.
    """
    c = CHROME[mode]
    parts = spec["parts"]
    total = sum(p["value"] for p in parts)
    pad_x = 34
    head, top = _head(c, width, spec.get("title"), spec.get("scope"),
                      pad_x, 44)
    bar_h, rail_w = 30, width - 2 * pad_x
    height = int(top + bar_h + 42 + len(parts) * 26 + 62)

    body, x = [], float(pad_x)
    for i, p in enumerate(parts):
        w = rail_w * p["value"] / total
        col = color(p.get("entity", p["label"]), i, mode)
        # 2px surface gap between segments so adjacent fills stay legible
        body.append(f'<rect x="{x:.1f}" y="{top}" width="{max(w - 2, 1):.1f}" '
                    f'height="{bar_h}" rx="4" fill="{col}"/>')
        x += w
    # legend rows: identity never rests on colour alone
    ly = top + bar_h + 40
    for i, p in enumerate(parts):
        col = color(p.get("entity", p["label"]), i, mode)
        pct = 100 * p["value"] / total
        body.append(f'<rect x="{pad_x}" y="{ly - 9}" width="10" height="10" '
                    f'rx="3" fill="{col}"/>')
        body.append(f'<text x="{pad_x + 20}" y="{ly}" font-family="{SANS}" '
                    f'font-size="{T_LABEL}" fill="{c["primary"]}">'
                    f'{_esc(p["label"])}</text>')
        body.append(f'<text x="{pad_x + 230}" y="{ly}" font-family="{MONO}" '
                    f'font-size="12" fill="{c["primary"]}">{p["value"]}'
                    f'<tspan fill="{c["muted"]}">  {pct:.0f}%</tspan></text>')
        if p.get("note"):
            body.append(f'<text x="{pad_x + 330}" y="{ly}" '
                        f'font-family="{SANS}" font-size="12" '
                        f'fill="{c["muted"]}">{_esc(p["note"])}</text>')
        ly += 26
    return _svg(width, height, c, head, body, spec.get("fence"), pad_x)


RENDERERS = {"gate_track": gate_track, "curves": curves, "ladder": ladder,
             "composition": composition}


def render(name: str, mode: str = "light", width: int = 880) -> str:
    spec = load(name)
    return RENDERERS[spec["kind"]](spec, mode=mode, width=width)
