"""Publication PDF figures for the entropy-bound packing paper.

House style only: llmopt.lab.figstyle.rc("light"), validated palette,
vendored fonts, provenance footer on every panel (the figsvg fence-strip
discipline). Light mode; PDF out to docs/paper/figs/.

Every number here is transcribed from a booked RESULTS.md verdict, named
in the figure's footer with its entry line. Nothing is derived, fitted, or
interpolated. Scatter forms stay inside the three all-pairs-validated
palette slots (figstyle.SERIES_ALLPAIRS) and separate further classes by
marker shape, never by a fourth hue.

    .venv/bin/python scratch/paper_figs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from llmopt.lab import figstyle
from llmopt.lab.figstyle import CHROME, SERIES_LIGHT, figure, footer

OUT = Path(__file__).resolve().parents[1] / "docs" / "paper" / "figs"
MUTED = CHROME["light"]["muted"]
SEC = CHROME["light"]["secondary"]
C0, C1, C2 = SERIES_LIGHT[:figstyle.SERIES_ALLPAIRS]


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.pdf"
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig] {p} ({p.stat().st_size // 1024} KB)")
    return p


# --------------------------------------------------------------- fig 1
# PACKED CRYSTAL C0+C1 (RESULTS.md L10406): raw packed 4.94 (L4d56) /
#   5.06 (d64h8) bits/wt; gates 56->55 and 58->58.
# C1 AT n=3 (L11232): d64h8 +2/-3 at seeds 2/3; L4d56 -5/-5.
# PACKED CRYSTAL C3 (L10458): d64h8 control 58; 5-bit rtn 59 gptq 58
#   awq 58 hqq 57; 3-bit rtn 57 gptq 55 awq 56 hqq 60.
# PACKED CRYSTAL C5 (L10529): matryoshka tiers, payload bits ~5,
#   eighth 42 (fp 48), half 53 (fp 57), dense 52 (fp 57).

def fig_packing_curve():
    fig, ax = figure(
        "At ~5 bits/weight the pack is free — where the crystal is not fragile",
        subtitle="gate delta vs the matched fp control, 120 problems; "
                 "sigma ~ 3.5 solves (shaded); x jittered +-0.06 within a class",
        figsize=(7.8, 4.6))

    ax.axhspan(-3.5, 3.5, color=CHROME["light"]["grid"], zorder=0)
    ax.axhline(0, color=CHROME["light"]["axis"], lw=0.9, zorder=1)

    def jit(x, n, w=0.06):
        return [x + w * (i - (n - 1) / 2) for i in range(n)]

    # sigma-law pack, n=3 per architecture
    ax.scatter(jit(5.06, 3), [0, 2, -3], s=46, color=C0, zorder=3,
               label="sigma-law pack, d64h8 (n=3)")
    ax.scatter(jit(4.94, 3), [-1, -5, -5], s=46, color=C0, marker="s",
               zorder=3, label="sigma-law pack, L4d56 (n=3)")
    # calibrated / RTN baselines on d64h8, matched bits
    ax.scatter(jit(5.0, 4), [1, 0, 0, -1], s=40, color=C1, marker="^",
               zorder=3, label="rtn / gptq / awq / hqq (matched bits)")
    ax.scatter(jit(3.0, 4), [-1, -3, -2, 2], s=40, color=C1, marker="^",
               facecolors="none", linewidths=1.3, zorder=3,
               label="same four at 3-bit stress")
    # tiered (matryoshka) pack
    ax.scatter(jit(5.0, 3), [-6, -4, -5], s=46, color=C2, marker="D",
               zorder=3, label="tiered matryoshka pack (C5 tiers)")

    ax.annotate("fragility is crystal-priced: the width-floor\n"
                "crystal and the joint-STE matryoshka pay\n"
                "where d64h8 does not",
                xy=(4.9, -5.2), xytext=(3.0, -8.6), fontsize=8.5, color=SEC,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    ax.set_xlabel("bits / weight (measured raw pack; 3-bit arm nominal)")
    ax.set_ylabel("gate delta (solves / 120)")
    ax.set_xlim(2.5, 6.6)
    ax.set_ylim(-10, 6.5)
    ax.legend(loc="upper right", ncol=1)
    footer(ax, "PACKED CRYSTAL C0+C1 L10406 · C3 L10458 · C5 L10529 · "
               "C1 AT n=3 L11232 · MPS · gates-only")
    return _save(fig, "packing_curve")


# --------------------------------------------------------------- fig 2
# CAPACITY METER VERDICT (L10808): cplx_none 0.96, d64h8 1.61, L4d56 1.61,
#   DeepSeek-V3 L30 experts 2.33, Qwen2.5-0.5B dense 3.62, SmolLM2-1.7B
#   dense 3.85. Decision rule M <~ 2.5 sigma-law / M >~ 3.5 calibrated.
# C7 VERDICT (L10895): house NNUE 0.82; OLMoE experts M 2.85 (16x premium),
#   OLMoE attn M 3.11 (22x), Qwen dense 3.62 (34x), crystals 0.8-1.6 (~1x).

def fig_capacity_meter():
    fig, ax = figure(
        "The capacity meter splits weights into two classes before any packing runs",
        subtitle="M = span bits - code-stream entropy at a per-row sigma/2 step, "
                 "param-weighted; read from disk, zero calibration",
        figsize=(7.2, 4.0))

    # (label, M, label dy, premium note)
    rows = [
        ("house crystals", C0, [
            ("NNUE 0.82", 0.82, 11, None), ("cplx_none 0.96", 0.96, -22, None),
            ("d64h8 / L4d56 1.61", 1.61, 11, "~1x premium")]),
        ("MoE routed experts", C1, [
            ("DeepSeek-V3 L30 2.33", 2.33, -22, "premium unmeasured"),
            ("OLMoE-1B-7B 2.85", 2.85, 11, "16.3x")]),
        ("dense / attention", C2, [
            ("OLMoE attn 3.11", 3.11, 11, "22x"),
            ("Qwen2.5-0.5B 3.62", 3.62, -22, "34x*"),
            ("SmolLM2-1.7B 3.85", 3.85, 11, None)]),
    ]
    for x, txt in ((2.0, "sigma-law below ~2 (C7-sharpened)"), (3.5, "calibrated above ~3.5")):
        ax.axvline(x, color=CHROME["light"]["axis"], lw=0.9, ls=(0, (4, 3)),
                   zorder=1)
        ax.text(x, 2.72, txt, fontsize=8.5, color=MUTED, ha="center")

    for i, (name, col, pts) in enumerate(rows):
        y = len(rows) - 1 - i
        for label, m, dy, note in pts:
            ax.scatter([m], [y], s=64, color=col, zorder=3)
            ax.annotate(label, xy=(m, y), xytext=(0, dy), fontsize=8.5,
                        textcoords="offset points", ha="center", color=SEC)
            if note:
                # a starred note sits ABOVE its point (the label below it)
                ndy = 24 if (dy > 0 or note.endswith("*")) else -35
                ax.annotate(note.rstrip("*"), xy=(m, y), xytext=(0, ndy),
                            fontsize=8.5, textcoords="offset points",
                            ha="center", color=MUTED, family="monospace")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=10)
    ax.tick_params(axis="y", pad=6)
    ax.set_ylim(-0.6, 3.0)
    ax.set_xlim(0.4, 4.3)
    ax.set_xlabel("capacity meter M (bits)")
    ax.grid(axis="y", visible=False)
    footer(ax, "CAPACITY METER VERDICT L10808 · C7 VERDICT L10895 "
               "(premiums = sigma-grid DeltaKL / hqq DeltaKL) · desk read, n=1 per class")
    return _save(fig, "capacity_meter")


# --------------------------------------------------------------- fig 3
# SNAP-ALLOC (L9664) d56 EMA baseline 63, Q=16 both 63; SNAP-ALLOC Q=8
#   (L9692) both 56. SLACK RESTORATION (L9780) wfloor d256 comparator 65,
#   Q=64 both 65; Q=16 both 53 (SLACK RESTORATION arms, same entry).
# VERDICT HARDENING-P2 (L22658) d256 n=3: Q=64 0/-1/0, Q=16 -12/-13/-9.
# KNEE VERDICT (L7655) 19M control 49: Q=16 26, 24 45, 32 43, 48 47, 64 48;
#   RATIONAL-SNAP VERDICT (L7613) Q=4 0.

def fig_quant_knee():
    fig, ax = figure(
        "One knee, three crystals: the grid is priced in weight-sigma, not in denominators",
        subtitle="best-rational snap of every 2-D weight, |p|,q <= Q; "
                 "gate delta vs each crystal's own control",
        figsize=(7.2, 4.2))

    ax.axhspan(-3.5, 3.5, color=CHROME["light"]["grid"], zorder=0)
    ax.axhline(0, color=CHROME["light"]["axis"], lw=0.9, zorder=1)

    ax.plot([8, 16], [-7, 0], color=C0, marker="o", zorder=3,
            label="d56 EMA crystal (control 63)")
    ax.plot([16, 64], [-12, 0], color=C1, marker="s", zorder=3,
            label="wfloor d256 (control 65)")
    # d256 n=3 dispersion (HARDENING-P2)
    ax.scatter([16] * 3, [-12, -13, -9], s=22, color=C1, alpha=0.65, zorder=4)
    ax.scatter([64] * 3, [0, -1, 0], s=22, color=C1, alpha=0.65, zorder=4)
    ax.plot([4, 16, 24, 32, 48, 64], [-49, -23, -4, -6, -2, -1],
            color=C2, marker="^", zorder=3, label="19M crystal (control 49)")

    ax.annotate("Q=16 free\n(grid 0.48 sigma)", xy=(16, 0), xytext=(17, 3.5),
                fontsize=8.5, color=SEC)
    ax.annotate("Q=8 bites\n(0.96 sigma)", xy=(8, -7), xytext=(4.6, -12),
                fontsize=8.5, color=SEC,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("Q=64 free (0.25 sigma)\nQ=16 bites (1.0 sigma)", xy=(64, 0),
                xytext=(24, -18), fontsize=8.5, color=SEC,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("cliff in (16, 24]\n= 0.65-1.0 sigma", xy=(20, -13),
                xytext=(9.5, -33), fontsize=8.5, color=SEC,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 24, 32, 48, 64])
    ax.set_xticklabels(["4", "8", "16", "24", "32", "48", "64"])
    ax.set_xlabel("snap denominator bound Q (grid step 1/Q)")
    ax.set_ylabel("gate delta (solves / 120)")
    ax.set_ylim(-52, 8)
    ax.legend(loc="lower right")
    footer(ax, "SNAP-ALLOC L9664 / Q=8 L9692 · SIGMA-PRICED SNAP L9803 · "
               "KNEE VERDICT L7655 · HARDENING-P2 n=3 L22658 · MPS · "
               "sigma ratios as booked")
    return _save(fig, "quant_knee")


# --------------------------------------------------------------- fig 4
# ROTATIONAL SNAP R3 (L8610) 2x: 64/120 @ anti-mass 0.0002 (comparator 65).
# SYMMETRY LADDER S1 (L8682) 4x: 61 @ 0.0007. S4+S3 (L8733) 8x: 59 @ 0.0028
#   ("toll curve 2x:-1, 4x:-4, 8x:-6 of 65"). S2 (L8786): exact rotational
#   form exists at 2x WIDTH, gate 65 (toll 0) — annotation, not a rung.
# HARDENING-P3-R7 (L23168) 4x at n=3 data orders: 59/59/60 = -6/-6/-5.

def fig_symmetry_toll():
    fig, ax = figure(
        "The symmetry toll rises sublinearly: dense gates retrofit into shared "
        "structure through 8x",
        subtitle="one warm epoch of penalized conversion, d256 crystal, "
                 "comparator 65/120; all heals at anti-mass <= 0.003",
        figsize=(7.2, 4.0))

    ax.axhline(0, color=CHROME["light"]["axis"], lw=0.9, zorder=1)
    xs, tolls = [2, 4, 8], [-1, -4, -6]
    ax.plot(xs, tolls, color=C0, marker="o", zorder=3,
            label="conversion toll (n=1 per rung)")
    ax.scatter([3.7, 4.0, 4.3], [-6, -6, -5], s=34, color=C1, marker="s",
               zorder=4, label="4x re-run at n=3 data orders (P3-R7)")
    for x, t, g, dy in zip(xs, tolls,
                           ["complex, 64/120", "quaternion, 61/120",
                            "circulant C8, 59/120"], (-20, 12, 12)):
        ax.annotate(g, xy=(x, t), xytext=(0, dy), textcoords="offset points",
                    fontsize=8.5, color=SEC, ha="center")
    ax.scatter([2], [0], s=90, color=C2, marker="*", zorder=4,
               label="exact rotational form at 2x WIDTH (65/120, no toll)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["2x (params/2)", "4x (params/4)", "8x (params/8)"])
    ax.set_xlim(1.7, 9.5)
    ax.set_ylim(-8.5, 3)
    ax.set_xlabel("parameter sharing factor")
    ax.set_ylabel("gate delta vs comparator (solves / 120)")
    ax.legend(loc="lower left")
    footer(ax, "ROTATIONAL SNAP R3 L8610 · SYMMETRY LADDER S1 L8682 · "
               "S4+S3 L8733 · S2 L8786 · HARDENING-P3-R7 L23168 · Mac/mps · "
               "gates-only")
    return _save(fig, "symmetry_toll")


if __name__ == "__main__":
    fig_packing_curve()
    fig_capacity_meter()
    fig_quant_knee()
    fig_symmetry_toll()
