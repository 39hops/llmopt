"""GT-1 crest small-multiples — the gallery Wanted figure (2026-08-08).

Authored by Grok (xAI, Artin relay), landed by Fable with two review
edits: the unlabeled y=64 reference line removed (no booked meaning);
transcription asserts moved BEFORE the render.

Data TRANSCRIBED VERBATIM from:
  VERDICT MOE-GT-1-R4 (seeds 4242/777/90210, 45.3% vs paired full)
  VERDICT MOE-GT-1-R5 (seeds 111/222/333, 45.3% vs paired full)
  VERDICT MOE-GT-2-D4-PHYS-B (seeds 707/808/909, physics 45.3% top-
  demand mask vs paired full — domain-specificity dissociation)

No model code here; regenerate:
  python scripts/plot_gt1_crest.py

Claim shown: masking Qwen3-30B-A3B to top-45.3% math-demand experts
(58/128 per layer) beats the paired full model on mathgen L1-3 at
6/6 fresh paired seeds (pooled +14.7). The same keep fraction /
recipe on a mechanics gate CRATERS (pooled -59, 3/3) — matched
recall does not predict sign; crest is domain-specific.

Fences: FORMAT-BOUND, one vehicle, mathgen L1-3 vs mechanics gate,
Mac MLX, discovery seeds quarantined from R5 confirmation seeds.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

BG = "#0d1117"
FG = "#c9d1d9"
DIM = "#8b949e"
FULL_C = "#8b949e"
MASK_C = "#58a6ff"
PHYS_C = "#f85149"
MATH_DELTA_C = "#3fb950"

# R4 descriptive crest + R5 fully registered confirmation (RESULTS)
MATH_SEEDS = [4242, 777, 90210, 111, 222, 333]
MATH_FULL = [62, 73, 66, 63, 73, 63]
MATH_MASK = [76, 87, 82, 80, 82, 81]
# R4: +14/+14/+16; R5: +17/+9/+18 -> pooled +14.7 over 6

# D4-PHYS-B — same 45.3% top-demand recipe on mechanics gate
PHYS_SEEDS = [707, 808, 909]
PHYS_FULL = [42, 28, 24]
PHYS_MASK = [10, 16, 9]

MATH_D = [m - f for f, m in zip(MATH_FULL, MATH_MASK)]
PHYS_D = [m - f for f, m in zip(PHYS_FULL, PHYS_MASK)]
# transcription checksums BEFORE any render (dicts-are-the-checksum)
assert MATH_D == [14, 14, 16, 17, 9, 18], MATH_D
assert PHYS_D == [-32, -12, -15], PHYS_D
assert abs(sum(MATH_D) / 6 - 14.666) < 0.01


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(14.0, 6.2),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )
    fig.patch.set_facecolor(BG)
    for ax in (ax0, ax1):
        ax.set_facecolor(BG)
        ax.tick_params(colors="#484f58")
        for s in ax.spines.values():
            s.set_color("#30363d")

    # --- Panel A: six paired seeds, full vs 45.3% math mask ---
    x = np.arange(len(MATH_SEEDS), dtype=float)
    w = 0.36
    ax0.bar(x - w / 2, MATH_FULL, width=w, color=FULL_C, alpha=0.92,
            label="full model")
    ax0.bar(x + w / 2, MATH_MASK, width=w, color=MASK_C, alpha=0.92,
            label="top-45.3% math-demand mask")
    for i, (f, m) in enumerate(zip(MATH_FULL, MATH_MASK)):
        ax0.text(i - w / 2, f + 1.0, str(f), color=FG, ha="center",
                 fontsize=8, family="monospace")
        ax0.text(i + w / 2, m + 1.0, str(m), color=FG, ha="center",
                 fontsize=8, family="monospace")
        ax0.text(i, max(f, m) + 6.5, f"+{m - f}", color=MATH_DELTA_C,
                 ha="center", fontsize=8, family="monospace")
    ax0.set_xticks(x)
    ax0.set_xticklabels(
        [f"s{s}" for s in MATH_SEEDS],
        color=FG, fontsize=9, family="monospace",
    )
    ax0.set_ylabel("gate solves / 120", color=FG, fontsize=10,
                   family="monospace")
    ax0.set_ylim(0, 112)
    ax0.set_title(
        "A  math crest — 45.3% keep beats paired full at 6/6 seeds",
        color=FG, fontsize=11, family="monospace", pad=10, loc="left",
    )
    ax0.legend(loc="upper left", facecolor=BG, edgecolor="#30363d",
               fontsize=8, labelcolor=FG)
    ax0.text(
        0.5, -0.14,
        "each full bar is that seed's own paired baseline (R4 pool v R5 "
        "pool) · R4 +14/+14/+16 · R5 +17/+9/+18 · pooled +14.7 vs +7 bar",
        transform=ax0.transAxes, color=DIM, fontsize=7.5,
        family="monospace", ha="center",
    )

    # --- Panel B: domain dissociation (delta vs full) ---
    rng = np.random.default_rng(0)
    ax1.axhline(0, color="#484f58", lw=0.9)
    ax1.scatter(
        rng.normal(0.0, 0.04, size=len(MATH_D)), MATH_D,
        s=55, color=MATH_DELTA_C, zorder=3, label="mathgen L1-3",
    )
    ax1.scatter(
        rng.normal(1.0, 0.04, size=len(PHYS_D)), PHYS_D,
        s=55, color=PHYS_C, zorder=3, label="mechanics gate",
    )
    ax1.hlines(
        [sum(MATH_D) / len(MATH_D), sum(PHYS_D) / len(PHYS_D)],
        xmin=[-0.25, 0.75], xmax=[0.25, 1.25],
        colors=[MATH_DELTA_C, PHYS_C], lw=2.5, zorder=2,
    )
    ax1.text(-0.32, sum(MATH_D) / len(MATH_D),
             f"pooled +{sum(MATH_D) / len(MATH_D):.1f}",
             color=MATH_DELTA_C, ha="right", va="center", fontsize=9,
             family="monospace")
    ax1.text(1.0, sum(PHYS_D) / len(PHYS_D) - 6.5,
             f"pooled {sum(PHYS_D):.0f}",
             color=PHYS_C, ha="center", fontsize=9,
             family="monospace")
    for d in MATH_D:
        ax1.annotate(f"{d:+d}", (0.0, d), textcoords="offset points",
                     xytext=(12, 0), color=DIM, fontsize=7,
                     family="monospace")
    for d in PHYS_D:
        ax1.annotate(f"{d:+d}", (1.0, d), textcoords="offset points",
                     xytext=(12, 0), color=DIM, fontsize=7,
                     family="monospace")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(
        ["math\n(6 seeds)", "mechanics\n(3 seeds)"],
        color=FG, fontsize=9, family="monospace",
    )
    ax1.set_ylabel("delta vs paired full (solves)", color=FG,
                   fontsize=10, family="monospace")
    ax1.set_xlim(-0.85, 1.55)
    ax1.set_ylim(-70, 35)
    ax1.set_title(
        "B  domain-specificity — same recipe, opposite sign",
        color=FG, fontsize=11, family="monospace", pad=10, loc="left",
    )
    ax1.legend(loc="lower left", facecolor=BG, edgecolor="#30363d",
               fontsize=8, labelcolor=FG)
    ax1.text(
        0.5, -0.16,
        "D4-PHYS-B: physics 45.3% top-demand mask · matched recall "
        "!= capability sign",
        transform=ax1.transAxes, color=DIM, fontsize=7.5,
        family="monospace", ha="center",
    )

    head = _git_head()
    fig.text(
        0.01, 0.01,
        "VERDICT MOE-GT-1-R4/R5 + MOE-GT-2-D4-PHYS-B (docs/RESULTS.md) "
        "· Qwen3-30B-A3B-4bit · FORMAT-BOUND, one vehicle · "
        f"plot_gt1_crest.py @ {head}",
        color=DIM, fontsize=7, family="monospace",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = Path("docs/assets/gt1-crest-small-multiples.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=BG)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
