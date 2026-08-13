"""GT-1 crest small-multiples — the gallery Wanted figure (2026-08-08).

Authored by Grok (xAI, Artin relay), landed by Fable with two review
edits: the unlabeled y=64 reference line removed (no booked meaning);
transcription asserts moved BEFORE the render. Restyled 2026-08-13 to
the house figstyle (light+dark pair, style v2 text budget); data and
checksums unchanged.

Data TRANSCRIBED VERBATIM from:
  VERDICT MOE-GT-1-R4 (seeds 4242/777/90210, 45.3% vs paired full)
  VERDICT MOE-GT-1-R5 (seeds 111/222/333, 45.3% vs paired full)
  VERDICT MOE-GT-2-D4-PHYS-B (seeds 707/808/909, physics 45.3% top-
  demand mask vs paired full — domain-specificity dissociation)

No model code here; regenerate:
  python scripts/plot_gt1_crest.py
  (writes gt1-crest-small-multiples.png light + -dark pair)

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


def render(mode: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from llmopt.figures import figstyle

    c = figstyle.CHROME[mode]
    plt.rcParams.update(figstyle.rc(mode))
    full_c = figstyle.color("full", mode=mode)
    mask_c = figstyle.color("masked", mode=mode)
    phys_c = figstyle.color("refuted", mode=mode)

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(14.0, 6.2),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )

    # --- Panel A: six paired seeds, full vs 45.3% math mask ---
    x = np.arange(len(MATH_SEEDS), dtype=float)
    w = 0.36
    ax0.bar(x - w / 2, MATH_FULL, width=w, color=full_c, alpha=0.92,
            label="full model")
    ax0.bar(x + w / 2, MATH_MASK, width=w, color=mask_c, alpha=0.92,
            label="top-45.3% math-demand mask")
    for i, (f, m) in enumerate(zip(MATH_FULL, MATH_MASK)):
        ax0.text(i, max(f, m) + 3.0, f"+{m - f}", color=mask_c,
                 ha="center", fontsize=8.5, family="monospace")
    ax0.set_xticks(x)
    ax0.set_xticklabels([f"s{s}" for s in MATH_SEEDS], fontsize=9)
    ax0.set_ylabel("gate solves / 120", fontsize=10)
    ax0.set_ylim(0, 108)
    ax0.set_title("MATH — mask beats full, 6/6 seeds",
                  fontsize=11, fontweight=500, pad=10, loc="left")
    ax0.legend(loc="upper left", fontsize=8.5)

    # --- Panel B: domain dissociation (delta vs full) ---
    rng = np.random.default_rng(0)
    ax1.axhline(0, color=c["axis"], lw=0.9)
    ax1.scatter(rng.normal(0.0, 0.04, size=len(MATH_D)), MATH_D,
                s=55, color=mask_c, zorder=3, label="mathgen L1-3")
    ax1.scatter(rng.normal(1.0, 0.04, size=len(PHYS_D)), PHYS_D,
                s=55, color=phys_c, zorder=3, label="mechanics gate")
    ax1.hlines(
        [sum(MATH_D) / len(MATH_D), sum(PHYS_D) / len(PHYS_D)],
        xmin=[-0.25, 0.75], xmax=[0.25, 1.25],
        colors=[mask_c, phys_c], lw=2.5, zorder=2,
    )
    ax1.text(-0.32, sum(MATH_D) / len(MATH_D),
             f"pooled +{sum(MATH_D) / len(MATH_D):.1f}",
             color=mask_c, ha="right", va="center", fontsize=9,
             family="monospace")
    ax1.text(1.0, sum(PHYS_D) / len(PHYS_D) - 6.5,
             f"pooled {sum(PHYS_D):.0f}",
             color=phys_c, ha="center", fontsize=9,
             family="monospace")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["math\n(6 seeds)", "mechanics\n(3 seeds)"],
                        fontsize=9)
    ax1.set_ylabel("delta vs paired full (solves)", fontsize=10)
    ax1.set_xlim(-0.85, 1.55)
    ax1.set_ylim(-70, 35)
    ax1.set_title("SAME RECIPE — opposite sign", fontsize=11,
                  fontweight=500, pad=10, loc="left")
    ax1.legend(loc="lower left", fontsize=8.5)

    fig.text(
        0.01, 0.01,
        "VERDICT MOE-GT-1-R4/R5 + MOE-GT-2-D4-PHYS-B (docs/RESULTS.md) "
        "· Qwen3-30B-A3B-4bit · FORMAT-BOUND, one vehicle · "
        f"plot_gt1_crest.py @ {_git_head()}",
        color=c["muted"], fontsize=7, family="monospace",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    suffix = "" if mode == "light" else "-dark"
    out = Path(f"docs/assets/gallery/gt1-crest-small-multiples{suffix}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=c["surface"])
    plt.close(fig)
    return out


def main() -> None:
    for mode in ("light", "dark"):
        print(f"saved {render(mode)}")


if __name__ == "__main__":
    main()
