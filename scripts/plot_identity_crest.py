"""The identity-era figure the gallery lacked (2026-08-08 pass).

Data are TRANSCRIBED VERBATIM from VERDICT EX-FRESH (docs/RESULTS.md
2026-08-07: all twelve dicts sum-verified there) — this script holds
no model code; it renders the booked table so the repo's strongest
claim has a visual. Regenerate: python scripts/plot_identity_crest.py

Claim shown: capability follows expert IDENTITY. At three fresh
paired seeds (1001/2002/3003, discovery pool quarantined), both a
54.7%-keep swap-derived mask (+53 pooled) and a 98.7%-keep named
80-carrier deletion (+55 pooled) beat the full 30B-class MoE on the
120-prompt mathematics gate; the rank-matched random deletion
control (+28) shows the router is over-inclusive at the carriers'
rank class, and identity roughly doubles that gain.
Fences travel: FORMAT-BOUND, REGIME-SCOPED (measured deployment
artifacts), one vehicle, mathgen L1-3, Mac MLX.
"""
import subprocess

BG = "#0d1117"
FG = "#c9d1d9"
DIM = "#8b949e"

SEEDS = [1001, 2002, 3003]
ARMS = [  # (label, per-seed solves /120, color)
    ("full model", [59, 67, 63], "#8b949e"),
    ("swap-derived keep-set\n(54.7% keep)", [77, 83, 82], "#58a6ff"),
    ("named 80-carrier deletion\n(98.7% keep)", [78, 85, 81], "#bc8cff"),
    ("rank-matched random\ndeletion control", [70, 76, 71], "#39c5cf"),
]
POOLED = [("swap - full", "+53"), ("deletion - full", "+55"),
          ("deletion - control", "+27"), ("control - full", "+28")]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for gi in range(len(SEEDS)):
        for ai, (label, vals, col) in enumerate(ARMS):
            x = gi * (len(ARMS) + 1.2) * 0.22 + ai * 0.22
            ax.bar(x, vals[gi], width=0.19, color=col, alpha=0.92,
                   label=label.replace("\n", " ") if gi == 0 else None)
            ax.text(x, vals[gi] + 1.2, str(vals[gi]), color=FG,
                    ha="center", fontsize=9, family="monospace")
    centers = [gi * (len(ARMS) + 1.2) * 0.22 + 1.5 * 0.22
               for gi in range(len(SEEDS))]
    ax.set_xticks(centers)
    ax.set_xticklabels([f"fresh seed {s}" for s in SEEDS],
                       color=FG, fontsize=10, family="monospace")
    ax.set_ylabel("gate solves / 120", color=FG, fontsize=10,
                  family="monospace")
    ax.set_ylim(0, 95)
    ax.tick_params(colors="#484f58")
    for s in ax.spines.values():
        s.set_color("#30363d")
    ax.set_title("capability follows expert IDENTITY — both crest "
                 "forms replicate at three fresh paired seeds",
                 color=FG, fontsize=12.5, family="monospace", pad=14)
    ax.legend(loc="upper left", facecolor=BG, edgecolor="#30363d",
                    fontsize=8.5, labelcolor=FG)
    pooled_txt = "pooled: " + "  ".join(f"{k} {v} (3/3)"
                                        for k, v in POOLED)
    ax.text(0.5, -0.115, pooled_txt, transform=ax.transAxes,
            color=FG, fontsize=9, family="monospace", ha="center")
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        head = "unknown"
    ax.text(0.0, -0.16,
            "VERDICT EX-FRESH (docs/RESULTS.md 2026-08-07) · 30B-class "
            "MoE, 120-prompt mathematics gate, Mac MLX · FORMAT-BOUND, "
            "REGIME-SCOPED, one vehicle · plot_identity_crest.py @ "
            + head,
            transform=ax.transAxes, color=DIM, fontsize=7,
            family="monospace")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    out = "docs/assets/gallery/identity-crest-fresh-seeds.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
