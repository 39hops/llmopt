"""The identity-era figure the gallery lacked (2026-08-08 pass).

Data are TRANSCRIBED VERBATIM from VERDICT EX-FRESH (docs/RESULTS.md
2026-08-07: all twelve dicts sum-verified there) — this script holds
no model code; it renders the booked table so the repo's strongest
claim has a visual. Restyled 2026-08-13 to the house figstyle
(light+dark pair, style v2 text budget); data unchanged.
Regenerate: python scripts/plot_identity_crest.py

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

SEEDS = [1001, 2002, 3003]
ARMS = [  # (label, per-seed solves /120, entity slot)
    ("full model", [59, 67, 63], "full"),
    ("swap-derived keep-set (54.7% keep)", [77, 83, 82], "masked"),
    ("named 80-carrier deletion (98.7% keep)", [78, 85, 81], "merged"),
    ("rank-matched random deletion control", [70, 76, 71], "control"),
]
POOLED = [("swap - full", "+53"), ("deletion - full", "+55"),
          ("deletion - control", "+27"), ("control - full", "+28")]


def render(mode: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from llmopt.figures import figstyle

    c = figstyle.CHROME[mode]
    plt.rcParams.update(figstyle.rc(mode))

    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    for gi in range(len(SEEDS)):
        for ai, (label, vals, entity) in enumerate(ARMS):
            x = gi * (len(ARMS) + 1.2) * 0.22 + ai * 0.22
            ax.bar(x, vals[gi], width=0.19,
                   color=figstyle.color(entity, mode=mode), alpha=0.92,
                   label=label if gi == 0 else None)
            ax.text(x, vals[gi] + 1.2, str(vals[gi]),
                    color=c["secondary"], ha="center", fontsize=9,
                    family="monospace")
    centers = [gi * (len(ARMS) + 1.2) * 0.22 + 1.5 * 0.22
               for gi in range(len(SEEDS))]
    ax.set_xticks(centers)
    ax.set_xticklabels([f"fresh seed {s}" for s in SEEDS], fontsize=10)
    ax.set_ylabel("gate solves / 120", fontsize=10)
    ax.set_ylim(0, 95)
    ax.set_title("Capability follows expert identity",
                 fontsize=13, fontweight=500, pad=40)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncols=2,
              fontsize=8.5, frameon=False)
    pooled_txt = "pooled: " + "  ".join(f"{k} {v} (3/3)"
                                        for k, v in POOLED)
    ax.text(0.5, -0.1, pooled_txt, transform=ax.transAxes,
            color=c["secondary"], fontsize=9, family="monospace",
            ha="center")
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        head = "unknown"
    ax.text(0.0, -0.145,
            "VERDICT EX-FRESH (docs/RESULTS.md 2026-08-07) · 30B-class "
            "MoE, 120-prompt mathematics gate, Mac MLX · FORMAT-BOUND, "
            "REGIME-SCOPED, one vehicle · plot_identity_crest.py @ "
            + head,
            transform=ax.transAxes, color=c["muted"], fontsize=7,
            family="monospace")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    suffix = "" if mode == "light" else "-dark"
    out = f"docs/assets/gallery/identity-crest-fresh-seeds{suffix}.png"
    fig.savefig(out, dpi=150, facecolor=c["surface"])
    plt.close(fig)
    return out


def main() -> None:
    for mode in ("light", "dark"):
        print(f"saved {render(mode)}")


if __name__ == "__main__":
    main()
