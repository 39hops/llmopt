"""Regenerate the README / paper figures from booked numbers.

Every number below is transcribed from a RESULTS verdict and carries
that entry's line in the figure's own provenance footer, so a figure
that travels away from the repo still says what backs it. Re-run after
any amendment that moves one of these numbers:

    .venv/bin/python scripts/gen_readme_figures.py

Writes light and dark PNG + SVG into docs/assets/. Regenerable exhaust
by the logs doctrine — the evidence is the RESULTS entry, never the
image.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llmopt.lab.figures import curves, gate_bars, ladder  # noqa: E402

OUT = "docs/assets"


def merge_space():
    """VERDICT MERGE-SPACE-1/2/3/5 — the crater and its mechanism."""
    gate_bars(
        "fig-merge-space",
        {"birth s1": (12, 120), "birth s2": (23, 120),
         "birth s3": (30, 120), "birth s4": (28, 120),
         "independent\nmerge ×6": (0, 120),
         "same-init\nmerge": (13, 120),
         "4-way\nsoup": (14, 120)},
        title="Averaging independently born weights does not degrade a "
              "model — it ends it",
        subtitle="d64 micro-stars on one RTX 3080. All six independent-pair "
                 "averages gated exactly zero at every level; merges inside "
                 "a shared-init lineage land in the parent band.",
        source="VERDICT MERGE-SPACE-1/4 · docs/RESULTS.md#L26770 · n=1 per cell",
        outdir=OUT)


def keff():
    """PRE-REG KEFF-PROBE-1 — measured effective context, no training."""
    curves(
        "fig-effective-context", [4, 8, 16, 32, 64, 128],
        {"d64": [2.2529, 1.8553, 1.577, 1.3943, 1.3021, 0.6742],
         "d128": [2.1089, 1.8968, 1.6617, 1.311, 1.0964, 0.4617],
         "d256": [2.112, 1.9077, 1.5719, 1.2432, 0.8571, 0.3232],
         "d512": [2.0495, 1.96, 1.5137, 1.1282, 0.7748, 0.2899]},
        title="Width buys the ability to exploit long context, not local "
              "precision",
        subtitle="Loss at deep positions when the model may see only the "
                 "last k tokens. The widths are indistinguishable to k=16 "
                 "and separate steadily after it.",
        xlabel="context the model may see, k (tokens)",
        ylabel="loss at positions ≥128 (nats)", logx=True,
        source="PRE-REG KEFF-PROBE-1 · 400 fixed rows · Mac CPU · no training",
        outdir=OUT)


def floor_ladder():
    """VERDICT FLOOR-HK-1 — the floor descends but never nears the knee."""
    ladder(
        "fig-loss-floor",
        {"d64": 0.4364, "d128": 0.3815, "d256": 0.3566, "d512": 0.3478},
        title="The training floor walks down with width — and stops nowhere "
              "near the corpus limit",
        subtitle="Fresh births on one warm diet, fp32. Eight times the width "
                 "buys roughly one token of effective context.",
        xlabel="model width", ylabel="final-epoch train loss (nats)",
        reference=("corpus entropy at k=32: H = 0.187", 0.187),
        entity="champion",
        source="VERDICT FLOOR-HK-1 · docs/RESULTS.md#L27055 · Mac · n=1",
        outdir=OUT)


def ssm():
    """VERDICT SSM-STAR-1 — an honest loss, booked like any other."""
    gate_bars(
        "fig-ssm-star",
        {"attention": (38, 120), "state-space": (2, 120)},
        title="The house's first state-space model lost on both axes",
        subtitle="Paired arms, same device, same diet, same seed. The SSM "
                 "also trained 22.6× slower. Booked as a loss, like every "
                 "other result here.",
        source="VERDICT SSM-STAR-1 · docs/RESULTS.md#L27282 · 3080 · n=1",
        outdir=OUT)


if __name__ == "__main__":
    merge_space()
    keff()
    floor_ladder()
    ssm()
    print("\nregenerated README figures in docs/assets/")
