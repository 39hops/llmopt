"""Classify every docs/assets/*.png into the figure taxonomy.

Classes come from the rows in docs/assets/README.md:
[PUBLISHED] web/, [HERO] the README front-door pair, [R] regenerable
gallery renders, [H] frozen pixels. Anything not matched prints as
UNKNOWN, first, for manual classification (surviving script+checkpoint
pair -> R, else -> H). Read-only: prints a `class<TAB>filename` table.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"

HERO = {"neurons-19m-light.png", "neurons-19m-dark.png"}

# [R] stems enumerated in the assets README (script+checkpoint survive).
R_PATTERNS = [
    "gt1-crest-*.png",
    "identity-crest-*.png",
    "neurons-gen6-*.png",
    "neurons-pca-*.png",
    "neurons-113m-growth-*.png",
    # the four-diets set
    "neurons-phase-density-four-diets.png",
    "neurons-polar-four-diets.png",
    "neurons-polar-four-diets-normalized.png",
]

# [H] enumerated frozen set: old hero, zoom, three-minds, crystal era.
H_PATTERNS = [
    "neurons-19m.png",
    "neurons-19m-zoom.png",
    "three-minds-*.png",
    # Resolved from UNKNOWN (2026-08-13): no recorded generator
    # invocation survives — plot_neurons.py takes arbitrary --out, and
    # no driver in scripts//scratch/ names these outputs, so there is
    # no script+checkpoint PAIR to regenerate from. Frozen as [H].
    # (gen_lab_overview_pdf.py CONSUMES neurons-qwen-vs-19m.png; a
    # consumer is not a generator.)
    "magic-scatter-*.png",
    "neuron-density-vs-phase.png",
    "neuron-weighting-pr.png",
    "neurons-binary768-vs-ternary.png",
    "neurons-polar-qwen-vs-19m.png",
    "neurons-polar-ternary-vs-fp32*.png",
    "neurons-polar-ternary6ep-vs-fp32.png",
    "neurons-polar-v3expert-vs-19m.png",
    "neurons-qwen-vs-19m.png",
    "neurons-sphere-45m-vs-113m.png",
    "neurons-sphere-ternary-vs-fp32.png",
    "neurons-ternary-growth-ep3-ep6.png",
    "neurons-wfloor-*.png",
    "neurons-zoom-fp32-vs-ternary-displacement.png",
    "rl-vs-sft-weight-delta.png",
    "symmetry-rmt-and-gtheta.png",
    "width-curve-gen4.png",
]


def classify(name: str) -> str:
    if name in HERO:
        return "HERO"
    for pat in R_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return "R"
    for pat in H_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return "H"
    return "UNKNOWN"


def main() -> None:
    rows = [(classify(p.name), p.name) for p in sorted(ASSETS.glob("*.png"))]
    rows += [("PUBLISHED", f"web/{p.name}") for p in sorted((ASSETS / "web").glob("*"))
             if p.suffix in (".png", ".svg")]
    order = {"UNKNOWN": 0, "HERO": 1, "R": 2, "H": 3, "PUBLISHED": 4}
    rows.sort(key=lambda r: (order[r[0]], r[1]))
    for cls, name in rows:
        print(f"{cls}\t{name}")
    n_unknown = sum(1 for c, _ in rows if c == "UNKNOWN")
    print(f"# total={len(rows)} unknown={n_unknown}")


if __name__ == "__main__":
    main()
