"""The gallery driver: every [R] render's invocation, recorded.

An image is [R] (reproducible gallery) only if running THIS script
regenerates it — the GALLERY list is the recorded invocation the
2026-08-13 rebirth program requires. Adding a render = adding an
entry; one-off unrecorded render commands do not produce [R] files.

Entries render through llmopt.lab.anatomy (single matrix -> the
dot-view triptych) and fan out through llmopt.figures.export
profiles into docs/assets/gallery/.

Usage:
  .venv/bin/python scripts/render_gallery.py            # all entries
  .venv/bin/python scripts/render_gallery.py --only NAME
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from llmopt.lab import anatomy
from llmopt.figures.export import export_profiles

OUTDIR = Path("docs/assets/gallery")


@dataclass(frozen=True)
class Entry:
    name: str          # output stem under docs/assets/gallery/
    ckpt: str          # checkpoint path (existence checked at render)
    key: str           # matrix family substring, e.g. "gate.weight"
    title: str         # display title (style v2 budget: <= ~6 words)
    ckpt2: str = ""    # second checkpoint -> two-panel compare
    label: str = ""    # left-panel label (compare only)
    label2: str = ""   # right-panel label (compare only)
    method: str = "triptych"  # triptych | pca | sphere | polar


# The recorded invocations. Populated from the verified rebirth
# survey (scratch/rebirth_pairs_2026-08-13.md) — high-confidence
# pairs only; deferred candidates stay commented until confirmed.
GALLERY: list[Entry] = [
    # Rebirth of archive/2026-08-12/neurons-polar-qwen-vs-19m.png
    # (survey: HIGH — invocation recovered from the archived pixels'
    # suptitle: plot_neurons --key gate.weight --method polar; both
    # checkpoints verified on disk. The Qwen side is the BASE model's
    # layer-14 gate slice, per the checkpoint name.)
    Entry(name="qwen-vs-19m-polar",
          ckpt="checkpoints/qwen05b_base_l14gate.pt",
          ckpt2="checkpoints/mathnative_19m.pt",
          key="gate.weight", method="polar",
          label="Qwen2.5-0.5B base, L14 gate",
          label2="math-native 19M, all layers",
          title="Qwen vs math-native 19M"),
]

# DEFERRED — needs Artin confirmation (substitutions change the claim;
# see scratch/rebirth_pairs_2026-08-13.md):
#  - qwen-vs-19m (pca): MEDIUM, pre-plot_neurons ad-hoc render
#  - ternary-vs-fp32 pca/sphere/polar: ternary ep0 gone; the honest
#    substitute mathnative_45m_ternary_3ep.pt is a DIFFERENT claim
#  - 19m-zoom (actually the Qwen L14 RL whisper): zoom quantile
#    unrecorded
#  - three-minds set: V3 expert tensor never existed on disk;
#    ternary ep1 overwritten — likely UNRECOVERABLE as-was


def render_compare(e: Entry, outdir: Path) -> list[str]:
    """Two checkpoints side by side, one projection method each,
    dots colored by rank-scaled magnitude on the house ramp."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from llmopt.figures import figstyle

    outs = []
    panels = [(e.ckpt, e.label), (e.ckpt2, e.label2)]
    for mode in ("light", "dark"):
        c = figstyle.CHROME[mode]
        plt.rcParams.update(figstyle.rc(mode))
        cmap = figstyle.continuous("magnitude", mode)
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
        for ax, (ckpt, label) in zip(axes, panels):
            _, W = anatomy.neuron_rows(ckpt, e.key)
            xs, ys, _ = anatomy.project(W, e.method)
            order = anatomy.rank_scale(W.norm(dim=1))
            ax.scatter(xs, ys, c=order, cmap=cmap, s=2.6, alpha=0.85,
                       linewidths=0, vmin=0, vmax=1, rasterized=True)
            ax.set_title(label, loc="left", fontsize=11,
                         fontweight=500, pad=12)
            ax.grid(False)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color(c["grid"])
        fig.text(0.5, 0.94, e.title, fontsize=18, fontweight=300,
                 color=c["primary"], ha="center")
        prov = " + ".join(anatomy.checkpoint_provenance(p)
                          for p, _ in panels)
        fig.text(0.045, 0.028, f"{prov} · render_gallery.py",
                 fontsize=7, color=c["muted"], family="monospace")
        fig.text(0.985, 0.028,
                 r"low $\leftarrow$ $\|w_i\|$ rank $\rightarrow$ high",
                 fontsize=8, fontweight=300, color=c["secondary"],
                 ha="right")
        fig.subplots_adjust(left=0.045, right=0.985, top=0.82,
                            bottom=0.1, wspace=0.12)
        out = str(outdir / f"{e.name}-{mode}.png")
        fig.savefig(out, dpi=300, facecolor=c["surface"])
        plt.close(fig)
        outs.append(out)
    return outs


def render_entry(e: Entry, outdir: Path = OUTDIR) -> list[str]:
    for ckpt in filter(None, (e.ckpt, e.ckpt2)):
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"{e.name}: checkpoint {ckpt} missing")
    if e.ckpt2:
        outs = render_compare(e, outdir)
    else:
        label, W = anatomy.neuron_rows(e.ckpt, e.key)
        outs = anatomy.render_dot_views(
            W, str(outdir / e.name), e.title,
            source_label=f"rows of {label}",
            provenance=(anatomy.checkpoint_provenance(e.ckpt)
                        + " · render_gallery.py"))
    for full in outs:
        p = Path(full)
        export_profiles(p, outdir, p.stem)
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    entries = [e for e in GALLERY if a.only in (None, e.name)]
    if a.only and not entries:
        raise SystemExit(f"no GALLERY entry named {a.only}")
    for e in entries:
        for o in render_entry(e):
            print(f"rendered {o}")
    print(f"{len(entries)} entries rendered")


if __name__ == "__main__":
    main()
