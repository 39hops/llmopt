"""Weight-space anatomy: neurons as dots, reusable for ANY matrix.

The lab's standing render family. Each dot is one NEURON — a row of a
projection matrix — and color is that neuron's magnitude (row norm,
rank-scaled across the panel). Three projections, unchanged since the
crystal era:

  pca     — global linear axes
  sphere  — rows unit-normalized, stereographic projection of the
            top-3-PC directions: pure DIRECTION structure
  polar   — variance-whitened PC1+i*PC2 as a complex number:
            angle = phase around the dominant plane, radius = row norm

Sources: a house checkpoint (``neuron_rows``), or any torch matrix you
hand ``render_dot_views`` — streamed big-model expert shards included.
Styling comes from lab.figstyle (validated palette, vendored fonts,
math notation in the text face); every render stamps a provenance
footer. This module is the library; scripts/render_hero_neurons.py is
its CLI.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

__all__ = ["neuron_rows", "project", "rank_scale", "render_dot_views",
           "checkpoint_provenance"]


def _sha8(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def _repo_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def neuron_rows(ckpt: str, key_sub: str = "gate.weight"):
    """All matrices of one family from a checkpoint, rows stacked:
    every such neuron in the model, not one layer's. Same-shaped,
    same-role matrices, so pooled PCA is a legal family view.
    Returns (label, W)."""
    import torch
    sd = torch.load(ckpt, map_location="cpu")
    keys = sorted(k for k in sd if key_sub in k)
    if not keys:
        raise ValueError(f"no '{key_sub}' matrices in {ckpt}")
    W = torch.cat([sd[k].float() for k in keys], dim=0)
    label = f"{len(keys)} layers of *.{key_sub.split('.')[0]}"
    return label, W


def project(W, method: str):
    """(xs, ys, mag) for one of pca | sphere | polar."""
    import torch
    mag = W.norm(dim=1)
    if method == "pca":
        X = W - W.mean(0)
        _, _, V = torch.linalg.svd(X, full_matrices=False)
        P = X @ V[:2].T
        return P[:, 0], P[:, 1], mag
    if method == "sphere":
        U = W / W.norm(dim=1, keepdim=True)
        X = U - U.mean(0)
        _, _, V = torch.linalg.svd(X, full_matrices=False)
        p3 = X @ V[:3].T
        p3 = p3 / p3.norm(dim=1, keepdim=True)
        denom = 1 + p3[:, 2].clamp(min=-0.99)  # stereographic, S pole
        return p3[:, 0] / denom, p3[:, 1] / denom, mag
    if method == "polar":
        # Variance-whitened before the angle (2026-07-25 instrument
        # fix): only then does a uniform ring mean isotropy.
        X = W - W.mean(0)
        _, S, V = torch.linalg.svd(X, full_matrices=False)
        P = (X @ V[:2].T) / S[:2].clamp(min=1e-12)
        z = torch.complex(P[:, 0], P[:, 1])
        return z.angle(), mag, mag
    raise ValueError(f"unknown method {method}")


def rank_scale(mag):
    """Magnitudes -> uniform [0,1] ranks. Row norms cluster tightly,
    so a linear color map paints one flat hue; ranks spread the ramp
    over the distribution's SHAPE. Captions must say 'rank-scaled'."""
    import torch
    order = torch.empty_like(mag)
    order[mag.argsort()] = torch.linspace(0, 1, len(mag))
    return order


# One tiny descriptor per panel — the README prose beside the image
# carries the full explanation (style v2 master rule).
PANELS = [
    ("pca", "PCA", "global axes"),
    ("sphere", "SPHERE", "directions only"),
    ("polar", "POLAR", "phase vs magnitude"),
]


def render_dot_views(W, out_stem: str, title: str, source_label: str,
                     provenance: str, modes=("dark", "light"),
                     dpi: int = 300) -> list[str]:
    """The dot-view triptych for ANY neuron matrix.

    W: (n, d) torch matrix — from neuron_rows, or streamed/dequantized
    shards of a big model. provenance: what goes in the footer beside
    the repo HEAD (for checkpoints use f"{name} {sha8}"; for streamed
    experts name the repo + shard selection). Writes
    <out_stem>-<mode>.png per mode and returns the paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from llmopt.figures import figstyle

    order = rank_scale(W.norm(dim=1))
    outs = []
    for mode in modes:
        c = figstyle.CHROME[mode]
        plt.rcParams.update(figstyle.rc(mode))
        cmap = figstyle.continuous("magnitude", mode)

        fig = plt.figure(figsize=(13.5, 5.4))
        grid = fig.add_gridspec(1, 3, wspace=0.14, left=0.045,
                                right=0.985, top=0.80, bottom=0.14)
        for i, (method, name, sub) in enumerate(PANELS):
            ax = fig.add_subplot(grid[0, i])
            xs, ys, _ = project(W, method)
            ax.scatter(xs, ys, c=order, cmap=cmap, s=2.2, alpha=0.85,
                       linewidths=0, vmin=0, vmax=1, rasterized=True)
            ax.set_title(name, loc="left", fontsize=12,
                         fontweight=500, pad=16)
            ax.text(0, 1.02, sub, transform=ax.transAxes,
                    fontsize=8.5, fontweight=300, color=c["muted"])
            ax.grid(False)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color(c["grid"])
            if method == "polar":
                ax.set_xlabel("phase", fontsize=8.5)
                ax.set_ylabel("magnitude", fontsize=8.5)

        # Type hierarchy: Light 300 centered display title, Medium 500
        # panel names, muted eyebrows. Prose beside the image explains;
        # the chart carries only the title, one ramp cue, and the fence.
        fig.text(0.5, 0.93, title, fontsize=19, fontweight=300,
                 color=c["primary"], ha="center")
        fig.text(0.985, 0.028,
                 r"low $\leftarrow$ $\|w_i\|$ rank $\rightarrow$ high",
                 fontsize=8, fontweight=300, color=c["secondary"],
                 ha="right")
        fig.text(0.045, 0.028, f"{provenance} · @ {_repo_head()}",
                 fontsize=7, color=c["muted"], family="monospace")

        out = f"{out_stem}-{mode}.png"
        fig.savefig(out, dpi=dpi, facecolor=c["surface"],
                    bbox_inches=None, pad_inches=0)
        plt.close(fig)
        outs.append(out)
    return outs


def checkpoint_provenance(ckpt: str) -> str:
    """Footer text for a checkpoint source: basename + sha256[:8]."""
    return f"{Path(ckpt).name} {_sha8(ckpt)}"
