"""Render the README hero: the weight-space anatomy of one crystal.

Each dot is one NEURON — a row of a projection matrix — and color is
that neuron's magnitude (row norm). Three views of the same matrix,
the three projections the crystal program has always used:

  pca     — global linear axes (the original crystal view)
  sphere  — neurons unit-normalized, stereographic projection of the
            top-3-PC directions: pure DIRECTION structure, magnitude
            moved entirely into color
  polar   — PC1+i*PC2 (variance-whitened) read as a complex number:
            angle = phase around the dominant plane, radius = neuron
            magnitude

Styling comes from llmopt.lab.figstyle (validated palette, vendored
Inter/JetBrains Mono), rendered at 300 dpi in light and dark variants.
Provenance footer: checkpoint basename + sha256[:8] + repo HEAD.

Usage:
  .venv/bin/python scripts/render_hero_neurons.py \
      --ckpt checkpoints/gallery19m_s1.pt \
      --out docs/assets/neurons-19m
  (writes <out>-light.png and <out>-dark.png)
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha8(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def repo_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def neuron_matrix(ckpt: str, key_sub: str):
    """All matrices of the family, rows stacked: every gate neuron in
    the model, not one layer's. Same-shaped, same-role matrices, so
    pooled PCA is a legal 'all gate neurons' view."""
    import torch
    sd = torch.load(ckpt, map_location="cpu")
    keys = sorted(k for k in sd if key_sub in k)
    if not keys:
        raise SystemExit(f"no '{key_sub}' matrices in {ckpt}")
    W = torch.cat([sd[k].float() for k in keys], dim=0)
    label = f"{len(keys)} layers of *.{key_sub.split('.')[0]}"
    return label, W


def project(W, method: str):
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
        denom = 1 + p3[:, 2].clamp(min=-0.99)  # stereographic, south pole
        return p3[:, 0] / denom, p3[:, 1] / denom, mag
    if method == "polar":
        # Variance-whitened before the angle (2026-07-25 instrument
        # fix): only then does a uniform ring mean isotropy.
        X = W - W.mean(0)
        _, S, V = torch.linalg.svd(X, full_matrices=False)
        P = (X @ V[:2].T) / S[:2].clamp(min=1e-12)
        z = torch.complex(P[:, 0], P[:, 1])
        return z.angle(), mag, mag
    raise SystemExit(f"unknown method {method}")


def render(ckpt: str, key: str, out_stem: str, title: str,
           mode: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from llmopt.lab import figstyle

    c = figstyle.CHROME[mode]
    plt.rcParams.update(figstyle.rc(mode))

    # Magnitude is a sequential job: ONE hue, surface-light to deep.
    ramp = figstyle.SEQUENTIAL
    if mode == "dark":
        # On the dark surface the ramp runs dim-to-bright so high
        # magnitude reads as the bright end.
        ramp = list(reversed(ramp))
    cmap = LinearSegmentedColormap.from_list("house-seq", ramp)

    layer_key, W = neuron_matrix(ckpt, key)
    n, d = W.shape

    fig = plt.figure(figsize=(13.5, 5.4))
    grid = fig.add_gridspec(1, 3, wspace=0.14,
                            left=0.045, right=0.985,
                            top=0.80, bottom=0.14)
    panels = [
        ("pca", "PCA", "global linear axes"),
        ("sphere", "SPHERE", "directions only — magnitude in color"),
        ("polar", "POLAR", "phase (rad) vs magnitude"),
    ]
    import torch
    mags = W.norm(dim=1)
    # Rank-scaled color: magnitudes cluster tightly, so a linear map
    # paints one blue. Ranks spread the ramp over the distribution's
    # SHAPE; the caption says so.
    order = torch.empty_like(mags)
    order[mags.argsort()] = torch.linspace(0, 1, len(mags))
    for i, (method, name, sub) in enumerate(panels):
        ax = fig.add_subplot(grid[0, i])
        xs, ys, mag = project(W, method)
        ax.scatter(xs, ys, c=order, cmap=cmap, s=2.2, alpha=0.85,
                   linewidths=0, vmin=0, vmax=1,
                   rasterized=True)
        ax.set_title(name, loc="left", fontsize=12,
                     fontweight="semibold", pad=16)
        ax.text(0, 1.02, sub, transform=ax.transAxes,
                fontsize=8.5, color=c["muted"])
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(c["grid"])
        if method == "polar":
            ax.set_xlabel("phase of PC1 + i·PC2", fontsize=8.5)
            ax.set_ylabel("neuron magnitude", fontsize=8.5)

    fig.text(0.045, 0.945, title, fontsize=16,
             fontweight="semibold", color=c["primary"])
    fig.text(0.045, 0.895,
             f"each dot one neuron (rows of {layer_key}, "
             f"{n:,} neurons × {d}-dim) · color = neuron magnitude "
             f"(rank-scaled)",
             fontsize=9.5, color=c["secondary"])
    fig.text(0.045, 0.028,
             f"{Path(ckpt).name} {sha8(ckpt)} · "
             f"render_hero_neurons.py @ {repo_head()}",
             fontsize=7, color=c["muted"], family="monospace")

    out = f"{out_stem}-{mode}.png"
    fig.savefig(out, dpi=300, facecolor=c["surface"],
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"saved {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--key", default="gate.weight")
    ap.add_argument("--title", default="THE CRYSTAL — weight-space "
                    "anatomy of a math-native 19M model")
    ap.add_argument("--out", required=True,
                    help="output stem; writes <out>-light.png and "
                         "<out>-dark.png")
    a = ap.parse_args()
    for mode in ("dark", "light"):
        render(a.ckpt, a.key, a.out, a.title, mode)


if __name__ == "__main__":
    main()
