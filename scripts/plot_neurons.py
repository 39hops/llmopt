"""Neuron-geometry plots for the micro-model program (docs/assets).

Each dot is one neuron (a row of a projection matrix), color = neuron
magnitude. Default layer: the mid-block `gate` projections (the
crystal layer). Three projection spaces:

  pca     — global linear axes (the original crystal view)
  sphere  — neurons unit-normalized, stereographic projection of the
            top-3-PC directions: pure DIRECTION structure, magnitude
            moved entirely into color (answers "is the lattice about
            angles or lengths")
  polar   — PC1+i*PC2 read as a complex number: angle = phase around
            the dominant plane, radius = neuron magnitude (Artin's
            mapped-in-complex-space ask, 2026-07-17)

Examples:
  python scripts/plot_neurons.py --ckpt checkpoints/mathnative_19m.pt \
      --d 384 --title "math-native 19M" --out docs/assets/gallery/neurons-19m.png
  python scripts/plot_neurons.py --ckpt A.pt --compare B.pt \
      --method sphere --out docs/assets/gallery/neurons-a-vs-b.png

Provenance (2026-08-08 gallery-hardening pass): every figure gets a
footer stamp — checkpoint basename + sha256[:8] + the repo HEAD at
render time — so a stranger can tie the pixels to exact artifacts
(figures are claims; the citation policy applies to them too).
--normalize divides each panel's magnitudes by that panel's MEDIAN
before plotting: cross-substrate comparisons (fp32 v ternary v a
671B expert shard) claim SHAPE similarity, so the radial axis must
be commensurable — raw scales differ per alphabet and would fake
or hide texture agreement.
"""
import argparse
import hashlib
import subprocess

BG = "#0d1117"
FG = "#c9d1d9"
DIM = "#8b949e"


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


def provenance_line(ckpts) -> str:
    parts = [f"{c.split('/')[-1]} {sha8(c)}" for c in ckpts]
    return (f"{' | '.join(parts)} · plot_neurons.py @ {repo_head()}")


def torch_svd_top2(X):
    import torch
    _, S, V = torch.linalg.svd(X, full_matrices=False)
    return None, S, V[:2]


def neuron_matrix(ckpt: str, key_sub: str):
    import torch
    sd = torch.load(ckpt, map_location="cpu")
    mats = [W for k, W in sd.items() if key_sub in k]
    if not mats:
        raise SystemExit(f"no '{key_sub}' matrices in {ckpt}")
    mid = mats[len(mats) // 2]
    return mid.float()


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
        # stereographic from the south pole
        denom = 1 + p3[:, 2].clamp(min=-0.99)
        return p3[:, 0] / denom, p3[:, 1] / denom, mag
    if method == "polar":
        # WHITENED (2026-07-25 instrument fix, reviewer catch): SVD
        # orders axes by variance, so the raw PC1+i*PC2 angle is
        # non-uniform even for a perfectly isotropic cloud. Equalize
        # the two variances before taking the angle — only then does
        # "uniform ring" mean isotropy.
        X = W - W.mean(0)
        _, S, V = torch.linalg.svd(X, full_matrices=False)
        P = X @ V[:2].T
        P = P / S[:2].clamp(min=1e-12)
        z = torch.complex(P[:, 0], P[:, 1])
        return z.angle(), mag, mag
    raise SystemExit(f"unknown method {method}")


def scatter(ax, xs, ys, mag, title, cmap, vmin=None, vmax=None):
    ax.scatter(xs, ys, c=mag, cmap=cmap, s=6, alpha=0.85,
               linewidths=0, vmin=vmin, vmax=vmax)
    ax.set_title(title, color=FG, fontsize=11, family="monospace")
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("#30363d")
    ax.tick_params(colors="#484f58", labelsize=6)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--compare", default=None,
                    help="second checkpoint for a side-by-side")
    ap.add_argument("--key", default="gate.weight",
                    help="substring picking the projection family")
    ap.add_argument("--method", default="pca",
                    choices=["pca", "sphere", "polar"])
    ap.add_argument("--title", default=None)
    ap.add_argument("--title2", default=None)
    ap.add_argument("--cmap", default="cool")
    ap.add_argument("--out", required=True)
    ap.add_argument("--displace", default=None,
                    help="second checkpoint: draw the central lattice "
                         "with neuron displacement lines FROM --ckpt "
                         "TO this (the whisper-zoom view)")
    ap.add_argument("--mult", type=float, default=60,
                    help="displacement magnification")
    ap.add_argument("--zoom", type=float, default=0.2,
                    help="central quantile box (0.2 = middle 60%)")
    ap.add_argument("--normalize", action="store_true",
                    help="divide each panel's magnitudes by that "
                         "panel's median (cross-substrate SHAPE "
                         "comparisons need commensurable axes)")
    ap.add_argument("--foot", default=None,
                    help="left-panel footer (layer/count/diet receipt)")
    ap.add_argument("--foot2", default=None,
                    help="right-panel footer")
    a = ap.parse_args()
    if a.displace:
        import numpy as np
        from matplotlib.collections import LineCollection
        W0 = neuron_matrix(a.ckpt, a.key)
        W1 = neuron_matrix(a.displace, a.key)
        X = W0 - W0.mean(0)
        _, _, V = torch_svd_top2(X)
        P0 = (X @ V.T).numpy()
        P1 = ((W1 - W0.mean(0)) @ V.T).numpy()
        disp = P1 - P0
        end = P0 + disp * a.mult
        lo = np.quantile(P0, a.zoom, axis=0)
        hi = np.quantile(P0, 1 - a.zoom, axis=0)
        m = ((P0[:, 0] > lo[0]) & (P0[:, 0] < hi[0]) &
             (P0[:, 1] > lo[1]) & (P0[:, 1] < hi[1]))
        fig, ax = plt.subplots(figsize=(12, 12))
        fig.patch.set_facecolor(BG)
        segs = np.stack([P0[m], end[m]], axis=1)
        lc = LineCollection(segs, cmap="cool", linewidths=0.7,
                            alpha=0.8)
        lc.set_array(np.linalg.norm(disp[m], axis=1))
        ax.add_collection(lc)
        ax.scatter(P0[m, 0], P0[m, 1], s=2, c="#30363d")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([])
        t = a.title or f"{a.ckpt} -> {a.displace}"
        ax.set_title(f"central lattice, displacements x{a.mult:g} — {t}",
                     color=FG, fontsize=11, family="monospace")
        fig.text(0.01, 0.005, provenance_line([a.ckpt, a.displace]),
                 color=DIM, fontsize=6.5, family="monospace")
        fig.tight_layout(rect=(0, 0.015, 1, 1))
        fig.savefig(a.out, dpi=150, facecolor=BG)
        print(f"saved {a.out}")
        raise SystemExit

    ckpts = [(a.ckpt, a.title or a.ckpt)]
    if a.compare:
        ckpts.append((a.compare, a.title2 or a.compare))
    fig, axes = plt.subplots(1, len(ckpts),
                             figsize=(9 * len(ckpts), 8.5))
    fig.patch.set_facecolor(BG)
    axes = axes if len(ckpts) > 1 else [axes]
    # shared color scale across compare panels (2026-07-25 fix:
    # per-panel norms made the same color mean different magnitudes)
    panels = []
    for ck, title in ckpts:
        W = neuron_matrix(ck, a.key)
        xs, ys, mag = project(W, a.method)
        if a.normalize:
            med = mag.median().clamp(min=1e-12)
            mag = mag / med
            if a.method == "polar":
                ys = ys / med  # polar's y IS the magnitude axis
        panels.append((xs, ys, mag, title))
    vmin = min(float(p[2].min()) for p in panels)
    vmax = max(float(p[2].max()) for p in panels)
    ylo = yhi = None
    if a.method == "polar" and a.normalize:
        # shared y-limits: SHAPE claims need commensurable axes
        ylo = min(float(p[1].min()) for p in panels)
        yhi = max(float(p[1].max()) for p in panels)
    feet = [a.foot, a.foot2]
    for i, (ax, (xs, ys, mag, title)) in enumerate(zip(axes, panels)):
        scatter(ax, xs, ys, mag, title, a.cmap, vmin=vmin, vmax=vmax)
        if a.method == "polar":
            ax.set_xlabel("phase (rad) of PC1 + i*PC2", color=FG,
                          fontsize=8, family="monospace")
            ax.set_ylabel("neuron magnitude"
                          + (" / panel median" if a.normalize else ""),
                          color=FG, fontsize=8, family="monospace")
            if ylo is not None and yhi is not None:
                ax.set_ylim(ylo * 0.98, yhi * 1.02)
        elif a.method == "pca":
            ax.set_xlabel("PC1", color=FG, fontsize=8,
                          family="monospace")
            ax.set_ylabel("PC2", color=FG, fontsize=8,
                          family="monospace")
        if i < len(feet) and feet[i]:
            ax.text(0.5, -0.09, feet[i], transform=ax.transAxes,
                    color=DIM, fontsize=7.5, family="monospace",
                    ha="center")
    fig.suptitle(f"each dot a neuron ({a.key}, {a.method}"
                 + (", per-panel median-normalized" if a.normalize
                    else "")
                 + "), color = magnitude", color=FG, fontsize=12,
                 family="monospace")
    fig.text(0.01, 0.005, provenance_line([c for c, _ in ckpts]),
             color=DIM, fontsize=6.5, family="monospace")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(a.out, dpi=150, facecolor=BG)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
