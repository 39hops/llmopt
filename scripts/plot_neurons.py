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
      --d 384 --title "math-native 19M" --out docs/assets/neurons-19m.png
  python scripts/plot_neurons.py --ckpt A.pt --compare B.pt \
      --method sphere --out docs/assets/neurons-a-vs-b.png
"""
import argparse

BG = "#0d1117"
FG = "#c9d1d9"


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
        X = W - W.mean(0)
        _, _, V = torch.linalg.svd(X, full_matrices=False)
        P = X @ V[:2].T
        z = torch.complex(P[:, 0], P[:, 1])
        return z.angle(), mag, mag
    raise SystemExit(f"unknown method {method}")


def scatter(ax, xs, ys, mag, title, cmap):
    ax.scatter(xs, ys, c=mag, cmap=cmap, s=6, alpha=0.85,
               linewidths=0)
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
        fig.tight_layout()
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
    for ax, (ck, title) in zip(axes, ckpts):
        W = neuron_matrix(ck, a.key)
        xs, ys, mag = project(W, a.method)
        scatter(ax, xs, ys, mag, title, a.cmap)
        if a.method == "polar":
            ax.set_xlabel("phase (rad) of PC1 + i*PC2", color=FG,
                          fontsize=8, family="monospace")
            ax.set_ylabel("neuron magnitude", color=FG, fontsize=8,
                          family="monospace")
    fig.suptitle(f"each dot a neuron ({a.key}, {a.method}), "
                 f"color = magnitude", color=FG, fontsize=12,
                 family="monospace")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, facecolor=BG)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
