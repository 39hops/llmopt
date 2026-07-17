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
    a = ap.parse_args()

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
