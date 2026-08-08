"""Provenance falsification test for docs/assets/neurons-19m.png's
RIGHT panel (Artin's ask 2026-08-08): recreate it two ways and let
the pixels decide.

Panel 1: SFT displacement (step_lora BA x2 alpha/r) — the frozen
         left panel's recipe, for calibration.
Panel 2: RL displacement (grpo minus pre_grpo_backup, x2) on the
         SAME PCA plane — the commit's claim for the right panel.
Panel 3: the 19M mid gate (blocks.4) drawn AT QWEN'S SCALE (same
         axis extents, same color normalization) — Artin's
         hypothesis for the right panel.

Whichever of 2/3 matches the frozen right panel is the vehicle.
Committed per the generator-loss rule.

Usage: .venv/bin/python scratch/crystal_recreate_test.py
Out:   figs/2026-08-08/crystal-recreate-test.png
"""
import sys

sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BG = "#0d1117"
K = "model.layers.14.mlp.gate_proj"
SCALE = 2.0  # lora alpha/r

base = torch.load("checkpoints/qwen05b_base_l14gate.pt",
                  map_location="cpu", weights_only=True)
W0 = list(base.values())[0].float()


def lora_delta(p):
    sd = torch.load(p, map_location="cpu", weights_only=True)
    return (sd[K + ".b"].float() @ sd[K + ".a"].float()) * SCALE


d_sft = lora_delta("checkpoints/step_lora.pt")
d_rl = (lora_delta("checkpoints/step_lora_grpo.pt")
        - lora_delta("checkpoints/step_lora_pre_grpo_backup.pt"))

X = W0 - W0.mean(0)
_, _, V = torch.linalg.svd(X, full_matrices=False)
P0 = (X @ V[:2].T).numpy()

m19 = torch.load("checkpoints/mathnative_19m_gen8.pt",
                 map_location="cpu", weights_only=True)
W19 = m19["blocks.4.gate.weight"].float()
X19 = W19 - W19.mean(0)
_, _, V19 = torch.linalg.svd(X19, full_matrices=False)
P19 = (X19 @ V19[:2].T).numpy()

fig, axes = plt.subplots(1, 3, figsize=(27, 9))
fig.patch.set_facecolor(BG)
lims = (np.quantile(P0, 0.001, 0), np.quantile(P0, 0.999, 0))
vmax = float(d_sft.norm(dim=1).max())

for ax, (title, dele) in zip(
        axes[:2],
        [("recreated SFT panel (x6)", d_sft),
         ("recreated RL panel (x6) — the commit's claim", d_rl)]):
    disp = (dele @ V[:2].T).numpy()
    mag = dele.norm(dim=1).numpy()
    from matplotlib.collections import LineCollection
    segs = np.stack([P0, P0 + disp * 6], axis=1)
    lc = LineCollection(segs, cmap="inferno", linewidths=0.6,
                        alpha=0.9)
    lc.set_array(mag)
    lc.set_clim(0, vmax)
    ax.add_collection(lc)
    ax.set_title(f"{title}\nmean disp {mag.mean():.3f}",
                 color="#e6edf3", family="monospace", fontsize=13)

ax = axes[2]
mag19 = W19.norm(dim=1).numpy()
ax.scatter(P19[:, 0], P19[:, 1], s=6, c=mag19, cmap="inferno",
           vmin=0, vmax=vmax)
ax.set_title("19M mid gate AT QWEN'S SCALE (axes+colors matched)\n"
             "— the alternative hypothesis",
             color="#e6edf3", family="monospace", fontsize=13)

for ax in axes:
    ax.set_facecolor(BG)
    ax.set_xlim(lims[0][0], lims[1][0])
    ax.set_ylim(lims[0][1], lims[1][1])
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("crystal right-panel provenance test — which one is "
             "the frozen near-black panel?", color="#e6edf3",
             family="monospace", fontsize=15)
fig.tight_layout()
fig.savefig("figs/2026-08-08/crystal-recreate-test.png", dpi=110,
            facecolor=BG)
print("saved figs/2026-08-08/crystal-recreate-test.png")
print(f"SFT mean disp {d_sft.norm(dim=1).mean():.4f} | "
      f"RL mean disp {d_rl.norm(dim=1).mean():.4f}")
