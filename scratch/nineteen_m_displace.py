"""The 19M in the crystal's displacement style (Artin's ask
2026-08-08). No training pair exists on disk until the snap birth
lands, but the rational-snap family gives a real displacement of
one 19M: snap19m_q32 (~base to ~1e-3, the Q=64/32 parity regime)
-> snap19m_q4 (the cracked regime). Panel = "what 1/Q^2-coarse
quantization moves", drawn with the crystal recipe: PCA plane of
the base-proxy, displacement segments, inferno, mean-disp footer.

Usage: .venv/bin/python scratch/nineteen_m_displace.py
Out:   figs/2026-08-08/nineteen-m-quant-displace.png
"""
import sys

sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

BG = "#0d1117"
K = "blocks.4.gate.weight"
W0 = torch.load("checkpoints/snap19m_q32.pt", map_location="cpu",
                weights_only=True)[K].float()
W1 = torch.load("checkpoints/snap19m_q4.pt", map_location="cpu",
                weights_only=True)[K].float()

X = W0 - W0.mean(0)
_, _, V = torch.linalg.svd(X, full_matrices=False)
P0 = (X @ V[:2].T).numpy()
disp = ((W1 - W0) @ V[:2].T).numpy()
mag = (W1 - W0).norm(dim=1).numpy()

fig, ax = plt.subplots(figsize=(12, 12))
fig.patch.set_facecolor(BG)
segs = np.stack([P0, P0 + disp * 6], axis=1)
lc = LineCollection(segs, cmap="inferno", linewidths=0.7, alpha=0.9)
lc.set_array(mag)
ax.add_collection(lc)
lo = np.quantile(P0, 0.001, 0); hi = np.quantile(P0, 0.999, 0)
ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
ax.set_facecolor(BG); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("math-native 19M, mid gate — what Q=4 snapping moves "
             f"(x6)\nmean disp {mag.mean():.4f} | 1,536 neurons, "
             "384-dim", color="#e6edf3", family="monospace",
             fontsize=13)
fig.tight_layout()
fig.savefig("figs/2026-08-08/nineteen-m-quant-displace.png",
            dpi=130, facecolor=BG)
print(f"saved; mean disp {mag.mean():.4f}")
