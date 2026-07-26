"""Margin census on the crown-tie ternary latents (pre-reg 2026-07-26).

Per-weight decision margin under the absmean ternary quantizer:
scale s = mean|w| per row; thresholds at 0.5*s. margin = distance of
|w| to the nearest threshold, in units of s. Near-threshold =
margin < 0.05 (undecided mass). Question: does undecided mass
cluster by neuron/layer, or is it uniform (democracy reading)?
"""
import sys

import torch

PATH = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/merged_grown_latent.pt"
NEAR = 0.05

sd = torch.load(PATH, map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "model" in sd:
    sd = sd["model"]

rows = []
for k, w in sd.items():
    if w.ndim != 2:
        continue
    if "emb" in k or "head" in k:
        continue
    w = w.float()
    s = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    a = w.abs() / s                      # in scale units
    margin = (a - 0.5).abs()             # distance to the flip threshold
    frac = (margin < NEAR).float().mean(dim=1)   # per-neuron undecided frac
    rows.append((k, frac))

per_neuron = torch.cat([f for _, f in rows])
mean = per_neuron.mean().item()
cv = (per_neuron.std() / per_neuron.mean()).item()
print(f"matrices: {len(rows)}  neurons: {per_neuron.numel()}")
print(f"near-threshold fraction: mean {mean:.4f}  per-neuron CV {cv:.3f}")
print(f"per-neuron quantiles: "
      f"{[round(torch.quantile(per_neuron, q).item(), 4) for q in (0.01, 0.25, 0.5, 0.75, 0.99)]}")

by_layer = {}
for k, f in rows:
    if k.startswith("blocks."):
        li = int(k.split(".")[1])
        by_layer.setdefault(li, []).append(f)
print("per-layer mean near-threshold fraction:")
for li in sorted(by_layer):
    lf = torch.cat(by_layer[li])
    print(f"  layer {li:2d}: {lf.mean().item():.4f}")

by_mat = sorted(((torch.cat([f]).mean().item(), k) for k, f in rows), reverse=True)
print("top/bottom matrices by undecided fraction:")
for m, k in by_mat[:3] + by_mat[-3:]:
    print(f"  {m:.4f}  {k}")
