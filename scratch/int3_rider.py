"""Rung-1 rider (iv): int3 PTQ gate delta, prefix vs infix twin
(spec 2026-07-25-native-transformer — prediction: prefix MORE
robust under quantization via delimiter-outlier removal).

Per-output-channel symmetric absmax int3 (levels -3..3) on every
2-D weight; norms/biases/1-D untouched. Same scheme both arms.
Writes <ckpt>_int3.pt next to the source; gates run separately.

    .venv/bin/python scratch/int3_rider.py <ckpt> [<ckpt> ...]
"""
import sys

import torch

for path in sys.argv[1:]:
    sd = torch.load(path, map_location="cpu")
    n2d = 0
    for k, w in sd.items():
        if w.ndim == 2:
            s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 3.0
            sd[k] = (torch.round(w / s).clamp(-3, 3)) * s
            n2d += 1
    out = path.replace(".pt", "_int3.pt")
    torch.save(sd, out)
    print(f"{out}: {n2d} matrices quantized int3", flush=True)
