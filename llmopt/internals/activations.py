"""Activation statistics per layer — the numbers quantization cares
about: RMS scale, kurtosis (outlier-heaviness; fp formats hate heavy
tails), and outlier fraction beyond k sigma. Captured with forward
hooks on each decoder layer's output residual stream.
"""

from __future__ import annotations


def activation_stats(model, ids, *, outlier_sigma: float = 6.0):
    """Returns per-layer dicts: rms, std, kurtosis (Fisher, 0 for
    normal), outlier_frac (|x - mean| > outlier_sigma * std)."""
    import torch

    captured: list[torch.Tensor] = []
    hooks = [
        layer.register_forward_hook(
            lambda mod, args, out: captured.append(
                (out[0] if isinstance(out, tuple) else out).detach()
            )
        )
        for layer in model.model.layers
    ]
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([list(ids)], device=model.device))
    finally:
        for h in hooks:
            h.remove()

    stats = []
    for x in captured:
        x = x.float().flatten()
        mean, std = x.mean(), x.std()
        z = (x - mean) / (std + 1e-12)
        stats.append(
            {
                "rms": float(x.pow(2).mean().sqrt()),
                "std": float(std),
                "kurtosis": float(z.pow(4).mean() - 3.0),
                "outlier_frac": float((z.abs() > outlier_sigma).float().mean()),
            }
        )
    return stats
