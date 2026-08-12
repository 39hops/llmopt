"""Default-accelerator resolution. PLACEMENT is torch-native
(.to(dev)); deliberate CPU islands (seeded samplers, oracles) are
correctness pins and never route through here. No ambient default:
device stays an explicit per-call value because instrument sigma
never transports across devices."""
from __future__ import annotations

import os


def pick_device(override: str | None = None) -> str:
    """explicit arg > LLMOPT_DEVICE env > cuda > mps > cpu."""
    if override:
        return override
    env = os.environ.get("LLMOPT_DEVICE")
    if env:
        return env
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
