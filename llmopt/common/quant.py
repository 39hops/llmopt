"""Weight quantization transforms (spec 2026-08-12 §4.2). ternary
adopted verbatim from scripts/train_ternary.py; the four scratch
copies (confluence, ternary_gate, absorb_1e5, ternary_control) are
the same body minus type hints."""
from __future__ import annotations

import torch


def ternary(w: torch.Tensor) -> torch.Tensor:
    s = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    return torch.where(w.abs() < 0.5 * s,
                       torch.zeros_like(w), torch.sign(w) * s)
