"""LoRA family: low-rank adapters on frozen linears, plus DoRA.

LoRA: W x + (alpha/r) B A x with A [r, in], B [out, r], B zero-init so
training starts at the base model exactly. Only A, B train — optimizer
state shrinks by ~10^3, and merge() folds the adapter back so inference
pays zero overhead.

DoRA decomposes W into magnitude * direction and applies LoRA to the
direction only, training the per-column magnitude separately — closes
most of the LoRA/full-FT gap at the same rank.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dev, dt = base.weight.device, base.weight.dtype
        self.a = nn.Parameter(
            torch.randn(r, base.in_features, device=dev, dtype=dt) / r**0.5
        )
        self.b = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.a.T @ self.b.T) * self.scaling

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Fold the adapter into a plain Linear (inference: zero overhead)."""
        merged = nn.Linear(
            self.base.in_features, self.base.out_features,
            bias=self.base.bias is not None,
        )
        merged.weight.copy_(self.base.weight + (self.b @ self.a) * self.scaling)
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged


class DoRALinear(nn.Module):
    """W = m * (W0 + BA)/||W0 + BA||_col — LoRA on direction, learned
    per-output magnitude m."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dev, dt = base.weight.device, base.weight.dtype
        self.a = nn.Parameter(
            torch.randn(r, base.in_features, device=dev, dtype=dt) / r**0.5
        )
        self.b = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        self.scaling = alpha / r
        self.magnitude = nn.Parameter(base.weight.norm(dim=1).clone())

    def forward(self, x):
        w = self.base.weight + (self.b @ self.a) * self.scaling
        w = self.magnitude[:, None] * w / w.norm(dim=1, keepdim=True)
        out = x @ w.T
        return out + self.base.bias if self.base.bias is not None else out


def apply_lora(model, target_names, *, r=8, alpha=16.0, cls=LoRALinear):
    """Wrap every nn.Linear whose qualified name contains any of
    target_names (e.g. ('q_proj', 'v_proj')). Freezes the whole base
    model; only adapter params train. Returns wrapped count."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(
                t in full for t in target_names
            ):
                setattr(module, child_name, cls(child, r=r, alpha=alpha))
                wrapped += 1
    return wrapped


def trainable_fraction(model) -> float:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return train / total
