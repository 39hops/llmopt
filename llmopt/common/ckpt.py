"""Checkpoint IO. weights_only=True is the default and the audit
point; the few legitimate weights_only=False loads stay explicit
torch.load calls at their sites."""
from __future__ import annotations


def load_ckpt(path, map_location="cpu", weights_only=True):
    import torch
    return torch.load(path, map_location=map_location,
                      weights_only=weights_only)
