"""Task vectors from LoRA adapters: skill = weight delta, applied by
arithmetic.

A LoRA adapter (train/lora.py) IS a low-rank task vector:
ΔW = (alpha/r) · B @ A per wrapped Linear. Applying it at scale λ to a
plain (unwrapped) model merges λ·ΔW into the Linear's weight in-place.
λ = 1 reproduces the fine-tune, λ = −1 negates the skill, two adapters
applied in sequence compose. undo() subtracts the exact delta that was
added, restoring the original weights bit-identically (the delta is
kept, not recomputed).
"""

from __future__ import annotations

from pathlib import Path

import torch


def load_adapter(path: str | Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Read the {module_path}.a / {module_path}.b flat dict saved by
    scripts/train_calculus.py into {module_path: (A, B)}."""
    flat = torch.load(path, map_location="cpu", weights_only=True)
    out: dict[str, tuple] = {}
    for key, tensor in flat.items():
        stem, leaf = key.rsplit(".", 1)
        a, b = out.get(stem, (None, None))
        out[stem] = (tensor, b) if leaf == "a" else (a, tensor)
    missing = [k for k, (a, b) in out.items() if a is None or b is None]
    assert not missing, f"unpaired adapter tensors: {missing}"
    return out


@torch.no_grad()
def apply_task_vector(
    model, adapter: dict, scale: float, *, r: int = 16, alpha: float = 32.0
):
    """Merge scale·(alpha/r)·B@A into each named Linear's weight.

    Module paths are resolved with get_submodule, so an adapter saved
    from a LoRA-wrapped model applies cleanly to the plain model (the
    wrapped path "...q_proj" is the plain Linear's path). Returns
    undo() restoring the exact previous weights.
    """
    # keep the ORIGINAL weights, not the delta: (w + d) - d is not
    # bit-identical to w under float rounding; copying back is.
    originals: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    for path, (a, b) in adapter.items():
        module = model.get_submodule(path)
        w = module.weight
        originals.append((w, w.detach().clone()))
        delta = (scale * alpha / r) * (
            b.to(w.device, torch.float32) @ a.to(w.device, torch.float32)
        )
        w += delta.to(w.dtype)

    def undo():
        with torch.no_grad():
            for w, orig in originals:
                w.copy_(orig)

    return undo
