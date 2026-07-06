"""Wire llmopt Metal kernels into a loaded mlx-lm model.

Where stock mlx-lm already calls mx.fast.* (rms_norm, rope, SDPA) our
kernels only match or tie — swapping those buys nothing (see
kernels/metal.py measured numbers). The one unfused chain in stock
models is the MLP: ``down(silu(gate(x)) * up(x))`` launches separate
silu/multiply elementwise ops; the fused swiglu kernel does that in one
pass (2.2x on the elementwise chain at 4096x4096).

Patching is class-level (obj(x) dispatches through type(obj).__call__)
and reversible, same pattern as moe/prune.py.
"""

from __future__ import annotations

from llmopt.kernels.metal import swiglu


def patch_swiglu(model):
    """Swap the fused swiglu kernel into every MLP-shaped module of an
    mlx-lm model (modules with gate_proj/up_proj/down_proj). Returns
    (n_patched, unpatch)."""
    mlps = [
        m for _, m in model.named_modules()
        if all(hasattr(m, a) for a in ("gate_proj", "up_proj", "down_proj"))
    ]
    if not mlps:
        return 0, lambda: None
    classes = {type(m) for m in mlps}
    originals = {cls: cls.__call__ for cls in classes}

    def fused_call(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        return self.down_proj(swiglu(g, u))

    for cls in classes:
        cls.__call__ = fused_call

    def unpatch():
        for cls, orig in originals.items():
            cls.__call__ = orig

    return len(mlps), unpatch
