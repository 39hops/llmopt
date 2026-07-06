"""Rotation vs RTN quantization error (spec 2026-07-06, part a).

Grid: {real Qwen2.5-0.5B layers, synthetic controls} x bits {4,3,2} x
rotation {none, hadamard, random}. Relative Frobenius error; lower is
better. Controls: iid Gaussian (rotation should do nothing) and
planted-outlier (rotation should clearly win) bracket the mechanism.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from llmopt.quantize.rotate import hadamard, random_orthogonal, rotation_error

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def real_layers():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    sd = model.state_dict()
    for layer in (0, 12, 23):
        for proj in ("self_attn.q_proj", "mlp.down_proj"):
            name = f"model.layers.{layer}.{proj}.weight"
            yield f"L{layer}.{proj.split('.')[-1]}", sd[name]


def synthetic():
    torch.manual_seed(0)
    g = torch.randn(1024, 1024)
    out = torch.randn(1024, 1024)
    out[:, :16] *= 30.0
    yield "iid-gaussian", g
    yield "outlier-cols", out


def pad_pow2(w):
    """Column-pad to the next power of 2 so hadamard applies; padding
    columns are zero and cannot change relative error rankings."""
    n = w.shape[1]
    m = 1 << (n - 1).bit_length()
    if m == n:
        return w
    return torch.cat([w, torch.zeros(w.shape[0], m - n)], dim=1)


def main() -> None:
    rows = list(synthetic()) + list(real_layers())
    print(f"{'matrix':18s} {'shape':14s}" + "".join(
        f"  {b}b none  {b}b hadam {b}b rand " for b in (4, 3, 2)))
    for name, w in rows:
        wp = pad_pow2(w.float())
        n = wp.shape[1]
        # seed=7 != the synthetic matrices' seed 0: random_orthogonal(QR
        # of a seeded Gaussian) built from the SAME draw as the test
        # matrix is correlated with it and produced garbage numbers once
        h, r = hadamard(n), random_orthogonal(n, seed=7)
        cells = []
        for bits in (4, 3, 2):
            for rot in (None, h, r):
                cells.append(f"{rotation_error(wp, bits, rot):8.4f}")
        print(f"{name:18s} {str(tuple(w.shape)):14s}" + " ".join(cells))


if __name__ == "__main__":
    main()
