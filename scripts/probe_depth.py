"""Depth anatomy: WHERE in the stack does the rewrite decision form?

Logit-lens over the micro-model: run gate-band prompts, project every
layer's residual through the model's own final norm + head, and
measure per-layer agreement with the final output (argmax match rate
over answer positions). If the decision forms early, the top layers
are spelling, not thinking — early-exit sampling (self-speculative)
would cut mining/GRPO wall (the speed-first lever from Artin's
B-tree riff, 2026-07-17).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmopt.train.mathnative import MathTokenizer, build_model
from step_grpo_micro import GATE_BAND, GATE_LEVELS, GATE_N


def main(ckpt: str, d: int, layers: int, ffn: int, heads: int,
         n: int) -> None:
    import sympy as sp
    import torch

    from bench_step_tokens import _gen_isolated

    tok = MathTokenizer()
    dev = ("mps" if torch.backends.mps.is_available() else
           "cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(tok.vocab), d=d, layers=layers,
                        heads=heads, ffn=ffn).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(dev).eval()

    agree = torch.zeros(layers, device=dev)
    total = 0
    with torch.no_grad():
        for lv in GATE_LEVELS:
            for i in range(n):
                p = _gen_isolated(lv, GATE_BAND + 1000 * lv + i)
                if p is None:
                    continue
                cur = f"Integral({sp.sstr(p._expr)}, x)"
                ids = torch.tensor([tok.encode(
                    f"Current: {cur}\nHints: none\nStep: {cur}\n")],
                    device=dev)
                x = model.emb(ids)
                per_layer = []
                for b in model.blocks:
                    x = b(x, None)
                    per_layer.append(
                        model.head(model.norm(x)).argmax(-1))
                final = per_layer[-1]
                # answer region only: positions after "Step: "
                start = ids.shape[1] - len(tok.encode(f"{cur}\n"))
                for li, pl in enumerate(per_layer):
                    agree[li] += (pl[0, start:] ==
                                  final[0, start:]).float().mean()
                total += 1
    for li in range(layers):
        bar = "#" * int(40 * agree[li] / total)
        print(f"layer {li:2d}: {100 * agree[li] / total:5.1f}% {bar}",
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--ffn", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--n", type=int, default=GATE_N)
    a = ap.parse_args()
    main(a.ckpt, a.d, a.layers, a.ffn, a.heads, a.n)
