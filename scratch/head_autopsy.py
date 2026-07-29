"""THE HEAD AUTOPSY (pre-reg 2026-07-29 eve): per-(layer, head)
single-cell deletion map on the h8 EMA crystal. The day census
deleted a head INDEX across all layers (a column); this deletes
one (layer, head) cell at a time — 64 cells on the proxy gate
(n=8/level, +-2 noise, read the map shape), then FULL gates on
control + min/max cells. __main__-guarded.
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

CKPT = "checkpoints/sym_birth_dense_mps_h8_ema.pt"
D, LAYERS, FFN, HEADS = 64, 8, 256, 8
HD = D // HEADS


def main():
    tok = MathTokenizer()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    base = torch.load(CKPT, map_location="cpu", weights_only=True)

    def gate(sd, label, n=None):
        m = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN).to(dev)
        m.load_state_dict({k: v.to(dev) for k, v in sd.items()})
        m.eval()
        with torch.no_grad():
            solves, valid = G.gate_eval(m, tok, dev, n=n)
        tot = sum(solves.values())
        print(f"AUTOPSY {label}: {tot}/{(n or 24) * 5} "
              f"@ {valid:.2f}%", flush=True)
        del m
        return tot

    def drop(li, h):
        sd = {k: v.clone() for k, v in base.items()}
        a, b = h * HD, (h + 1) * HD
        W = sd[f"blocks.{li}.qkv.weight"]
        for part in range(3):
            W[part * D + a:part * D + b, :] = 0
        sd[f"blocks.{li}.o.weight"][:, a:b] = 0
        return sd

    ctrl = gate(base, "control (proxy)", n=8)
    grid = {}
    for li in range(LAYERS):
        for h in range(HEADS):
            grid[(li, h)] = gate(drop(li, h), f"L{li}h{h}", n=8)
    lo = min(grid, key=grid.get)
    hi = max(grid, key=grid.get)
    print(f"MAP: ctrl {ctrl} | min {lo}={grid[lo]} | "
          f"max {hi}={grid[hi]} | "
          f"cells<=ctrl-3: {sum(v <= ctrl - 3 for v in grid.values())}"
          f"/64", flush=True)
    gate(base, "control (FULL)")
    gate(drop(*lo), f"min L{lo[0]}h{lo[1]} (FULL)")
    gate(drop(*hi), f"max L{hi[0]}h{hi[1]} (FULL)")


if __name__ == "__main__":
    main()
