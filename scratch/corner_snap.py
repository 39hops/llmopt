"""The compression corner (pre-reg 2026-07-28 night): rational-
snap (direct, exact-best p/q, q <= Q) x {dense wfloor d256,
circulant-8x substrate}, Q in {8, 16}. Paired gates on one
device. Delta-of-deltas reads orthogonality of the bits and
sharing compression axes. Snap code inlined from
scratch/rational_snap.py (same operator, no subprocess).
"""
from llmopt.common.device import pick_device
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

SUBSTRATES = {
    "dense65": "checkpoints/mathnative_wfloor_d256.pt",
    "circ59": "checkpoints/sym_circ8_b.pt",
}


def snap_sd(sd, Q):
    out = {}
    for k, w in sd.items():
        if w.ndim != 2 or not w.is_floating_point():
            out[k] = w
            continue
        wf = w.float()
        best = torch.round(wf)
        err = (wf - best).abs()
        for q in range(2, Q + 1):
            cand = torch.round(wf * q) / q
            e = (wf - cand).abs()
            m = e < err
            best = torch.where(m, cand, best)
            err = torch.where(m, e, err)
        out[k] = best.to(w.dtype)
    return out


tok = MathTokenizer()
dev = pick_device()
for name, path in SUBSTRATES.items():
    base = torch.load(path, map_location="cpu", weights_only=True)
    for Q in (8, 16):
        sd = snap_sd(base, Q)
        m = build_model(len(tok.vocab), d=256, layers=8, heads=4,
                       ffn=1024).to(dev)
        m.load_state_dict({k: v.to(dev) for k, v in sd.items()})
        m.eval()
        with torch.no_grad():
            solves, valid = G.gate_eval(m, tok, dev)
        print(f"CORNER {name} Q{Q}: {solves} = "
              f"{sum(solves.values())}/120 @ {valid:.2f}%",
              flush=True)
        del m
