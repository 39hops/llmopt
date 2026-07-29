"""d56 exact twin (pre-reg 2026-07-29 night): snap EVERY
floating tensor (incl. emb, head, 1D norm gains) to
best-rational Q<=16, gate, and report per-tensor-class snap
error in sigma units (blockwise-rule diagnostic). Desk, MPS.
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

D, LAYERS, FFN, HEADS, Q = 56, 8, 224, 4, 16

base = torch.load("checkpoints/sym_birth_dense_w56_ema.pt",
                  map_location="cpu", weights_only=True)


def snap(w, q_max):
    wf = w.float()
    best = torch.round(wf)
    err = (wf - best).abs()
    for q in range(2, q_max + 1):
        cand = torch.round(wf * q) / q
        e = (wf - cand).abs()
        m = e < err
        best = torch.where(m, cand, best)
        err = torch.where(m, e, err)
    return best.to(w.dtype)


sd = {}
for k, w in base.items():
    if w.is_floating_point():
        sd[k] = snap(w, Q)
        cls = k.split(".")[-2] if "." in k else k
        sig = float(w.float().std())
        derr = float((sd[k] - w).float().abs().max())
        if "norm" in k or k.endswith("g") or w.ndim == 1:
            print(f"  1D {k}: sigma {sig:.3f} max-err {derr:.4f} "
                  f"({derr/max(sig,1e-9):.2f} sigma)", flush=True)
    else:
        sd[k] = w

tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
m = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                ffn=FFN).to(dev)
m.load_state_dict({k: v.to(dev) for k, v in sd.items()})
m.eval()
with torch.no_grad():
    solves, valid = G.gate_eval(m, tok, dev)
print(f"EXACT-TWIN d56 Q={Q} full-model: {solves} = "
      f"{sum(solves.values())}/120 @ {valid:.2f}%", flush=True)

# artifact note: weights are the fp32 IMAGE of best-rational
# n/q, q<=Q — non-dyadic q (3,5,...) are not exact in binary
# fp, so the rational-of-record is the (n,q) TABLE, derivable
# deterministically by re-running snap() on this checkpoint's
# parent. Denominator census (dyadic fraction = exactly
# representable share):
tot = dy = 0
for k, v in sd.items():
    if not v.is_floating_point() or v.ndim != 2:
        continue
    w = v.float()
    tot += w.numel()
    dy += int(((w * 16) == (w * 16).round()).sum())  # q | 16
print(f"dyadic (q|16, exact-in-fp) share: {dy/tot:.3f}", flush=True)
torch.save(sd, "checkpoints/exact_twin_d56_q16.pt")
