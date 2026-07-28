"""Symmetry ladder S2 (pre-reg 2026-07-28): complexification
control. Double wfloor d256 -> d512 by W(+)W on every linear
(block layout); the doubled gates commute with J_half by theorem
(asserted). Same function in real arithmetic; fp last-bit ties
permitted per amended bar. No training — pure control gate.
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402
from rot_commutant import J_half, anti_mass  # noqa: E402

D, LAYERS, FFN, HEADS = 256, 8, 1024, 4


def blockdiag(W):
    o, i = W.shape
    M = torch.zeros(2 * o, 2 * i)
    M[:o, :i] = W
    M[o:, i:] = W
    return M


base = torch.load("checkpoints/mathnative_wfloor_d256.pt",
                  map_location="cpu", weights_only=True)
sd = {}
for k, v in base.items():
    v = v.float()
    if k == "emb.weight":
        sd[k] = torch.cat([v, v], dim=1)
    elif k == "head.weight":
        sd[k] = torch.cat([v / 2, v / 2], dim=1)
    elif k.endswith("qkv.weight"):
        # preserve q/k/v segment order: block-diag per segment
        segs = [blockdiag(s) for s in v.chunk(3, 0)]
        sd[k] = torch.cat(segs, dim=0)
    elif v.ndim == 2:
        sd[k] = blockdiag(v)
    else:  # norm gains
        sd[k] = torch.cat([v, v], dim=0)

# theorem check: doubled gates commute with J_half exactly
for li in range(LAYERS):
    W = sd[f"blocks.{li}.gate.weight"]
    am = anti_mass(W, J_half(2 * FFN), J_half(2 * D))
    assert am < 1e-12, f"layer {li} anti-mass {am}"
print("theorem check: all 8 doubled gates anti-mass < 1e-12",
      flush=True)

tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(len(tok.vocab), d=2 * D, layers=LAYERS,
                    heads=2 * HEADS, ffn=2 * FFN).to(dev)
model.load_state_dict({k: v.to(dev) for k, v in sd.items()})
model.eval()
with torch.no_grad():
    solves, valid = G.gate_eval(model, tok, dev)
print(f"[S2] doubled-model gate {solves} = "
      f"{sum(solves.values())}/120 @ {valid:.2f}% (comparator 65)",
      flush=True)
