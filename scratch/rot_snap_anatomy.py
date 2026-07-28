"""Rotational snap R2 (pre-reg 2026-07-28): gate wfloor_d256 with
gate matrices projected toward the commutant — W - t*W_a under
adjacent pairing, t in {0.25, 0.5, 1.0}. Fence: gate matrices
only (attention/up/down untouched). Flips-probe fingerprint per t
rides (vs the unmodified model, teacher-forced argmax diff).
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402
from rot_commutant import J_perm  # noqa: E402

CKPT = "checkpoints/mathnative_wfloor_d256.pt"
D, LAYERS, FFN, HEADS = 256, 8, 1024, 4

tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
base = torch.load(CKPT, map_location="cpu", weights_only=True)
Jo = J_perm(FFN, list(range(FFN)))
Ji = J_perm(D, list(range(D)))


def project(sd, t):
    out = dict(sd)
    for li in range(LAYERS):
        k = f"blocks.{li}.gate.weight"
        W = sd[k].float()
        Wa = 0.5 * (W + Jo @ W @ Ji)
        out[k] = (W - t * Wa).to(sd[k].dtype)
    return out


@torch.no_grad()
def gate(sd, tag):
    m = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
    m.load_state_dict(sd)
    m.eval()
    solves, valid = G.gate_eval(m, tok, dev)
    print(f"{tag}: {solves} = {sum(solves.values())}/120 "
          f"@ {valid:.2f}%", flush=True)
    del m


gate(base, "t=0 (control)")
for t in (0.25, 0.5, 1.0):
    gate(project(base, t), f"t={t}")
