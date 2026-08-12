"""G5 POLAR (pre-reg 2026-07-29 eve): the predicted BREAK of
geometry-blindness. cplx_G5_dep.pt carries DEPLOYED star weights
({0, +-s, +-is} per complex — anisotropic by construction);
cplx_none.pt is the isotropic control crystal. Cells per crystal:
control; polar 4 angles ALIGNED (1s mag); polar 4 angles ROTATED
45 deg (same bits); uniform u=1s. Prediction: rotation hurts ONLY
the star crystal. alpha=none for both (dep weights are already
hard). __main__-guarded.
"""
from llmopt.common.device import pick_device
import math
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

import complex_model as C  # noqa: E402
import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer  # noqa: E402

D, LAYERS, FFN, HEADS = 384, 8, 1536, 6
KEYS = [f"blocks.{li}.{nm}.weight" for li in range(LAYERS)
        for nm in ("gate", "up")]


def polar_q(W, mstep, na, rot=0.0):
    h = W.shape[0] // 2
    wr, wi = W[:h].float(), W[h:].float()
    s = mstep * W.float().std()
    m = torch.sqrt(wr * wr + wi * wi)
    th = torch.atan2(wi, wr) - rot
    mq = torch.round(m / s) * s
    thq = torch.round(th / (2 * math.pi / na)) \
        * (2 * math.pi / na) + rot
    return torch.cat([mq * torch.cos(thq), mq * torch.sin(thq)])


def uni_q(W, u):
    s = u * W.float().std()
    return torch.round(W.float() / s) * s


def main():
    tok = MathTokenizer()
    dev = pick_device()
    C.set_alpha("none")

    def gate(sd, tag):
        m = C.build_complex_model(len(tok.vocab), d=D, layers=LAYERS,
                                  heads=HEADS, ffn=FFN).to(dev)
        m.load_state_dict(sd)
        m.eval()
        with torch.no_grad():
            solves, valid = G.gate_eval(m, tok, dev)
        print(f"G5-POLAR {tag}: {sum(solves.values())}/120 "
              f"@ {valid:.2f}%", flush=True)
        del m

    for name, ckpt in (("STAR", "checkpoints/cplx_G5_dep.pt"),
                       ("ISO", "checkpoints/cplx_none.pt")):
        base = torch.load(ckpt, map_location="cpu",
                          weights_only=True)
        gate(base, f"{name} control")
        for tag, fn in (
                ("polar4 aligned",
                 lambda W: polar_q(W, 1.0, 4)),
                ("polar4 rot45",
                 lambda W: polar_q(W, 1.0, 4, rot=math.pi / 4)),
                ("uniform 1s", lambda W: uni_q(W, 1.0))):
            sd = dict(base)
            for k in KEYS:
                sd[k] = fn(base[k])
            gate(sd, f"{name} {tag}")


if __name__ == "__main__":
    main()
