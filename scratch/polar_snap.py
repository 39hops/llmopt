"""POLAR-SPLIT SNAP (pre-reg 2026-07-29 day: escalation-engine
cell 4). cplx_none.pt (unconstrained complex FFN, d384/f1536/h6):
quantize the complex gate/up weights |c| COARSE x arg(c) FINE
v uniform re/im grids, both expressed in sigma units (sigma law
rider), bits/complex MEASURED as log2(#distinct values used).
Fence: gate+up only (the complex-paired tensors); qkv/o/down
untouched. Desk, MPS.
"""
import math
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

import complex_model as C  # noqa: E402
import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer  # noqa: E402

CKPT = "checkpoints/cplx_none.pt"
D, LAYERS, FFN, HEADS = 384, 8, 1536, 6

tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
base = torch.load(CKPT, map_location="cpu", weights_only=True)
C.set_alpha("none")


def split(W):  # rows pair (re, im) along the ffn dim
    h = W.shape[0] // 2
    return W[:h], W[h:]


def snap_uniform(sd, u):  # step = u * sigma per tensor
    out, vals = dict(sd), 0.0
    for li in range(LAYERS):
        for nm in ("gate", "up"):
            k = f"blocks.{li}.{nm}.weight"
            W = sd[k].float()
            s = u * W.std()
            Wq = torch.round(W / s) * s
            out[k] = Wq
            vals += (math.log2(len(Wq[:W.shape[0] // 2].unique()))
                     + math.log2(len(Wq[W.shape[0] // 2:].unique())))
    return out, vals / (2 * LAYERS)  # mean bits/complex


def snap_polar(sd, mstep, na):  # |c| step = mstep*sigma; arg = na bins
    out, vals = dict(sd), 0.0
    for li in range(LAYERS):
        for nm in ("gate", "up"):
            k = f"blocks.{li}.{nm}.weight"
            W = sd[k].float()
            wr, wi = split(W)
            s = mstep * W.std()
            m = torch.sqrt(wr * wr + wi * wi)
            th = torch.atan2(wi, wr)
            mq = torch.round(m / s) * s
            thq = torch.round(th / (2 * math.pi / na)) \
                * (2 * math.pi / na)
            Wq = torch.cat([mq * torch.cos(thq), mq * torch.sin(thq)])
            out[k] = Wq
            pairs = torch.stack(
                [mq.flatten(), thq.flatten()]).T.unique(dim=0)
            vals += math.log2(len(pairs))
    return out, vals / (2 * LAYERS)


@torch.no_grad()
def gate(sd, tag, bits=None):
    m = C.build_complex_model(len(tok.vocab), d=D, layers=LAYERS,
                              heads=HEADS, ffn=FFN).to(dev)
    m.load_state_dict(sd)
    m.eval()
    solves, valid = G.gate_eval(m, tok, dev)
    b = f" | {bits:.2f} bits/cplx" if bits is not None else ""
    print(f"POLAR-SNAP {tag}: {sum(solves.values())}/120 "
          f"@ {valid:.2f}%{b}", flush=True)
    del m


gate(base, "t=0 control")
for u in (0.5, 1.0, 2.0):
    sd, b = snap_uniform(base, u)
    gate(sd, f"uniform u={u}sigma", b)
for mstep, na in ((1.0, 64), (1.0, 16), (2.0, 64), (0.5, 8)):
    sd, b = snap_polar(base, mstep, na)
    gate(sd, f"polar m={mstep}sigma x {na} angles", b)
