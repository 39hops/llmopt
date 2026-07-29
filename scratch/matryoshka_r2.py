"""Matryoshka rung 2 (pre-reg 2026-07-29 night): 3-tier ladder
in one tensor. Joint loss CE(W) + CE(STE P_C2(W)) +
CE(STE P_C8(W)) on gate weights, 1 warm epoch from the d56 EMA
crystal. Gates all three tiers. MPS.
"""
import os
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.nn.utils.parametrize as P  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from train_mathnative import load_rows  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

D, LAYERS, FFN, HEADS, BS = 56, 8, 224, 4, 8
CKPT = os.environ.get("CKPT", "checkpoints/sym_birth_dense_w56_ema.pt")
OUT = os.environ.get("OUT", "checkpoints/matryoshka_d56_3tier.pt")
TIER = {"nb": 0}  # 0 = dense; 2 = half; 8 = eighth


def shift_perm(n, nb, sh, dev):
    return torch.tensor([nb * (r // nb) + (r % nb - sh) % nb
                         for r in range(n)], device=dev)


class TierP(torch.nn.Module):
    def __init__(self, n_out, n_in, dev):
        super().__init__()
        self.perms = {}
        for nb in (2, 8):
            self.perms[nb] = (
                [shift_perm(n_out, nb, s, dev) for s in range(nb)],
                [shift_perm(n_in, nb, s, dev) for s in range(nb)])

    def project(self, W, nb):
        po, pi = self.perms[nb]
        acc = torch.zeros_like(W)
        for a, b in zip(po, pi):
            acc = acc + W[a][:, b]
        return acc / nb

    def forward(self, W):
        if not TIER["nb"]:
            return W
        return W + (self.project(W, TIER["nb"]) - W).detach()  # STE


torch.manual_seed(1)
tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
model.load_state_dict(torch.load(CKPT, map_location="cpu",
                                 weights_only=True))
for blk in model.blocks:
    P.register_parametrization(blk.gate, "weight", TierP(FFN, D, dev))

rows = load_rows(gen4=True)
rows = [r for r in rows
        if r["cur"].replace(" ", "") != r["nxt"].replace(" ", "")]
enc = []
for r in rows:
    try:
        ids = tok.encode(f"Current: {r['cur']}\nHints: none\n"
                         f"Step: {r['nxt']}\n") + [tok.eos_id]
    except ValueError:
        continue
    if len(ids) <= 512:
        enc.append(ids)
enc.sort(key=len)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
order = list(range(0, len(enc) - BS + 1, BS))
random.Random(0).shuffle(order)
for bi, off in enumerate(order):
    batch = enc[off:off + BS]
    L = max(len(q) for q in batch)
    x = torch.tensor([q + [tok.pad_id] * (L - len(q)) for q in batch],
                     device=dev)
    y = x[:, 1:]
    loss = 0.0
    for nb in (0, 2, 8):
        TIER["nb"] = nb
        logits = model(x)[:, :-1]
        loss = loss + F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
            ignore_index=tok.pad_id)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if bi % 2000 == 0:
        print(f"{bi}/{len(order)} joint3 loss {float(loss):.4f}",
              flush=True)

model.eval()
for nb, name in ((0, "DENSE"), (2, "HALF"), (8, "EIGHTH")):
    TIER["nb"] = nb
    with torch.no_grad():
        solves, valid = G.gate_eval(model, tok, dev)
    print(f"MATRYOSHKA-R2 {name}: {solves} = "
          f"{sum(solves.values())}/120 @ {valid:.2f}%", flush=True)
TIER["nb"] = 0
for blk in model.blocks:
    P.remove_parametrizations(blk.gate, "weight")
torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
           OUT)
