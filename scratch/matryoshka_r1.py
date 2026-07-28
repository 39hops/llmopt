"""Matryoshka rung 1 (pre-reg 2026-07-28 night): joint loss
CE(W) + CE(STE P_C8(W)) — one crystal whose OWN circulant
projection must also work. 1 warm epoch from wfloor d256 on
MPS. Implementation: parametrize gate weights with a toggleable
STE projection (flag off -> raw W; flag on -> W + (P(W)-W)
.detach(), i.e. forward uses P(W), gradient flows to W).
Gates BOTH tiers at the end; saves the single weight tensor.
"""
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

D, LAYERS, FFN, HEADS, BS, NB = 256, 8, 1024, 4, 32, 8
OUT = "checkpoints/matryoshka_d256.pt"
TIER = {"on": False}  # global toggle read by the parametrization


def shift_perm(n, sh, dev):
    return torch.tensor([NB * (r // NB) + (r % NB - sh) % NB
                         for r in range(n)], device=dev)


class TierP(torch.nn.Module):
    def __init__(self, n_out, n_in, dev):
        super().__init__()
        self.po = [shift_perm(n_out, s, dev) for s in range(NB)]
        self.pi = [shift_perm(n_in, s, dev) for s in range(NB)]

    def project(self, W):
        acc = torch.zeros_like(W)
        for po, pi in zip(self.po, self.pi):
            acc = acc + W[po][:, pi]
        return acc / NB

    def forward(self, W):
        if not TIER["on"]:
            return W
        return W + (self.project(W) - W).detach()  # STE


torch.manual_seed(1)
tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
model.load_state_dict(torch.load(
    "checkpoints/mathnative_wfloor_d256.pt", map_location="cpu",
    weights_only=True))
for blk in model.blocks:
    P.register_parametrization(blk.gate, "weight",
                               TierP(FFN, D, dev))

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
opt = torch.optim.AdamW(model.parameters(), lr=1e-4,
                        weight_decay=0.01)
order = list(range(0, len(enc) - BS + 1, BS))
random.Random(0).shuffle(order)
for bi, off in enumerate(order):
    batch = enc[off:off + BS]
    L = max(len(q) for q in batch)
    x = torch.tensor([q + [tok.pad_id] * (L - len(q))
                      for q in batch], device=dev)
    y = x[:, 1:]
    loss = 0.0
    for tier_on in (False, True):
        TIER["on"] = tier_on
        logits = model(x)[:, :-1]
        loss = loss + F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
            ignore_index=tok.pad_id)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if bi % 500 == 0:
        print(f"{bi}/{len(order)} joint loss {float(loss):.4f}",
              flush=True)

model.eval()
for tier_on, name in ((False, "DENSE-TIER"), (True, "CHEAP-TIER")):
    TIER["on"] = tier_on
    with torch.no_grad():
        solves, valid = G.gate_eval(model, tok, dev)
    print(f"MATRYOSHKA-R1 {name}: {solves} = "
          f"{sum(solves.values())}/120 @ {valid:.2f}%", flush=True)
TIER["on"] = False
for blk in model.blocks:
    P.remove_parametrizations(blk.gate, "weight")
torch.save({k: v.detach().cpu()
            for k, v in model.state_dict().items()}, OUT)
