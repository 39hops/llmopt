"""Call-span paired arms (pre-reg 2026-07-29 night: Leg B first
read). Pilot 500 (axiom, sha de6c9f15): plain v span hints,
same rows, d64, 20 ep; held-out greedy next-step exact match.
Atoms pinned in sidecar order. MPS.
"""
from llmopt.common.device import pick_device
import json
import os
import random
import sys

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

PILOT = os.path.expanduser(os.environ.get(
    "PILOT", "~/code/axiom/data/llmopt/nt_callspan_pilot500.jsonl"))
ATOMS = ["gcd", "Mod", "**", "call:", "->",  # sidecar order (fence)
         "Hints: ", ";"]  # format atoms appended AFTER the diet
# atoms (amendment booked: base tokenizer covers "Hints: none"
# only as one fixed template atom, so span hints need "Hints: ")
D, LAYERS, FFN, HEADS, BS, EPOCHS, LR = 64, 8, 256, 4, 8, 20, 1.5e-3

rows = [json.loads(ln) for ln in open(PILOT)]
assert len(rows) == 500
random.Random(7).shuffle(rows)
train, evalr = rows[:400], rows[400:]


def text(r, span):
    hints = "; ".join(r["calls"]) if span else "none"
    return (f"Current: {r['cur']}\nHints: {hints}\nStep: ",
            f"{r['nxt']}\n")


def run_arm(span):
    torch.manual_seed(1)
    tok = MathTokenizer(extra=ATOMS)
    dev = pick_device()
    model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN).to(dev)
    enc = []
    for r in train:
        p, s = text(r, span)
        try:
            ids = tok.encode(p + s) + [tok.eos_id]
        except ValueError:
            continue
        if len(ids) <= 512:
            enc.append(ids)
    enc.sort(key=len)
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=0.01)
    order = list(range(0, len(enc) - BS + 1, BS))
    for ep in range(EPOCHS):
        random.Random(ep).shuffle(order)
        for off in order:
            batch = enc[off:off + BS]
            L = max(len(q) for q in batch)
            x = torch.tensor([q + [tok.pad_id] * (L - len(q))
                              for q in batch], device=dev)
            logits = model(x)[:, :-1]
            y = x[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                ignore_index=tok.pad_id)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    nl = tok.encode("\n")[-1]
    hit = tried = 0
    with torch.no_grad():
        for r in evalr:
            p, s = text(r, span)
            try:
                ids = tok.encode(p)
                want = tok.encode(s.rstrip("\n"))
            except ValueError:
                continue
            tried += 1
            cur = list(ids)
            out = []
            for _ in range(96):
                nxt = int(model(torch.tensor([cur], device=dev)
                                )[0, -1].argmax())
                if nxt in (tok.eos_id, nl):
                    break
                out.append(nxt)
                cur.append(nxt)
            hit += out == want
    print(f"CALLSPAN-ARM {'SPAN' if span else 'PLAIN'}: "
          f"{hit}/{tried} held-out exact", flush=True)
    return hit, tried


run_arm(False)
run_arm(True)
