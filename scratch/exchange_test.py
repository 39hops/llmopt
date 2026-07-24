"""THE EXCHANGE TEST (pre-registered 2026-07-23): train the v4
organism on axiom's engine-farmed chains at OUR stuck states, re-probe
the SAME fixed seeds (55_000_000, cuda — device law), must beat 2/12.
v4 measured self-practice at +1/12 (no gradient at true walls); the
exchange supplies exactly the missing gradient. 10/12 walls have
chains; ceiling = 12/12, bar = >=3/12, headline read = how many of
the 10 taught walls flip.

Run ON THE 3080 (WSL) when the box has a window:
  python scratch/exchange_test.py checkpoints/metab_v4.pt \
      data/stuck_states_p1.jsonl data/stuck_chains_p1.jsonl
"""
import os, sys, json, random
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
import torch.nn as nn
import torch.nn.functional as F
import step_grpo_micro as G
from llmopt.train.mathnative import MathTokenizer, build_model
from bench_verify_fast import verify_wave

CKPT, WORKLIST, CHAINS = sys.argv[1], sys.argv[2], sys.argv[3]
D, LAYERS, FFN, HEADS = 768, 8, 3840, 12
LR = 1e-5
EPOCHS = 30

def ternary(w):
    s = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    return torch.where(w.abs() < 0.5 * s, torch.zeros_like(w),
                       torch.sign(w) * s)

class TLin(nn.Linear):
    def forward(self, x):
        if self.out_features == 40:
            return F.linear(x, self.weight, self.bias)
        w = self.weight
        return F.linear(x, w + (ternary(w) - w).detach(), self.bias)

nn.Linear = TLin
tok = MathTokenizer()
dev = "cuda"
model = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
for li, blk in enumerate(model.blocks):
    if li < LAYERS // 2:
        for p in blk.parameters():
            p.requires_grad_(False)
model.emb.weight.requires_grad_(False)
params = [p for p in model.parameters() if p.requires_grad]
masters = [p.detach().double().clone() for p in params]
for m in masters:
    m.requires_grad_(True)
opt = torch.optim.AdamW(masters, lr=LR, weight_decay=0.0)

stuck = [json.loads(l) for l in open(WORKLIST)]
chains = [json.loads(l.lstrip("﻿"))
          for l in open(CHAINS, encoding="utf-8-sig")]
print(f"[ex] worklist {len(stuck)} | chain rows {len(chains)} | "
      f"LR {LR} x {EPOCHS} ep", flush=True)

def try_state(cur0, seed0, plies=8):
    cur = cur0
    visited = {cur.replace(" ", "")}
    for ply in range(plies):
        prompt = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
        with torch.no_grad():
            texts, _, _ = G.sample_wave_lp(
                model, tok, prompt,
                [seed0 + ply * 7 + b for b in range(8)], dev)
        distinct = [t for t in dict.fromkeys(texts)
                    if t and t.replace(" ", "") not in visited]
        wv = verify_wave(cur, distinct) if distinct else {}
        nxt = None
        for t in texts:
            ok, so = wv.get(t, (False, False))
            if ok and t.replace(" ", "") not in visited:
                if nxt is None:
                    nxt = "SOLVED" if so else t
        if nxt == "SOLVED":
            return True
        if nxt is None:
            return False
        cur = nxt
        visited.add(cur.replace(" ", ""))
    return False

def probe(tag):
    model.eval()
    res = {}
    for i, s in enumerate(stuck):
        res[s["id"]] = try_state(s["root"], 55_000_000 + i * 101)
    print(f"[ex] {tag} resolution: {sum(res.values())}/{len(stuck)} "
          f"{[k for k, v in res.items() if v]}", flush=True)
    return res

pre = probe("PRE")            # must reproduce v4's POST read (2/12)
s0, v0 = G.gate_eval(model, tok, dev, n=8)
print(f"[ex] anchor proxy {sum(s0.values())} @ {v0:.1f}%", flush=True)

rows = []
for r in chains:
    ids = tok.encode(f"Current: {r['cur']}\nHints: none\n"
                     f"Step: {r['nxt']}\n") + [tok.eos_id]
    if len(ids) <= 512:
        rows.append(ids)
rng = random.Random("exchange-p1")
model.train()
for ep in range(EPOCHS):
    rng.shuffle(rows)
    for b0 in range(0, len(rows), 8):
        batch = rows[b0:b0 + 8]
        L = max(len(q) for q in batch)
        x = torch.tensor([q + [tok.pad_id] * (L - len(q))
                          for q in batch], device=dev)
        logits = model(x)[:, :-1]
        y = x[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
            ignore_index=tok.pad_id)
        opt.zero_grad(); loss.backward()
        for m, p in zip(masters, params):
            m.grad = p.grad.double()
        opt.step()
        with torch.no_grad():
            for m, p in zip(masters, params):
                p.copy_(m.float())
    if ep % 10 == 9:
        print(f"[ex] ep {ep + 1} loss {loss.item():.4f}", flush=True)

post = probe("POST")          # same seeds = paired
s1, v1 = G.gate_eval(model, tok, dev, n=8)
taught = {r["cur"].replace(" ", "") for r in chains}
print(f"[ex] FINAL: {sum(pre.values())}/12 -> {sum(post.values())}/12 "
      f"(bar: beat 2) | proxy {sum(s0.values())} -> {sum(s1.values())} "
      f"| flips {[k for k in post if post[k] and not pre[k]]}",
      flush=True)
torch.save(model.state_dict(), "checkpoints/exchange_p1.pt")
