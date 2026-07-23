"""METABOLIC V3 — the stacked LLMUE session (spec: four banked
upgrades, one run, separately toggleable via env):

  FP64_MASTERS=1   fp64 master weights for the trainable (late) layers
                   (arm B pattern: 5x flip recovery at 2.5e-6)
  SURPRISE=1       per-row LR scale = (1 - wave verified-fraction)
                   (own-difficulty read; mastered rows ~zero LR)
  CONTRAST=1       wave-contrast: small unlikelihood push on rejected
                   siblings from mixed waves (free preference pairs)
  (always)         late-layer control rod (first half frozen),
                   two-tier gates, snapshot/rollback with ABSOLUTE
                   anchor, live absorption + flip census.

HEADLINE PRE-REG (the ceiling-on-slow-learning theory, Artin):
paired arms at equal food — arm fp32 (FP64_MASTERS=0) vs arm fp64
(=1), same seeds. Prediction (Artin+law): fp64 arm shows MORE
frontier-band gain and MORE flips at equal food; if flat, online
precision joins birth precision as closed.

Usage: metabolic_v3.py <ckpt_latent> <label> <minutes>
Target: the crown-tier ternary lineage (merged_grown latent).
"""
import os, sys, time, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy as sp
import step_grpo_micro as G
from llmopt.train.mathnative import MathTokenizer, build_model
from bench_step_tokens import _gen_isolated
from bench_verify_fast import verify_wave

CKPT, LABEL, MINUTES = sys.argv[1], sys.argv[2], int(sys.argv[3])
FP64 = os.environ.get("FP64_MASTERS") == "1"
SURPRISE = os.environ.get("SURPRISE") == "1"
CONTRAST = os.environ.get("CONTRAST") == "1"
D, LAYERS, FFN, HEADS = 768, 8, 3840, 12
LR = 2.5e-6
FOOD_BAND = 94_000_000
FOOD_LEVELS = [4, 5, 6, 7, 8, 9]

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
dev = ("cuda" if torch.cuda.is_available() else
       "mps" if torch.backends.mps.is_available() else "cpu")
model = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
# control rod: freeze the first half + embeddings
for li, blk in enumerate(model.blocks):
    if li < LAYERS // 2:
        for p in blk.parameters():
            p.requires_grad_(False)
model.emb.weight.requires_grad_(False)
params = [p for p in model.parameters() if p.requires_grad]
start_sign = {id(p): torch.sign(ternary(p.detach().float().cpu()))
              for p in params if p.dim() == 2}

MASTER_DEV = "cpu" if dev == "mps" else dev  # MPS has no fp64
if FP64:
    masters = [p.detach().to(MASTER_DEV).double().clone()
               for p in params]
    for m in masters:
        m.requires_grad_(True)
    opt = torch.optim.AdamW(masters, lr=LR, weight_decay=0.0)
else:
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)

model.eval()
s0, v0 = G.gate_eval(model, tok, dev, n=8)
ANCHOR = sum(s0.values())  # absolute anchor (slow-leak tripwire)
print(f"[{LABEL}] pre: proxy {ANCHOR} @ {v0:.1f}% | fp64={FP64} "
      f"surprise={SURPRISE} contrast={CONTRAST}", flush=True)
snap = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

buf = []          # (ids, weight)
neg = []          # rejected sibling ids (contrast)
absorbed = tried_upd = 0
t0 = time.time()
cycle = 0
frontier_solved = 0
while time.time() - t0 < MINUTES * 60:
    cycle += 1
    for k in range(4):
        lv = FOOD_LEVELS[(cycle * 4 + k) % len(FOOD_LEVELS)]
        p = _gen_isolated(lv, FOOD_BAND + cycle * 17 + k)
        if p is None:
            continue
        cur = f"Integral({sp.sstr(p._expr)}, x)"
        prompt = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
        with torch.no_grad():
            texts, _, _ = G.sample_wave_lp(
                model, tok, prompt,
                [FOOD_BAND + cycle * 31 + b for b in range(8)], dev)
        distinct = [t for t in dict.fromkeys(texts) if t]
        wv = verify_wave(cur, distinct) if distinct else {}
        ver = [t for t in distinct if wv.get(t, (False, False))[0]]
        rej = [t for t in distinct if not wv.get(t, (False, False))[0]]
        vfrac = sum(wv.get(t, (False, False))[0] for t in texts) / max(len(texts), 1)
        frontier_solved += any(wv.get(t, (False, False))[1] for t in ver)
        w_row = (1.0 - vfrac) if SURPRISE else 1.0
        for t_ in ver:
            ids = tok.encode(f"Current: {cur}\nHints: none\n"
                             f"Step: {t_}\n") + [tok.eos_id]
            if len(ids) <= 512 and w_row > 0.01:
                buf.append((ids, w_row))
        if CONTRAST and ver and rej:
            ids = tok.encode(f"Current: {cur}\nHints: none\n"
                             f"Step: {rej[0]}\n") + [tok.eos_id]
            if len(ids) <= 512:
                neg.append((ids, 0.1 * w_row))
    if len(buf) >= 16:
        model.train()
        rows = buf[:24] + [(i, -w) for i, w in neg[:8]]
        buf, neg = buf[24:], neg[8:]
        L = max(len(q) for q, _ in rows)
        x = torch.tensor([q + [tok.pad_id] * (L - len(q))
                          for q, _ in rows], device=dev)
        wts = torch.tensor([w for _, w in rows], device=dev)
        logits = model(x)[:, :-1]
        y = x[:, 1:]
        per_tok = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
            ignore_index=tok.pad_id, reduction="none").view(y.shape)
        per_row = per_tok.sum(1) / (y != tok.pad_id).sum(1).clamp(min=1)
        loss = (per_row * wts).sum() / wts.abs().sum().clamp(min=1e-8)
        opt.zero_grad(); loss.backward()
        if FP64:
            for m, p in zip(masters, params):
                m.grad = p.grad.to(MASTER_DEV).double()
            opt.step()
            with torch.no_grad():
                for m, p in zip(masters, params):
                    p.copy_(m.float().to(p.device))
        else:
            with torch.no_grad():  # absorption census (fp32 arm)
                for p in params:
                    if p.grad is not None and p.dim() == 2:
                        delta = -LR * p.grad  # first-order proxy
                        absorbed += int(((p + delta) == p).sum())
                        tried_upd += delta.numel()
            opt.step()
        model.eval()
    if cycle % 40 == 0:
        s1, v1 = G.gate_eval(model, tok, dev, n=8)
        cur_p = sum(s1.values())
        print(f"[{LABEL}] cyc {cycle} proxy {cur_p} "
              f"frontier {frontier_solved} "
              f"{(time.time()-t0)/60:.0f}m", flush=True)
        if cur_p < ANCHOR - 3:  # absolute-anchor rollback
            model.load_state_dict(snap); model.to(dev)
            print(f"[{LABEL}] ROLLBACK (proxy {cur_p} < anchor-3)",
                  flush=True)
        else:
            snap = {k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()}

flips = tot = 0
for p in params:
    if p.dim() == 2:
        now = torch.sign(ternary(p.detach().float().cpu()))
        flips += int((now != start_sign[id(p)]).sum())
        tot += now.numel()
s1, v1 = G.gate_eval(model, tok, dev, n=8)
print(f"[{LABEL}] post: proxy {sum(s1.values())} @ {v1:.1f}% | "
      f"frontier solves {frontier_solved} | "
      f"FLIPS {flips}/{tot} ({100*flips/max(tot,1):.4f}%)"
      + (f" | absorbed {absorbed}/{tried_upd}" if not FP64 else ""),
      flush=True)
torch.save(model.state_dict(), f"checkpoints/metab_v3_{LABEL}.pt")
