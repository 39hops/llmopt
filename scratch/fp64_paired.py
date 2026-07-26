"""THE ROUNDING-LOSS DECIDER (overnight GO): fp32 vs fp64-master
paired burst at LR 2.5e-6 (GRPO's real regime). Same food stream,
late-layer STE, 40 min/arm. PRIMARY metric: committed ternary
flips vs own start (sub-ULP nudges absorbed by fp32 should COMMIT
under fp64 masters -> more flips at equal food). Secondary: proxy.
Usage: fp64_paired.py <fp32|fp64>"""
import sys, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy as sp
import step_grpo_micro as G
from llmopt.train.mathnative import MathTokenizer, build_model
from bench_step_tokens import _gen_isolated
from bench_verify_fast import verify_wave

ARM = sys.argv[1]
def ternary(w):
    s = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    return torch.where(w.abs() < 0.5*s, torch.zeros_like(w),
                       torch.sign(w)*s)

class TLin(nn.Linear):
    def forward(self, x):
        if self.out_features == 40:
            return F.linear(x, self.weight, self.bias)
        w = self.weight
        wq = w + (ternary(w) - w).detach()
        return F.linear(x, wq, self.bias)

nn.Linear = TLin
torch.backends.cuda.matmul.allow_tf32 = True
tok = MathTokenizer()
dev = "cuda"
model = build_model(len(tok.vocab), d=512, layers=12, heads=8,
                    ffn=2048).to(dev)
model.load_state_dict(torch.load(
    "checkpoints/mathnative_gen6_ternary_latent.pt",
    map_location="cpu"))
for li, blk in enumerate(model.blocks):
    if li < 8:
        for p in blk.parameters():
            p.requires_grad_(False)
model.emb.weight.requires_grad_(False)
params = [p for p in model.parameters() if p.requires_grad]
start = {id(p): torch.sign(ternary(p.detach().float().cpu()))
         for p in params if p.dim() == 2}

if ARM == "fp64":
    masters = [p.detach().double().clone() for p in params]
    for m in masters:
        m.requires_grad_(True)
    opt = torch.optim.AdamW(masters, lr=2.5e-6, weight_decay=0.0)
else:
    opt = torch.optim.AdamW(params, lr=2.5e-6, weight_decay=0.0)

model.eval()
s0, v0 = G.gate_eval(model, tok, dev, n=8)
print(f"[{ARM}] pre: proxy {sum(s0.values())} @ {v0:.1f}%",
      flush=True)
buf = []
SEED0 = 99_000_000
t0 = time.time()
cycle = 0
while time.time() - t0 < 40 * 60:
    cycle += 1
    for k in range(4):
        p = _gen_isolated(9, SEED0 + cycle * 17 + k)
        if p is None: continue
        cur = f"Integral({sp.sstr(p._expr)}, x)"
        prompt = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
        with torch.no_grad():
            texts, _, _ = G.sample_wave_lp(
                model, tok, prompt,
                [SEED0 + cycle * 31 + b for b in range(8)], dev)
        distinct = [t for t in dict.fromkeys(texts) if t]
        wv = verify_wave(cur, distinct) if distinct else {}
        for t_ in distinct:
            ok, _s = wv.get(t_, (False, False))
            if ok:
                ids = tok.encode(f"Current: {cur}\nHints: none\n"
                                 f"Step: {t_}\n") + [tok.eos_id]
                if len(ids) <= 512:
                    buf.append(ids)
    if len(buf) >= 16:
        model.train()
        L = max(len(q) for q in buf)
        x = torch.tensor([q + [tok.pad_id]*(L-len(q)) for q in buf],
                         device=dev)
        logits = model(x)[:, :-1]
        y = x[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
            ignore_index=tok.pad_id)
        opt.zero_grad(); loss.backward()
        if ARM == "fp64":
            for m, p in zip(masters, params):
                m.grad = p.grad.double()
            opt.step()
            with torch.no_grad():
                for m, p in zip(masters, params):
                    p.copy_(m.float())
        else:
            opt.step()
        buf = []
        model.eval()
    if cycle % 25 == 0:
        print(f"[{ARM}] cycle {cycle} "
              f"{(time.time()-t0)/60:.0f} min", flush=True)
flips = 0
tot = 0
per = {}
names = {id(p): n for n, p in model.named_parameters()}
for p in params:
    if p.dim() == 2:
        now = torch.sign(ternary(p.detach().float().cpu()))
        f = int((now != start[id(p)]).sum())
        flips += f
        tot += now.numel()
        per[names[id(p)]] = f
print(f"[{ARM}] per-matrix flips: " + " ".join(
    f"{k}={v}" for k, v in sorted(per.items(),
                                  key=lambda kv: -kv[1])[:8]),
    flush=True)
s1, v1 = G.gate_eval(model, tok, dev, n=8)
print(f"[{ARM}] post: proxy {sum(s1.values())} @ {v1:.1f}% | "
      f"FLIPS {flips}/{tot} ({100*flips/tot:.4f}%)", flush=True)
