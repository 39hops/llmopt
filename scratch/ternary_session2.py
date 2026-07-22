"""Ternary compounding session #2 (Mac, MPS lineage, paired
gates): the doctrine-composed organism — STE ternary latents,
LATE layers only (8-11), LR 1e-4 cap, ABSOLUTE-anchor tripwire,
fp32-vs-fp64 update-absorption instrument riding along.
Pre/post MPS gates make it a clean paired delta."""
import sys, json, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy as sp
import step_grpo_micro as G
from llmopt.train.mathnative import MathTokenizer, build_model
from bench_step_tokens import _gen_isolated
from bench_verify_fast import verify_wave

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
tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(len(tok.vocab), d=512, layers=12, heads=8,
                    ffn=2048).to(dev)
model.load_state_dict(torch.load(
    "checkpoints/ternary_nnue_latent.pt", map_location="cpu"))
for li, blk in enumerate(model.blocks):
    if li < 8:
        for p in blk.parameters():
            p.requires_grad_(False)
model.emb.weight.requires_grad_(False)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
model.eval()
s0, v0 = G.gate_eval(model, tok, dev, n=8)
base = sum(s0.values())
print(f"pre: proxy {base} @ {v0:.1f}%", flush=True)
BEST = base
sidecar = open("data/ternary_s2_sidecar.jsonl", "a")
snap = "checkpoints/ternary_s2_snap.pt"
torch.save(model.state_dict(), snap)
absorbed = total_upd = 0
buf = []
SEED0 = 97_000_000
t0 = time.time()
cycle = 0
while time.time() - t0 < 90 * 60:
    cycle += 1
    rows = []
    for k in range(4):
        p = _gen_isolated(9, SEED0 + cycle * 17 + k)
        if p is None: continue
        cur = f"Integral({sp.sstr(p._expr)}, x)"
        prompt = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
        model.eval()
        with torch.no_grad():
            texts, _, _ = G.sample_wave_lp(
                model, tok, prompt,
                [SEED0 + cycle * 31 + b for b in range(8)], dev)
        distinct = [t for t in dict.fromkeys(texts) if t]
        wv = verify_wave(cur, distinct) if distinct else {}
        for t_ in distinct:
            ok, _s = wv.get(t_, (False, False))
            if ok:
                rows.append({"cur": cur, "nxt": t_, "level": 9,
                             "source": "ternary-s2"})
    for r in rows:
        sidecar.write(json.dumps(r) + "\n")
        ids = tok.encode(f"Current: {r['cur']}\nHints: none\n"
                         f"Step: {r['nxt']}\n") + [tok.eos_id]
        if len(ids) <= 512:
            buf.append(ids)
    sidecar.flush()
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
        pre = [p.detach().clone() for p in params]
        opt.zero_grad(); loss.backward(); opt.step()
        # absorption instrument: fp32 steps that changed nothing
        for p_new, p_old in zip(params, pre):
            g = p_new.grad if hasattr(p_new, "grad") else None
            same = (p_new == p_old)
            absorbed += int(same.sum())
            total_upd += p_new.numel()
        buf = []
    if cycle % 10 == 0:
        model.eval()
        s, v = G.gate_eval(model, tok, dev, n=8)
        sc = sum(s.values())
        print(f"cycle {cycle}: proxy {sc} @ {v:.1f}% | absorbed "
              f"{absorbed}/{total_upd} "
              f"({100*absorbed/max(total_upd,1):.2f}%) | "
              f"{(time.time()-t0)/60:.0f} min", flush=True)
        if sc < BEST - 4:   # ABSOLUTE anchor
            model.load_state_dict(torch.load(snap,
                                             map_location="cpu"))
            model.to(dev)
            print(f"ROLLBACK (absolute anchor) at cycle {cycle}",
                  flush=True)
        else:
            BEST = max(BEST, sc)
            torch.save(model.state_dict(), snap)
torch.save(model.state_dict(),
           "checkpoints/ternary_s2_latent.pt")
model.eval()
s1, v1 = G.gate_eval(model, tok, dev, n=8)
print(f"post: proxy {sum(s1.values())} @ {v1:.1f}% (pre {base}) | "
      f"absorption {100*absorbed/max(total_upd,1):.2f}%", flush=True)
