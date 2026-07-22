"""HOT METABOLISM (2026-07-21, Artin GO): map the safe-plasticity
frontier. Pilot harness + LR ladder: start 3e-5, x1.8 every 20
stable cycles; immune system: proxy gate n=8 every 5 cycles,
2 consecutive drops >5 -> ROLLBACK + halve LR (frontier found).
Optional --late: freeze layers 0-7 (confluence shortcut: delta
mass is 8-11-heavy; backward stops at layer 8).
Band 95M (fresh). ~150 cycles."""
import sys, json, time, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
import step_grpo_micro as G
from llmopt.train.mathnative import MathTokenizer, build_model
from bench_step_tokens import _gen_isolated
from bench_verify_fast import verify_wave
import sympy as sp

LATE = "--late" in sys.argv
TAG = "late" if LATE else "hot"
tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(len(tok.vocab), d=512, layers=12, heads=8,
                    ffn=2304).to(dev)
model.load_state_dict(torch.load(
    "checkpoints/mathnative_gen6_grown.pt", map_location="cpu"))
if LATE:
    for li, blk in enumerate(model.blocks):
        if li < 8:
            for p in blk.parameters():
                p.requires_grad_(False)
    model.emb.weight.requires_grad_(False)
LR = 3e-5
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
sidecar = open(f"data/metabolic_{TAG}_sidecar.jsonl", "a")
snap = f"checkpoints/metabolic_{TAG}_snap.pt"
torch.save(model.state_dict(), snap)
best = None; drops = 0; stable = 0; buf = []
SEED0 = 95_000_000
t0 = time.time()
for cycle in range(1, 151):
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
                             "source": f"metabolic-{TAG}"})
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
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
            ignore_index=tok.pad_id)
        opt.zero_grad(); loss.backward(); opt.step()
        buf = []
    if cycle % 5 == 0:
        model.eval()
        solves, valid = G.gate_eval(model, tok, dev, n=8)
        s = sum(solves.values())
        print(f"cycle {cycle}: LR {LR:.1e} | proxy {s} @ "
              f"{valid:.1f}% | {(time.time()-t0)/60:.0f} min",
              flush=True)
        if best is None: best = s
        if s < best - 5:
            drops += 1
            if drops >= 2:
                model.load_state_dict(torch.load(snap,
                                                 map_location="cpu"))
                model.to(dev)
                LR /= 2
                for g in opt.param_groups: g["lr"] = LR
                print(f"ROLLBACK at cycle {cycle}; LR -> {LR:.1e} "
                      f"(FRONTIER)", flush=True)
                drops = 0; stable = 0
        else:
            drops = 0; stable += 5
            best = max(best, s)
            if stable >= 20:
                LR *= 1.8
                for g in opt.param_groups: g["lr"] = LR
                print(f"LADDER UP: LR -> {LR:.1e}", flush=True)
                stable = 0
        if cycle % 20 == 0:
            torch.save(model.state_dict(), snap)
torch.save(model.state_dict(), f"checkpoints/metabolic_{TAG}.pt")
print(f"DONE {TAG}: final LR {LR:.1e}", flush=True)
