"""Valuation-routed metabolism v0: committee-gated per-neuron
plasticity, one-variable A/B (see RIFF-LEDGER 2026-07-21).
Mask: per-layer FFN committee probe; per-family top-5% neurons =
heavy -> LR x0.2, rest x1.5, field normalized to mean 1.0 (equal
average LR vs uniform arm). Arms: uniform vs routed, same 8k rows,
1 epoch, honest 120-gate each. Baseline 19m_v21 = 64."""
import sys, json, glob, random, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
import torch.nn.functional as F
from llmopt.train.mathnative import MathTokenizer, build_model
import step_grpo_micro as G

D, LAYERS, FFN, HEADS = 384, 8, 1536, 6
CKPT = "checkpoints/mathnative_19m_v21.pt"
FAM = {
    "power": lambda r: f"x**{r.randint(2,9)}",
    "trig":  lambda r: f"{r.choice(['sin','cos'])}({r.randint(2,7)}*x)",
    "exp":   lambda r: f"exp({r.randint(2,7)}*x)",
    "recip": lambda r: f"1/(x + {r.randint(1,9)})",
    "log":   lambda r: f"log({r.randint(2,7)}*x)",
}
tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def committee_masks():
    m = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN)
    m.load_state_dict(torch.load(CKPT, map_location="cpu"))
    m.eval()
    store = {}
    hooks = [blk.gate.register_forward_hook(
        (lambda li: lambda mod, i, o: store.__setitem__(li, o))(li))
        for li, blk in enumerate(m.blocks)]
    sums = {li: {} for li in range(LAYERS)}
    for fam, gen in FAM.items():
        r = random.Random(f"vrm-{fam}")
        for _ in range(30):
            p = (f"Current: Integral({gen(r)}, x)\n"
                 f"Hints: none\nStep: ")
            with torch.no_grad():
                m(torch.tensor([tok.encode(p)]))
            for li in range(LAYERS):
                a = store[li][0].mean(0).abs()
                sums[li][fam] = sums[li].get(fam, 0) + a
    for h in hooks:
        h.remove()
    masks, heavy_tot = [], 0
    for li in range(LAYERS):
        mu = torch.stack([sums[li][f] for f in FAM])
        heavy = torch.zeros(FFN, dtype=torch.bool)
        k = FFN // 20
        for fi in range(len(FAM)):
            others = mu[[j for j in range(len(FAM)) if j != fi]].mean(0)
            ratio = mu[fi] / (others + 1e-6)
            heavy[torch.topk(ratio, k).indices] = True
        field = torch.where(heavy, torch.tensor(0.2),
                            torch.tensor(1.5))
        field = field / field.mean()
        masks.append(field)
        heavy_tot += int(heavy.sum())
    print(f"[mask] {heavy_tot} heavy neurons / {LAYERS*FFN} "
          f"({100*heavy_tot/(LAYERS*FFN):.1f}%)", flush=True)
    return masks


def load_rows(n=8000):
    rows = []
    for f in sorted(glob.glob("data/micromodel_chains_shard*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    random.Random("vrm-rows").shuffle(rows)
    return rows[:n]


def run_arm(name, masks, rows):
    model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    if masks is not None:
        for li, blk in enumerate(model.blocks):
            fld = masks[li].to(dev)
            blk.gate.weight.register_hook(
                (lambda f: lambda g: g * f[:, None])(fld))
            blk.up.weight.register_hook(
                (lambda f: lambda g: g * f[:, None])(fld))
            blk.down.weight.register_hook(
                (lambda f: lambda g: g * f[None, :])(fld))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5,
                            weight_decay=0.0)
    enc = []
    for r in rows:
        ids = tok.encode(f"Current: {r['cur']}\nHints: none\n"
                         f"Step: {r['nxt']}\n") + [tok.eos_id]
        if len(ids) <= 512:
            enc.append(ids)
    enc.sort(key=len)
    batches = [enc[i:i+32] for i in range(0, len(enc), 32)]
    random.Random("vrm-order").shuffle(batches)
    model.train()
    t0, tot = time.time(), 0.0
    for b in batches:
        L = max(len(q) for q in b)
        x = torch.tensor([q + [tok.pad_id]*(L-len(q)) for q in b],
                         device=dev)
        logits = model(x)[:, :-1]
        y = x[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1), ignore_index=tok.pad_id)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(b)
    print(f"[{name}] 1ep loss {tot/len(enc):.4f} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)
    model.eval()
    solves, valid = G.gate_eval(model, tok, dev)
    print(f"[{name}] gate: {solves} = {sum(solves.values())}/120 "
          f"@ {valid:.2f}%", flush=True)
    torch.save(model.state_dict(), f"checkpoints/vrm_{name}.pt")
    return sum(solves.values())


if __name__ == "__main__":
    t0 = time.time()
    masks = committee_masks()
    rows = load_rows()
    print(f"{len(rows)} rows loaded", flush=True)
    u = run_arm("uniform", None, rows)
    r_ = run_arm("routed", masks, rows)
    print(f"VRM_VERDICT: routed {r_} vs uniform {u} "
          f"(baseline 19m_v21 = 64) | {(time.time()-t0)/60:.0f} min",
          flush=True)
