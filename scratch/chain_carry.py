"""CHAIN-CARRY ABLATION (Artin's carry hypothesis, spec'd
2026-07-21): same content, format ablated, equal TOKEN budget,
both arms from scratch (d384/8L/3ep). Arm 'chains' = cur->nxt
pairs as-is. Arm 'oneshot' = reconstructed root->final-answer
rows (chains followed by nxt->cur linkage), upsampled to equal
tokens. Gate both. If chains >> oneshot, capability numbers carry
a format dividend. Usage: chain_carry.py <chains|oneshot>"""
import sys, json, glob, random, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
import torch.nn.functional as F
from llmopt.train.mathnative import MathTokenizer, build_model
import step_grpo_micro as G

ARM = sys.argv[1]
tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"

rows = []
for f in sorted(glob.glob("data/micromodel_chains_shard*.jsonl")):
    rows += [json.loads(l) for l in open(f)]
random.Random("cc-rows").shuffle(rows)
pair_rows = rows[:8000]
tok_budget = sum(len(tok.encode(
    f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"))
    for r in pair_rows)
print(f"[{ARM}] token budget {tok_budget}", flush=True)

if ARM == "chains":
    train_rows = [(r["cur"], r["nxt"]) for r in pair_rows]
else:
    nxt_of = {}
    for r in rows:
        nxt_of[r["cur"].replace(" ", "")] = r
    starts = [r for r in rows
              if r["cur"].startswith("Integral(")]
    finals = []
    for r in starts:
        seen = set()
        cur = r
        while True:
            key = cur["nxt"].replace(" ", "")
            if key in seen:
                break
            seen.add(key)
            nx = nxt_of.get(key)
            if nx is None:
                break
            cur = nx
        if "Integral(" not in cur["nxt"]:
            finals.append((r["cur"], cur["nxt"]))
    print(f"[oneshot] {len(finals)} root->answer chains "
          f"reconstructed", flush=True)
    random.Random("cc-one").shuffle(finals)
    train_rows, used = [], 0
    i = 0
    while used < tok_budget:
        c, n = finals[i % len(finals)]
        used += len(tok.encode(f"Current: {c}\nHints: none\n"
                               f"Step: {n}\n"))
        train_rows.append((c, n))
        i += 1
    print(f"[oneshot] {len(train_rows)} rows at equal tokens",
          flush=True)

model = build_model(len(tok.vocab), d=384, layers=8, heads=6,
                    ffn=1536).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                        weight_decay=0.01)
enc = []
for c, n in train_rows:
    ids = tok.encode(f"Current: {c}\nHints: none\nStep: {n}\n") \
        + [tok.eos_id]
    if len(ids) <= 512:
        enc.append(ids)
enc.sort(key=len)
batches = [enc[i:i+32] for i in range(0, len(enc), 32)]
t0 = time.time()
for ep in range(3):
    random.Random(f"cc-{ARM}-{ep}").shuffle(batches)
    model.train()
    tot = 0.0
    for b in batches:
        L = max(len(q) for q in b)
        x = torch.tensor([q + [tok.pad_id]*(L-len(q)) for q in b],
                         device=dev)
        logits = model(x)[:, :-1]
        y = x[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
            ignore_index=tok.pad_id)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(b)
    print(f"[{ARM}] ep{ep} loss {tot/len(enc):.4f} "
          f"({(time.time()-t0)/60:.0f} min)", flush=True)
model.eval()
solves, valid = G.gate_eval(model, tok, dev)
print(f"CHAIN-CARRY [{ARM}] gate: {solves} = "
      f"{sum(solves.values())}/120 @ {valid:.2f}%", flush=True)
torch.save(model.state_dict(), f"checkpoints/cc_{ARM}.pt")
