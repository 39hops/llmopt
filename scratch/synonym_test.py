"""Synonym gauge test: TWO label tokens per family on the frozen
19M readout (vocab 40 -> 55: <name> + 7x2 synonyms). Train rows
pick either synonym 50/50. Gauge-law prediction: both fire
near-equal off the same concept. Reports family-accuracy +
per-synonym share."""
import sys, random, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
import torch.nn.functional as F
from llmopt.train.mathnative import MathTokenizer, build_model
from bench_step_tokens import _gen_isolated
import sympy as sp
from concurrent.futures import ProcessPoolExecutor

SYN = {"polynomial": ["poly-fam", "power-fam"],
       "exponential": ["exp-fam", "growth-fam"],
       "logarithm": ["log-fam", "ln-fam"],
       "trig": ["trig-fam", "wave-fam"],
       "inverse-trig": ["itrig-fam", "arc-fam"],
       "root": ["root-fam", "surd-fam"],
       "mixed": ["mixed-fam", "combo-fam"]}
NEW = ["<name>"] + [s for v in SYN.values() for s in v]
OLD_V = 40
tok = MathTokenizer()
for t in NEW:
    tok.id[t] = len(tok.vocab); tok.vocab.append(t)
tok._by_len = sorted((t for t in tok.vocab
                      if t not in ("<pad>", "<eos>")),
                     key=len, reverse=True)
NAME_ID = tok.id["<name>"]

def label_of(s):
    fams = set()
    if "atan(" in s or "asin(" in s: fams.add("inverse-trig")
    if "sin(" in s.replace("asin(", "") or "cos(" in s or \
       "tan(" in s.replace("atan(", ""): fams.add("trig")
    if "exp(" in s: fams.add("exponential")
    if "log(" in s: fams.add("logarithm")
    if "sqrt(" in s: fams.add("root")
    if not fams: return "polynomial"
    return "mixed" if len(fams) > 1 else fams.pop()

def _one(args):
    lvl, seed = args
    p = _gen_isolated(lvl, seed)
    return None if p is None else sp.sstr(p._expr)

def gen(n, band, exclude=None):
    rows, seen = [], set(exclude or ())
    rng = random.Random(f"syn-{band}")
    tasks = [(rng.choice(range(1, 9)), band * 10_000_000 + i)
             for i in range(n * 3)]
    with ProcessPoolExecutor(max_workers=8) as ex:
        for e in ex.map(_one, tasks, chunksize=4):
            if e is None or e in seen: continue
            seen.add(e); rows.append((e, label_of(e)))
            if len(rows) >= n: break
    return rows, seen

def encode(e):
    return tok.encode(f"Current: Integral({e}, x)\n") + [NAME_ID]

if __name__ == "__main__":
    t0 = time.time()
    train, seen = gen(2000, 1)
    evl, _ = gen(200, 2, exclude=seen)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_model(len(tok.vocab), d=384, layers=8, heads=6,
                        ffn=1536).to(dev)
    sd = torch.load("checkpoints/mathnative_19m_v21.pt",
                    map_location="cpu")
    cur = model.state_dict()
    for k, W in sd.items():
        if k in ("emb.weight", "head.weight"):
            cur[k][:OLD_V] = W
        else:
            cur[k] = W
    for k in ("emb.weight", "head.weight"):
        torch.nn.init.normal_(cur[k][OLD_V:], std=0.02)
    model.load_state_dict(cur)
    for p in model.parameters():
        p.requires_grad_(False)
    model.emb.weight.requires_grad_(True)
    model.head.weight.requires_grad_(True)
    opt = torch.optim.AdamW([model.emb.weight, model.head.weight],
                            lr=3e-3, weight_decay=0.0)
    rng = random.Random("syn-pick")
    enc = [(encode(e), tok.id[rng.choice(SYN[lab])])
           for e, lab in train]
    for ep in range(3):
        random.Random(f"syn-shuf-{ep}").shuffle(enc)
        model.train()
        for i in range(0, len(enc), 32):
            b = enc[i:i+32]
            L = max(len(q) for q, _ in b)
            x = torch.tensor([q + [tok.pad_id]*(L-len(q))
                              for q, _ in b], device=dev)
            y = torch.tensor([t for _, t in b], device=dev)
            pos = torch.tensor([len(q)-1 for q, _ in b], device=dev)
            logits = model(x)
            loss = F.cross_entropy(
                logits[torch.arange(len(b)), pos], y)
            opt.zero_grad(); loss.backward()
            model.emb.weight.grad[:OLD_V] = 0
            model.head.weight.grad[:OLD_V] = 0
            opt.step()
    model.eval()
    hit = 0
    share = {s: 0 for v in SYN.values() for s in v}
    with torch.no_grad():
        for e, lab in evl:
            q = encode(e)
            lg = model(torch.tensor([q], device=dev))[0, -1]
            pred = tok.vocab[int(lg[OLD_V+1:].argmax()) + OLD_V + 1]
            fam = next((f for f, v in SYN.items() if pred in v), None)
            share[pred] += 1
            hit += (fam == lab)
    print(f"SYNONYM: family-accuracy {hit}/{len(evl)} = "
          f"{100*hit/len(evl):.1f}%", flush=True)
    for f, v in SYN.items():
        a, b = share[v[0]], share[v[1]]
        print(f"  {f}: {v[0]}={a} {v[1]}={b}", flush=True)
    print(f"total {time.time()-t0:.0f}s", flush=True)
