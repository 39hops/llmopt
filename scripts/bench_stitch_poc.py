"""Representation stitching, tier 1 (Artin's change-of-basis riff).

Teacher: SmolLM2-1.7B — different lineage, different tokenizer
(real cross-basis test). Same 4573 oracle-labeled states, same
orbital enrichment, same hash split, same bar family as the probe
arc (native layer-15 baseline: 90.5% exact / 0.979 micro-F1).

Rung 1 — foreign geometry: probe the teacher's layers directly
  (sweep every 3rd layer). Does an alien model's representation of
  OUR expressions carry OUR task signal?
Rung 2 — the bridge: fit a linear map teacher-best-layer ->
  student-layer-15 space on the train split (the overlap matrix,
  computed over shared text), then feed BRIDGED teacher vectors to
  a probe trained in student space. Signal surviving the change of
  basis = the stitching thesis lives; tier 2/3 unlock.

    .venv/bin/python scripts/bench_stitch_poc.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEACHER = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
STUDENT = "Qwen/Qwen2.5-0.5B-Instruct"


def pooled_layers(model_name, texts, layers, dev, bs=16, ml=384):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16).to(dev).eval()
    pools = {l: [] for l in layers}
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                    truncation=True, max_length=ml).to(dev)
            out = enc.model(input_ids=b["input_ids"],
                            attention_mask=b["attention_mask"],
                            output_hidden_states=True)
            m = b["attention_mask"].unsqueeze(-1).float()
            for l in layers:
                hs = out.hidden_states[l]
                pools[l].append(((hs * m).sum(1) / m.sum(1))
                                .float().cpu())
    del enc
    if dev == "mps":
        torch.mps.empty_cache()
    return {l: torch.cat(v) for l, v in pools.items()}


def probe(X, Y, tr, te, seed=0):
    import torch
    mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-6)
    Xn = (X - mu) / sd
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 96), torch.nn.ReLU(),
        torch.nn.Linear(96, 96), torch.nn.ReLU(),
        torch.nn.Linear(96, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(300):
        opt.zero_grad()
        lossf(net(Xn[tr]), Y[tr]).backward()
        opt.step()
    with torch.no_grad():
        pred = (net(Xn[te]) > 0).float()
    exact = (pred == Y[te]).all(1).float().mean().item()
    tp = (pred * Y[te]).sum().item()
    p = tp / max(pred.sum().item(), 1)
    r = tp / max(Y[te].sum().item(), 1)
    return 100 * exact, 2 * p * r / max(p + r, 1e-9)


def main() -> None:
    import torch
    rows = [json.loads(l) for l in
            Path("data/pred_syndrome_labels.jsonl").read_text().splitlines()]
    orb = {json.loads(l)["s"]: json.loads(l)["orb"] for l in
           Path("data/pred_syndrome_orbitals.jsonl").read_text().splitlines()}
    texts = [f"{r['s']} || orbitals: {orb.get(r['s'], '?')}" for r in rows]
    vocab = sorted({rn for r in rows for rn in r["fires"]})
    vi = {r: i for i, r in enumerate(vocab)}
    Y = torch.zeros(len(rows), len(vocab))
    for i, r in enumerate(rows):
        for rn in r["fires"]:
            Y[i, vi[rn]] = 1.0
    h = [int(hashlib.sha1(r["s"].encode()).hexdigest(), 16) % 5
         for r in rows]
    tr = torch.tensor([v != 0 for v in h])
    te = torch.tensor([v == 0 for v in h])
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    # rung 1: sweep the teacher's layers
    print(f"# teacher {TEACHER}", flush=True)
    t_layers = list(range(3, 25, 3))
    tp = pooled_layers(TEACHER, texts, t_layers, dev)
    best_l, best = None, (0.0, 0.0)
    for l in t_layers:
        ex, f1 = probe(tp[l], Y, tr, te)
        print(f"teacher layer {l:2d}: exact {ex:.1f}% f1 {f1:.3f}",
              flush=True)
        if ex > best[0]:
            best_l, best = l, (ex, f1)
    print(f"RUNG 1: teacher best layer {best_l} -> {best[0]:.1f}% / "
          f"{best[1]:.3f}  (native Qwen layer-15: 90.5 / 0.979)",
          flush=True)

    # rung 2: bridge teacher-best -> student layer-15 space
    sp = pooled_layers(STUDENT, texts, [15], dev)[15]
    Xt = tp[best_l]
    # least-squares linear bridge fit on TRAIN split only
    A = torch.linalg.lstsq(Xt[tr], sp[tr]).solution
    bridged = Xt @ A
    fit = 1 - ((bridged[te] - sp[te]).norm() /
               max(sp[te].norm(), 1e-9))
    ex, f1 = probe(bridged, Y, tr, te, seed=1)
    print(f"RUNG 2: bridged exact {ex:.1f}% f1 {f1:.3f} "
          f"(bridge R~{fit:.2f}); student-space probe on bridged "
          f"vectors", flush=True)
    # control: does a probe TRAINED on native student vectors accept
    # bridged ones? (the strong form of basis translation)
    mu, sd = sp[tr].mean(0), sp[tr].std(0).clamp(min=1e-6)
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(sp.shape[1], 96), torch.nn.ReLU(),
        torch.nn.Linear(96, 96), torch.nn.ReLU(),
        torch.nn.Linear(96, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    Xn = (sp - mu) / sd
    for _ in range(300):
        opt.zero_grad()
        lossf(net(Xn[tr]), Y[tr]).backward()
        opt.step()
    with torch.no_grad():
        pred = (net(((bridged - mu) / sd)[te]) > 0).float()
    exact = (pred == Y[te]).all(1).float().mean().item()
    print(f"RUNG 2b (strong): native-trained probe reading BRIDGED "
          f"vectors: exact {100 * exact:.1f}%", flush=True)


if __name__ == "__main__":
    main()
