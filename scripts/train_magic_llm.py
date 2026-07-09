"""(d) LLM-trunk magic estimator: the 0.5B proposer trunk replaces
the 20 hand features. Same labels, same seed-parity split as the MLP
(train_magic_estimator.py, rho 0.855 baseline), integrand string in,
log2(1+nodes) + solved out. Frozen trunk by default (--unfreeze-lora
for the joint version). Note the Bayes-floor finding: the 20 features
already carry ~99% of explainable variance, so this tests whether a
language trunk converts that variance to rank accuracy better than a
64x64 MLP — capacity, not representation."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")


def spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((p - ma) * (q - mb) for p, q in zip(ra, rb))
    da = math.sqrt(sum((p - ma) ** 2 for p in ra))
    db = math.sqrt(sum((q - mb) ** 2 for q in rb))
    return num / (da * db) if da and db else 0.0


def main(labels: Path, epochs: int, batch: int, unfreeze: bool) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmopt.train.lora import apply_lora

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(device)
    apply_lora(model, TARGETS, r=16, alpha=32)
    model.load_state_dict(
        torch.load("checkpoints/proposer_lora.pt", weights_only=True,
                   map_location="cpu"), strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    lora_params = []
    if unfreeze:
        for name, p in model.named_parameters():
            if name.endswith((".a", ".b")):
                p.requires_grad_(True)
                lora_params.append(p)

    rows = [json.loads(l) for l in labels.read_text().splitlines()]
    train = [r for r in rows if r["seed"] % 2 == 0]
    test = [r for r in rows if r["seed"] % 2 == 1]
    print(f"{len(train)} train / {len(test)} test (seed-parity split, "
          f"same as MLP)")

    d = model.config.hidden_size
    head = torch.nn.Sequential(
        torch.nn.Linear(d, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU()).to(device).float()
    h_cost = torch.nn.Linear(64, 1).to(device).float()
    h_solv = torch.nn.Linear(64, 1).to(device).float()
    opt = torch.optim.Adam(
        [{"params": list(head.parameters()) + list(h_cost.parameters())
          + list(h_solv.parameters()), "lr": 1e-3}]
        + ([{"params": lora_params, "lr": 1e-4}] if lora_params else []))
    bce = torch.nn.BCEWithLogitsLoss()

    def hidden(strs, grad=False):
        enc = tok(strs, return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(device)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = model.model(input_ids=enc.input_ids,
                              attention_mask=enc.attention_mask)
        h = out.last_hidden_state
        idx = enc.attention_mask.sum(1) - 1
        return h[torch.arange(h.shape[0]), idx].float()

    yc = [math.log2(1 + r["nodes"]) for r in train]
    mu = sum(yc) / len(yc)
    sd = (sum((v - mu) ** 2 for v in yc) / len(yc)) ** 0.5
    rng = random.Random("magic-llm-0")
    for ep in range(epochs):
        idx = list(range(len(train)))
        rng.shuffle(idx)
        tot = nb = 0
        for i in range(0, len(idx), batch):
            ch = idx[i:i + batch]
            h = head(hidden([train[j]["integrand"] for j in ch],
                            grad=unfreeze))
            c = torch.tensor([(math.log2(1 + train[j]["nodes"]) - mu) / sd
                              for j in ch], device=device)
            s = torch.tensor([float(train[j]["solved"]) for j in ch],
                             device=device)
            loss = (torch.nn.functional.mse_loss(
                        h_cost(h).squeeze(-1), c)
                    + bce(h_solv(h).squeeze(-1), s))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        print(f"epoch {ep}: loss {tot / nb:.4f}", flush=True)

    preds_c, preds_s = [], []
    for i in range(0, len(test), batch):
        h = head(hidden([r["integrand"] for r in test[i:i + batch]]))
        preds_c += h_cost(h).squeeze(-1).detach().cpu().tolist()
        preds_s += h_solv(h).squeeze(-1).detach().cpu().tolist()
    true_c = [math.log2(1 + r["nodes"]) for r in test]
    rho = spearman(preds_c, true_c)
    pos = [preds_s[i] for i in range(len(test)) if test[i]["solved"]]
    neg = [preds_s[i] for i in range(len(test)) if not test[i]["solved"]]
    auc = (sum(1 for p in pos for q in neg if p > q)
           / (len(pos) * len(neg))) if pos and neg else float("nan")
    print(f"held-out rho {rho:.3f} (MLP baseline 0.855), AUC {auc:.3f} "
          f"(MLP 0.975)")
    import torch as _t
    _t.save({"head": head.state_dict(), "h_cost": h_cost.state_dict(),
             "h_solv": h_solv.state_dict(), "mu": mu, "sd": sd},
            "checkpoints/magic_llm.pt")
    print("saved checkpoints/magic_llm.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path,
                    default=Path("data/magic_labels.jsonl"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--unfreeze-lora", action="store_true")
    a = ap.parse_args()
    main(a.labels, a.epochs, a.batch, a.unfreeze_lora)
