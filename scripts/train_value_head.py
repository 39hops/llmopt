"""Fused value head (Artin's architecture, 2026-07-08): one trunk,
two heads, one forward pass. The transformer body replaces NNUE's 20
hand-crafted features — the value head is a tiny MLP (d_model->64->1)
on the last hidden state of the state string, trained on the same
probe labels (log2 nodes-to-solve) the NNUE used. The LM head keeps
ranking moves; value now comes ~free from the hidden state the ranker
already computed. AlphaZero's policy+value fusion on a language trunk.

Phase 1 (CPU):  --gen-labels   probe-label states, SAVE the dataset
                               (data/value_labels.jsonl — reusable,
                               unlike train_nnue's on-the-fly labels).
Phase 2 (GPU):  --train        train the head (trunk + LoRA frozen).
Question it answers: does a language trunk beat hand features as the
eval's representation at equal search cost? (bench follows.)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
from pathlib import Path

import sympy as sp

X = sp.Symbol("x")
PROBE_NODES = 200
PROBE_SECONDS = 60
FAIL_LABEL = 9.0
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")
LABELS_PATH = Path("data/value_labels.jsonl")


class _Timeout(BaseException):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def gen_labels(per_cell: int, cap: int) -> None:
    import sys
    sys.path.insert(0, "scripts")
    from train_nnue import collect_states, label

    states, _roots = collect_states("train", per_cell, cap)
    print(f"{len(states)} states to label")
    with LABELS_PATH.open("w") as f:
        for i, s in enumerate(states):
            y = label(s)
            f.write(json.dumps({"state": sp.sstr(s.expr), "y": y}) + "\n")
            if (i + 1) % 100 == 0:
                print(f"labeled {i + 1}/{len(states)}", flush=True)
    print(f"saved {LABELS_PATH}")


def train_head(epochs: int, batch: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmopt.train.lora import apply_lora

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
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

    rows = [json.loads(l) for l in LABELS_PATH.open()]
    ys = torch.tensor([r["y"] for r in rows], dtype=torch.float32)
    mean, std = ys.mean().item(), ys.std().item()

    d_model = model.config.hidden_size
    head = torch.nn.Sequential(
        torch.nn.Linear(d_model, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 1)).to(device).to(torch.float32)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)

    def hidden(states: list[str]) -> "torch.Tensor":
        enc = tok(states, return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.model(input_ids=enc.input_ids,
                              attention_mask=enc.attention_mask)
        h = out.last_hidden_state
        # last REAL token per row (right padding)
        idx = enc.attention_mask.sum(1) - 1
        return h[torch.arange(h.shape[0]), idx].to(torch.float32)

    order = list(range(len(rows)))
    rng = random.Random("value-head-0")
    n_eval = max(1, len(rows) // 10)
    eval_idx = set(order[-n_eval:])
    train_idx = [i for i in order if i not in eval_idx]
    for ep in range(epochs):
        rng.shuffle(train_idx)
        tot = nb = 0
        for i in range(0, len(train_idx), batch):
            chunk = train_idx[i:i + batch]
            hcat = hidden([rows[j]["state"] for j in chunk])
            y = torch.tensor([(rows[j]["y"] - mean) / std for j in chunk],
                             device=device)
            loss = torch.nn.functional.mse_loss(head(hcat).squeeze(-1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        print(f"epoch {ep}: mse {tot / nb:.4f}", flush=True)
    # held-out spearman
    import sys
    sys.path.insert(0, "scripts")
    from train_nnue import spearman
    ev = sorted(eval_idx)
    with_h = hidden([rows[j]["state"] for j in ev])
    pred = head(with_h).squeeze(-1).detach().cpu().tolist()
    true = [rows[j]["y"] for j in ev]
    print(f"held-out spearman rho = {spearman(pred, true):+.3f} "
          f"(NNUE baseline: +0.937 on its own split)")
    import torch as _t
    _t.save({"state_dict": head.state_dict(), "mean": mean, "std": std},
            "checkpoints/value_head.pt")
    print("saved checkpoints/value_head.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-labels", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--per-cell", type=int, default=15)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    signal.signal(signal.SIGALRM, _alarm)
    if a.gen_labels:
        gen_labels(a.per_cell, a.cap)
    if a.train:
        train_head(a.epochs, a.batch)
