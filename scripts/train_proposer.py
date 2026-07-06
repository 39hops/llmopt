"""Proposer SFT: choose the winning move number given state + legal
moves. Recipe verbatim from scripts/train_calculus.py (LoRA r=16 all
proj linears, loss on answer tokens only, length-sorted token-budget
batches, per-epoch cut shuffle, cosine schedule). Spec:
2026-07-07-move-proposer-design.md. Runs on CUDA (3080) or MPS."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.search.proposer import build_prompt
from llmopt.train.lora import apply_lora

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EPOCHS, BATCH, LR = 3, 8, 2e-4
TOKEN_BUDGET = 2048
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")
RANK = 16
OUT = Path("checkpoints/proposer_lora.pt")  # overridable via --out


def encode(tok, row):
    prompt_ids = tok(build_prompt(row["state"], row["moves"]),
                     add_special_tokens=False).input_ids
    ans = tok(f" {row['answer'] + 1}", add_special_tokens=False).input_ids
    ans.append(tok.eos_token_id)
    ids = prompt_ids + ans
    labels = [-100] * len(prompt_ids) + ans
    return ids, labels


def cut_batches(examples, batch_size, token_budget):
    cuts, i = [], 0
    while i < len(examples):
        j = i + 1
        while (j < len(examples) and j - i < batch_size
               and len(examples[j][0]) * (j - i + 1) <= token_budget):
            j += 1
        cuts.append((i, j))
        i = j
    return cuts


def batches(examples, pad_id, device, epoch):
    cuts = cut_batches(examples, BATCH, TOKEN_BUDGET)
    random.Random(epoch).shuffle(cuts)
    for i, j in cuts:
        chunk = examples[i:j]
        width = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids)
        for r, (seq, lab) in enumerate(chunk):
            ids[r, : len(seq)] = torch.tensor(seq)
            labels[r, : len(lab)] = torch.tensor(lab)
            mask[r, : len(seq)] = 1
        yield ids.to(device), labels.to(device), mask.to(device)


@torch.no_grad()
def move_accuracy(model, tok, rows, device, k=(1, 3)):
    from llmopt.search.proposer import hf_score_fn

    score = hf_score_fn(model, tok, device)
    hits = {kk: 0 for kk in k}
    for row in rows:
        s = score(row["state"], row["moves"])
        order = sorted(range(len(s)), key=lambda i: -s[i])
        for kk in k:
            hits[kk] += row["answer"] in order[:kk]
    return {kk: hits[kk] / len(rows) for kk in k}


def main(extra_data: list[str] | None = None, out: Path = OUT) -> None:
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(device)

    train_rows = [json.loads(l) for l in open("data/proposer_train.jsonl")]
    for path in extra_data or []:
        extra = [json.loads(l) for l in open(path)]
        print(f"+ {len(extra)} rows from {path}")
        train_rows += extra
    eval_rows = [json.loads(l) for l in open("data/proposer_eval.jsonl")]
    eval_rows = eval_rows[:300]
    print(f"train rows: {len(train_rows)}, eval rows: {len(eval_rows)}")

    model.eval()
    base_acc = move_accuracy(model, tok, eval_rows, device)
    print(f"baseline move accuracy: top1={base_acc[1]:.1%} top3={base_acc[3]:.1%}",
          flush=True)

    wrapped = apply_lora(model, TARGETS, r=RANK, alpha=2 * RANK)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"LoRA wrapped {wrapped} linears, "
          f"{sum(p.numel() for p in params) / 1e6:.1f}M trainable")

    train = [encode(tok, r) for r in train_rows]
    train.sort(key=lambda e: len(e[0]))
    opt = torch.optim.AdamW(params, lr=LR)
    steps = EPOCHS * len(cut_batches(train, BATCH, TOKEN_BUDGET))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    pad = tok.pad_token_id or tok.eos_token_id

    model.train()
    for epoch in range(EPOCHS):
        tot, nb = 0.0, 0
        for ids, labels, mask in batches(train, pad, device, epoch):
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            out.loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            tot += float(out.loss)
            nb += 1
        print(f"epoch {epoch}: mean loss {tot / nb:.4f}", flush=True)

    model.eval()
    acc = move_accuracy(model, tok, eval_rows, device)
    print(f"tuned move accuracy: top1={acc[1]:.1%} top3={acc[3]:.1%}")

    # train/lora.py names adapter params `.a` / `.b` inside LoRALinear —
    # save exactly the trainable set, keyed as in state_dict
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    lora_state = {k: v for k, v in model.state_dict().items() if k in trainable}
    assert lora_state, "no trainable params found — adapter naming changed?"
    out.parent.mkdir(exist_ok=True)
    torch.save(lora_state, out)
    print(f"saved {out} ({len(lora_state)} tensors)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-data", nargs="+", default=None,
                    help="additional winning-path jsonl files (e.g. frontier)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    main(extra_data=a.extra_data, out=a.out)
