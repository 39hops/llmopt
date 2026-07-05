"""LoRA fine-tune Qwen2.5-0.5B-Instruct on generated, sympy-verified calculus.

Data is generated (mathgen/), never downloaded; the eval metric is
symbolic equivalence, never string match. Baseline eval -> LoRA train
(loss on the answer tokens only) -> eval again. Adapter saved to
checkpoints/calculus_lora.pt.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.mathgen.evaluate import SYSTEM, evaluate_model
from llmopt.mathgen.problems import make_dataset
from llmopt.train.lora import apply_lora

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_TRAIN, N_EVAL = 9000, 300
EPOCHS, BATCH, LR = 3, 8, 2e-4
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
RANK = 16
OUT = Path("checkpoints/calculus_lora.pt")


def encode(tok, problem):
    """input_ids + labels (-100 on everything but the answer tokens)."""
    from llmopt.mathgen.evaluate import format_chat

    prompt = format_chat(tok, problem)
    answer = tok(problem.answer + "\n", add_special_tokens=False).input_ids
    answer.append(tok.eos_token_id)
    ids = prompt + answer
    labels = [-100] * len(prompt) + answer
    return ids, labels


def batches(examples, pad_id, batch_size, device, epoch=0):
    """Length-bucketed (little padding) but order-shuffled (decorrelated):
    examples arrive length-sorted; batch composition stays, batch order is
    re-shuffled every epoch."""
    import random as _random

    order = list(range(0, len(examples), batch_size))
    _random.Random(epoch).shuffle(order)
    for i in order:
        chunk = examples[i : i + batch_size]
        width = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids)
        for j, (seq, lab) in enumerate(chunk):
            ids[j, : len(seq)] = torch.tensor(seq)
            labels[j, : len(lab)] = torch.tensor(lab)
            mask[j, : len(seq)] = 1
        yield ids.to(device), labels.to(device), mask.to(device)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16
    ).cuda()

    eval_problems = make_dataset(N_EVAL, seed=99)
    banned = frozenset(p.prompt for p in eval_problems)
    print("baseline eval (symbolic equivalence):")
    model.eval()
    for k, v in evaluate_model(model, tok, eval_problems).items():
        print(f"  {k:20s} {v:6.1%}")

    wrapped = apply_lora(model, TARGETS, r=RANK, alpha=2 * RANK)
    train_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\nLoRA: wrapped {wrapped} linears, "
          f"{sum(p.numel() for p in train_params) / 1e6:.1f}M trainable params")

    train = [
        encode(tok, p)
        for p in make_dataset(N_TRAIN, seed=0, exclude=banned)
    ]
    train.sort(key=lambda e: len(e[0]))  # length-sorted batches: less padding
    opt = torch.optim.AdamW(train_params, lr=LR)
    steps_total = EPOCHS * (len(train) // BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_total)

    model.train()
    step = 0
    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        for ids, labels, mask in batches(
            train, tok.pad_token_id, BATCH, model.device, epoch=epoch
        ):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{steps_total} "
                      f"loss {loss.item():.4f} "
                      f"({step / (time.perf_counter() - t0):.1f} steps/s)")

    print("\npost-training eval:")
    model.eval()
    for k, v in evaluate_model(model, tok, eval_problems).items():
        print(f"  {k:20s} {v:6.1%}")

    OUT.parent.mkdir(exist_ok=True)
    torch.save(
        {k: v for k, v in model.state_dict().items()
         if "a" == k.split(".")[-1] or "b" == k.split(".")[-1]},
        OUT,
    )
    print(f"\nadapter saved: {OUT}")


if __name__ == "__main__":
    main()
