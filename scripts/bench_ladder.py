"""0.5B capability ladder: cold vs LoRA-tuned accuracy per rung.

Which rung does a small model climb to? Hypothesis: the learned-mapping
rungs (encode/decode/diagnose) train up fast; the simulation rungs
(output, -O2 asm) resist. Every answer is scored by the toolchain
(ladder.py), never by string proximity.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.codegen.ladder import RUNGS, build_ladder, evaluate_ladder
from llmopt.train.lora import apply_lora

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_PROGRAMS, EVAL_PROGRAMS = 400, 60
CAP_TRAIN, CAP_EVAL = 1200, 50      # per rung
EPOCHS, BATCH, LR, RANK = 2, 8, 2e-4, 16
SYSTEM = (
    "You are a compiler and machine-code expert. Answer with exactly what "
    "is asked — bytes, one instruction, program output, or assembly — and "
    "nothing else."
)


def format_chat(tok, prompt):
    text = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False,
    )
    return tok(text, add_special_tokens=False).input_ids


def make_generate_fn(model, tok, max_new_tokens=96):
    def generate(prompt: str) -> str:
        ids = torch.tensor([format_chat(tok, prompt)], device=model.device)
        with torch.inference_mode():
            out = model.generate(
                input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return generate


def encode_example(tok, task):
    prompt = format_chat(tok, task.prompt)
    answer = tok(task.target + "\n", add_special_tokens=False).input_ids
    answer.append(tok.eos_token_id)
    return prompt + answer, [-100] * len(prompt) + answer


def batches(examples, pad_id, batch_size, device, epoch):
    import random

    order = list(range(0, len(examples), batch_size))
    random.Random(epoch).shuffle(order)
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
    print("building eval ladder...")
    t0 = time.perf_counter()
    import random as _random

    eval_tasks = build_ladder(EVAL_PROGRAMS, seed=99)
    for ts in eval_tasks.values():  # shuffle before capping: keep the
        _random.Random(0).shuffle(ts)  # clean/buggy diagnose mix balanced
    eval_tasks = {r: ts[:CAP_EVAL] for r, ts in eval_tasks.items()}
    banned = frozenset(t.prompt for ts in eval_tasks.values() for t in ts)
    print("building train ladder...")
    train_tasks = build_ladder(TRAIN_PROGRAMS, seed=0, exclude=banned)
    train_tasks = {r: ts[:CAP_TRAIN] for r, ts in train_tasks.items()}
    print(f"data built in {time.perf_counter() - t0:.0f}s: "
          + ", ".join(f"{r}={len(ts)}/{len(eval_tasks[r])}"
                      for r, ts in train_tasks.items()))

    tok = AutoTokenizer.from_pretrained(MODEL)
    device = "cuda" if torch.cuda.is_available() else "mps"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    model.eval()

    print("\ncold eval:")
    cold = evaluate_ladder(make_generate_fn(model, tok), eval_tasks)
    for r in RUNGS:
        print(f"  {r:10s} {cold.get(r, 0):6.1%}")

    apply_lora(model, ("q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"), r=RANK, alpha=2 * RANK)
    params = [p for p in model.parameters() if p.requires_grad]
    train = [encode_example(tok, t) for ts in train_tasks.values() for t in ts]
    train.sort(key=lambda e: len(e[0]))
    opt = torch.optim.AdamW(params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS * (len(train) // BATCH)
    )
    print(f"\ntraining on {len(train)} examples...")
    model.train()
    step = 0
    for epoch in range(EPOCHS):
        for ids, labels, mask in batches(
            train, tok.pad_token_id or tok.eos_token_id, BATCH, model.device, epoch
        ):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 100 == 0:
                print(f"  step {step} loss {loss.item():.4f}")

    print("\ntuned eval:")
    model.eval()
    tuned = evaluate_ladder(make_generate_fn(model, tok), eval_tasks)
    print(f"\n{'rung':<10} {'cold':>7} {'tuned':>7}")
    for r in RUNGS:
        print(f"{r:<10} {cold.get(r, 0):6.1%} {tuned.get(r, 0):6.1%}")


if __name__ == "__main__":
    main()
