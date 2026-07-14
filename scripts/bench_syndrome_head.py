"""Syndrome head, payoff 3 (the re-aimed spec): does TRAINING-time
rule-awareness improve step validity, with nothing said at inference?

Two arms, identical in every bit except the auxiliary gradient:
same corpus, same balance, same LoRA init seed, same data order —
lam=0 (control) vs lam=0.3 (multi-task). The aux head reads
hidden_states[15] (the measured syndrome plateau, RESULTS layer
sweep) at the last prompt token and predicts the 16 rule-fire bits
(labels from data/pred_syndrome_labels.jsonl; unlabeled rows
contribute zero aux loss). Inference prompting unchanged (hints
off) — any gain is representation shaping, full stop.

Eval: solve_chain on a fresh band, both adapters, same seeds.
Pre-registered (from the spec): interesting if multi-task validity
> control + noise; kill switch if it REGRESSES at this lam.

    .venv/bin/python scripts/bench_syndrome_head.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LAM = 0.3
SYN_LAYER = 15
EVAL_SEED = 9_500_000
EVAL_N, EVAL_BUDGET = 8, 384


def build_examples(tok):
    """phase_train's example building + class balance, verbatim, plus
    per-row syndrome targets threaded through."""
    rows = [json.loads(l)
            for l in Path("data/step_chains.jsonl").read_text().splitlines()]
    syn = {json.loads(l)["s"]: json.loads(l)["fires"]
           for l in Path("data/pred_syndrome_labels.jsonl")
           .read_text().splitlines()}
    vocab = sorted({rn for f in syn.values() for rn in f})
    vi = {r: i for i, r in enumerate(vocab)}

    coeff = [r for r in rows if r.get("source") == "coeff"]
    noncoeff = [r for r in rows if r.get("source") != "coeff"]

    def _one_hop(r):
        return (r["cur"].startswith("Integral(")
                and "Integral" not in r["nxt"]
                and r.get("source") != "skip")
    hops = [r for r in noncoeff if _one_hop(r)]
    rest = [r for r in noncoeff if not _one_hop(r)]
    if len(hops) > 400:
        random.Random(0).shuffle(hops)
        hops = hops[:400]
    if len(coeff) > 250:
        random.Random(1).shuffle(coeff)
        coeff = coeff[:250]
    rows = rest + hops + coeff

    examples = []
    n_lab = 0
    for r in rows:
        hints = ", ".join(r.get("hints") or []) or "none"
        prompt = tok(f"Current: {r['cur']}\nHints: {hints}\nStep:",
                     add_special_tokens=False).input_ids
        target = ((r["think"] + " => ") if r.get("think") else "") + r["nxt"]
        step = tok(" " + target + "\n", add_special_tokens=False).input_ids
        ids = prompt + step
        if len(ids) > 512:
            continue
        labels = [-100] * len(prompt) + step
        bits = None
        if r["cur"] in syn:
            bits = [0.0] * len(vocab)
            for rn in syn[r["cur"]]:
                bits[vi[rn]] = 1.0
            n_lab += 1
        examples.append((ids, labels, len(prompt) - 1, bits))
    examples.sort(key=lambda e: len(e[0]))
    print(f"{len(examples)} pairs, {n_lab} syndrome-labeled, "
          f"{len(vocab)} rules", flush=True)
    return examples, len(vocab)


def train_arm(lam: float, out: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmopt.train.lora import apply_lora
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32).to(dev)
    torch.manual_seed(0)          # identical LoRA init across arms
    apply_lora(model, TARGETS, r=16, alpha=32)
    model.train()
    examples, n_rules = build_examples(tok)
    torch.manual_seed(0)
    head = torch.nn.Linear(896, n_rules).to(dev)
    pad = tok.pad_token_id or tok.eos_token_id
    params = ([p for p in model.parameters() if p.requires_grad]
              + list(head.parameters()))
    opt = torch.optim.AdamW(params, lr=1e-4)
    bce = torch.nn.BCEWithLogitsLoss(reduction="mean")
    # epoch-level resume: environment gremlins (swap-out, QoS
    # demotion during USB transfers, 2026-07-14) must never cost
    # more than one epoch again
    marker = Path(str(out) + ".ep")
    start_ep = 0
    if marker.exists() and out.exists():
        start_ep = int(marker.read_text()) + 1
        model.load_state_dict(
            torch.load(out, weights_only=False, map_location="cpu"),
            strict=False)
        model.to(dev)
        print(f"resuming lam={lam} at epoch {start_ep}", flush=True)
    n_steps_ep = (len(examples) + 7) // 8
    for ep in range(start_ep, 3):
        idx = list(range(len(examples)))
        random.Random(ep).shuffle(idx)   # per-epoch order shuffle
        tot = tot_syn = steps = 0
        t_ep = time.time()
        for i in range(0, len(idx), 8):
            batch = [examples[j] for j in idx[i:i + 8]]
            L = max(len(b[0]) for b in batch)
            ids = torch.tensor([b[0] + [pad] * (L - len(b[0]))
                                for b in batch], device=dev)
            labs = torch.tensor([b[1] + [-100] * (L - len(b[1]))
                                 for b in batch], device=dev)
            mask = torch.tensor([[1] * len(b[0]) + [0] * (L - len(b[0]))
                                 for b in batch], device=dev)
            mo = model(input_ids=ids, attention_mask=mask, labels=labs,
                       output_hidden_states=(lam > 0))
            loss = mo.loss
            if lam > 0:
                labeled = [(k, b) for k, b in enumerate(batch)
                           if b[3] is not None]
                if labeled:
                    hs = mo.hidden_states[SYN_LAYER]
                    z = torch.stack([hs[k, b[2]] for k, b in labeled])
                    y = torch.tensor([b[3] for _, b in labeled],
                                     device=dev)
                    syn_loss = bce(head(z), y)
                    loss = loss + lam * syn_loss
                    tot_syn += float(syn_loss)
            loss.backward()
            opt.step()
            opt.zero_grad()
            tot += float(mo.loss.detach())
            steps += 1
            if steps % 50 == 0:
                rate = steps / (time.time() - t_ep)
                eta = (n_steps_ep - steps) / max(rate, 1e-9) / 60
                print(f"  lam={lam} ep{ep} step {steps}/{n_steps_ep} "
                      f"ce {tot / steps:.3f} "
                      f"({rate:.2f} it/s, ~{eta:.0f}m left)",
                      flush=True)
        print(f"lam={lam} epoch {ep}: ce {tot / steps:.4f} "
              f"syn {tot_syn / steps:.4f}", flush=True)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()
                    if k.split(".")[-1] in ("a", "b")}, out)
        marker.write_text(str(ep))
    marker.unlink(missing_ok=True)
    print(f"saved {out}", flush=True)


def evaluate(adapter: str):
    import sympy as sp

    from bench_step_tokens import _gen_isolated, load, solve_chain
    tok, model = load(adapter)
    solved = valid = tried = 0
    n = 0
    for lv in (2, 3, 4, 5):
        for i in range(EVAL_N):
            p = _gen_isolated(lv, EVAL_SEED + 1000 * lv + i)
            if p is None:
                continue
            n += 1
            ok, _, v, t = solve_chain(tok, model, sp.sstr(p._expr),
                                      EVAL_BUDGET,
                                      seed0=EVAL_SEED + 1000 * lv + i)
            solved += ok
            valid += v
            tried += t
    del model
    return solved, n, 100 * valid / max(tried, 1)


def main() -> None:
    t0 = time.time()
    for lam, out in ((0.0, Path("checkpoints/step_lora_syn0.pt")),
                     (LAM, Path("checkpoints/step_lora_syn3.pt"))):
        incomplete = Path(str(out) + ".ep").exists()
        if not out.exists() or incomplete:
            train_arm(lam, out)
    for name, ad in (("control lam=0", "checkpoints/step_lora_syn0.pt"),
                     (f"multi-task lam={LAM}",
                      "checkpoints/step_lora_syn3.pt")):
        s, n, v = evaluate(ad)
        print(f"{name}: solves {s}/{n} validity {v:.2f}%", flush=True)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
