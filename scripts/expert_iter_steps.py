"""Step-level expert iteration (the repo's founding long-term goal,
first concrete round; step-tokens measured 5/30 vs one-shot 0/30 at 5%
step validity — this trains the 5%).

Round 1 = imitation seeding: the ENGINE's winning derivations are
verified step chains by construction (replay_verify's walk yields the
intermediate expressions). Train the 0.5B to emit one rewrite per call
in the bench_step_tokens format; the eval is the same oracle-gated
race on fresh seeds. Round 2+ (later): mix in the model's OWN verified
chains at rising difficulty — generator as self-play opponent.

  --phase chains : fork-isolated engine solves (L2-L5), replay winning
                   histories, emit (cur, nxt) sympy-text pairs
  --phase train  : LoRA (train_calculus recipe: r=16 all-proj, loss on
                   step tokens only, length-bucketed, order-shuffled)
  --phase race   : bench_step_tokens arms with the adapter loaded,
                   fresh seeds — bar: validity% AND solves > base
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

CHAINS = Path("data/step_chains.jsonl")
ADAPTER = Path("checkpoints/step_lora.pt")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")


def _chain_worker(level: int, seed: int, q: "mp.Queue") -> None:
    import sympy as sp

    from llmopt.mathgen.problems import make_integrate
    from llmopt.search import derivation as D
    from llmopt.search.derivation import State, successors
    from llmopt.search.engine import solve

    p = make_integrate(level, seed)
    root = sp.Integral(p._expr, sp.Symbol("x"))
    res = solve(root, budget=200)
    if not res.solved:
        q.put([])
        return
    # replay the winning history, collecting the expression chain
    # (replay_verify's backtracking walk, kept verbatim: labels are
    # not unique across siblings)
    saved, D.RULE_WALL = D.RULE_WALL, 60.0
    hist = res.state.history
    out: list[tuple[str, str]] = []

    def walk(cur: State, i: int, acc: list) -> bool:
        if i == len(hist):
            out.extend(acc)
            return True
        for name, child in successors(cur, use_macros=True, verify_p=1.0):
            if name == hist[i] and walk(
                    child, i + 1,
                    acc + [(sp.sstr(cur.expr), sp.sstr(child.expr))]):
                return True
        return False

    walk(State(root), 0, [])
    D.RULE_WALL = saved
    q.put(out)


def phase_chains(n_per_level: int, seed_base: int) -> None:
    ctx = mp.get_context("fork")
    seen: set = set()
    n = 0
    with CHAINS.open("w") as f:
        for level in (2, 3, 4, 5):
            for i in range(n_per_level):
                q = ctx.Queue()
                pr = ctx.Process(target=_chain_worker,
                                 args=(level, seed_base + i, q))
                pr.start()
                pr.join(90)
                if pr.is_alive():
                    pr.kill()
                    pr.join()
                    continue
                try:
                    pairs = q.get(timeout=10)
                except Exception:
                    continue
                for cur, nxt in pairs:
                    if (cur, nxt) in seen:
                        continue
                    seen.add((cur, nxt))
                    f.write(json.dumps({"cur": cur, "nxt": nxt,
                                        "level": level}) + "\n")
                    n += 1
            print(f"L{level} done: {n} pairs total", flush=True)
    print(f"CHAINS done: {n} verified step pairs -> {CHAINS}")


def phase_train(epochs: int, lr: float) -> None:
    import sys
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_calculus import batches
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmopt.train.lora import apply_lora

    device = ("cuda" if torch.cuda.is_available() else "mps")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32).to(device)
    apply_lora(model, TARGETS, r=16, alpha=32)
    model.train()

    rows = [json.loads(l) for l in CHAINS.read_text().splitlines()]
    examples = []
    for r in rows:
        # the bench_step_tokens plain-completion format, verbatim
        prompt = tok(f"Current: {r['cur']}\nStep:",
                     add_special_tokens=False).input_ids
        step = tok(" " + r["nxt"] + "\n",
                   add_special_tokens=False).input_ids
        ids = prompt + step
        if len(ids) > 512:
            continue
        labels = [-100] * len(prompt) + step
        examples.append((ids, labels))
    examples.sort(key=lambda e: len(e[0]))
    print(f"{len(examples)} training pairs")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    for ep in range(epochs):
        tot = steps = 0
        for ids, labels, mask in batches(examples, tok.pad_token_id
                                         or tok.eos_token_id, 8, device,
                                         epoch=ep):
            out = model(input_ids=ids, attention_mask=mask,
                        labels=labels)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            tot += float(out.loss)
            steps += 1
        print(f"epoch {ep}: loss {tot / max(steps, 1):.4f}", flush=True)
    # the train_calculus save convention: raw {**.a, **.b} adapter dict
    torch.save({k: v.cpu() for k, v in model.state_dict().items()
                if k.split(".")[-1] in ("a", "b")}, ADAPTER)
    print(f"saved adapter -> {ADAPTER}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["chains", "train"])
    ap.add_argument("--n-per-level", type=int, default=150)
    ap.add_argument("--seed-base", type=int, default=8_000_000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    if a.phase == "chains":
        phase_chains(a.n_per_level, a.seed_base)
    else:
        phase_train(a.epochs, a.lr)
