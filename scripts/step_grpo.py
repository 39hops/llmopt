"""GRPO at the frontier band — sustained RL over verified steps.

Spec: docs/superpowers/specs/2026-07-14-step-grpo-design.md.
Waves ARE groups (B=8 resamples at a stuck state, sample_batch);
the verifier is the reward (1.0 verified step, 0.0 not); ONLY mixed
groups train (all-fail/all-pass have zero group-relative advantage
— the gradient lives where variance lives). Incremental climb from
the promoted adapter: no from-scratch retrain, no reallocation
lottery (measured five times this week, incl. the own-diet confirm:
even byte-identical diets re-roll the level allocation).

Loop: collect GROUPS_PER_CYCLE mixed groups (walking chains with
the current policy at the frontier band) -> one PPO-clip epoch ->
every GATE_EVERY cycles, trimmed evaluate on a FIXED band ->
checkpoint if validity holds, else rollback + halve lr (one retry,
then halt). Verified steps found during collection append to the
corpus (source "grpo") — mining stays free.

    .venv/bin/python -u scripts/step_grpo.py --cycles 8
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

B = 8
GROUPS_PER_CYCLE = 64  # --groups overrides (smoke tests)
GATE_EVERY = 2          # spec said 4; v0 gates twice as often (safety)
LR0 = 5e-6
CLIP = 0.2
LEVELS = (2, 3, 4, 5)
SEED_BASE = 42_000_000  # collection band
GATE_BAND = 8_600_000   # fixed gate band (fresh)
CKPT = Path("checkpoints/step_lora_grpo.pt")
CORPUS = Path("data/step_chains.jsonl")


def collect_groups(tok, model, n_groups: int, seed0: int):
    """Walk chains with the current policy; keep mixed waves."""
    import sympy as sp

    from bench_step_tokens import (FEWSHOT, _gen_isolated, sample_batch,
                                   verify_step)
    groups, mined = [], []
    stats = {"waves": 0, "mixed": 0, "allfail": 0, "allpass": 0}
    pi = 0
    while len(groups) < n_groups:
        lv = LEVELS[pi % len(LEVELS)]
        p = _gen_isolated(lv, seed0 + pi)
        pi += 1
        if p is None:
            continue
        cur = f"Integral({sp.sstr(p._expr)}, x)"
        for ply in range(12):
            prompt = FEWSHOT + f"\nCurrent: {cur}\nHints: none\nStep:"
            texts, _sp, tok_ids, logps = sample_batch(
                tok, model, prompt,
                seeds=[seed0 * 31 + pi * 977 + ply * 101 + b
                       for b in range(B)],
                constrain=True, return_logps=True)
            stats["waves"] += 1
            # dedup-then-verify; map verdicts back to streams
            lines = [t.splitlines()[0].strip() if t else ""
                     for t in texts]
            verdict: dict[str, tuple[bool, bool]] = {}
            for ln in set(lines):
                expr = ln.split("=>")[-1].strip()
                if not ln or not expr:
                    verdict[ln] = (False, False)
                else:
                    verdict[ln] = verify_step(cur, expr)
            rewards = [1.0 if verdict[ln][0] else 0.0 for ln in lines]
            n_ok = sum(1 for r in rewards if r > 0)
            if 0 < n_ok < B:
                stats["mixed"] += 1
                groups.append({"prompt": prompt, "tok_ids": tok_ids,
                               "logps": logps, "rewards": rewards})
            elif n_ok == 0:
                stats["allfail"] += 1
            else:
                stats["allpass"] += 1
            # advance the chain along a verified step (prefer solving)
            nxt_state = None
            for ln in lines:
                ok, solved = verdict[ln]
                if ok:
                    expr = ln.split("=>")[-1].strip()
                    mined.append({"cur": cur, "nxt": expr,
                                  "level": lv, "source": "grpo"})
                    if solved:
                        nxt_state = "SOLVED"
                        break
                    nxt_state = expr
            if nxt_state in (None, "SOLVED"):
                break
            cur = nxt_state
            if len(groups) >= n_groups:
                break
    return groups, mined, stats


def logp_new(tok, model, group, device):
    """Teacher-forced logp of each stream's completion under the
    CURRENT policy — same temperature (0.7) and charset mask as
    sampling, or the ratio is meaningless."""
    import torch

    from bench_step_tokens import _expr_mask
    enc = tok(group["prompt"], return_tensors="pt").input_ids[0]
    seqs = [list(enc) + g for g in group["tok_ids"]]
    L = max(len(s) for s in seqs)
    pad = tok.eos_token_id
    ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs],
                       device=device)
    mask = torch.tensor([[1] * len(s) + [0] * (L - len(s))
                         for s in seqs], device=device)
    logits = model(input_ids=ids, attention_mask=mask).logits
    m = _expr_mask(tok)
    if m.shape[0] < logits.shape[-1]:
        m = torch.cat([m, torch.zeros(
            logits.shape[-1] - m.shape[0], dtype=torch.bool)])
    out = []
    plen = len(enc)
    for b, g in enumerate(group["tok_ids"]):
        if not g:
            out.append(torch.tensor(0.0, device=device))
            continue
        lg = logits[b, plen - 1: plen - 1 + len(g)].float() / 0.7
        lg = lg.masked_fill(~m.to(device), float("-inf"))
        lp = torch.log_softmax(lg, -1)
        idx = torch.tensor(g, device=device)
        out.append(lp.gather(1, idx[:, None]).sum())
    return torch.stack(out)


def gate_eval(adapter: str):
    from expert_loop import evaluate
    from bench_step_tokens import load
    tok, model = load(adapter)
    sb = evaluate(tok, model, levels=LEVELS, n_per=24,
                  seed_base=GATE_BAND, budget=512)
    del model
    return sb


def main(cycles: int, groups_per_cycle: int = GROUPS_PER_CYCLE,
         skip_baseline: bool = False) -> None:
    import torch

    from llmopt.train.preference import grpo_advantages, grpo_loss
    from bench_step_tokens import load
    import bench_step_tokens as bst

    tok, model = load("checkpoints/step_lora.pt")
    device = bst.DEV
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    lr = LR0
    opt = torch.optim.Adam(params, lr=lr)

    if skip_baseline:
        best_validity = 0.0  # smoke mode: no gate protection
    else:
        print("baseline gate eval...", flush=True)
        model.eval()
        base = gate_eval("checkpoints/step_lora.pt")
        print(f"baseline: {base['solves']} validity "
              f"{base['validity']:.2f}%", flush=True)
        best_validity = base["validity"]
    torch.save({k: v.cpu() for k, v in model.state_dict().items()
                if k.split(".")[-1] in ("a", "b")}, CKPT)
    tok, model = load(str(CKPT))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    retried = False

    for cyc in range(1, cycles + 1):
        t0 = time.time()
        model.eval()
        groups, mined, stats = collect_groups(
            tok, model, groups_per_cycle, SEED_BASE + 100_000 * cyc)
        with CORPUS.open("a") as f:
            for r in mined:
                f.write(json.dumps(r) + "\n")
        mix_rate = stats["mixed"] / max(stats["waves"], 1)
        print(f"cycle {cyc}: {stats} mix-rate {mix_rate:.2f} "
              f"mined +{len(mined)} ({time.time() - t0:.0f}s)",
              flush=True)

        model.train()
        order = list(range(len(groups)))
        random.Random(cyc).shuffle(order)
        tot = n = 0
        for gi in order:
            g = groups[gi]
            lp_new = logp_new(tok, model, g, device)
            lp_old = torch.tensor(g["logps"], device=device)
            rw = torch.tensor(g["rewards"], device=device)
            gid = torch.zeros(B, dtype=torch.long, device=device)
            adv = grpo_advantages(rw, gid)
            loss = grpo_loss(lp_new, lp_old, adv, clip=CLIP)
            loss.backward()
            opt.step()
            opt.zero_grad()
            tot += float(loss.detach())
            n += 1
        print(f"cycle {cyc}: grpo loss {tot / max(n, 1):.4f} "
              f"({n} groups)", flush=True)

        if cyc % GATE_EVERY == 0:
            tmp = Path("checkpoints/step_lora_grpo_tmp.pt")
            torch.save({k: v.cpu() for k, v in model.state_dict().items()
                        if k.split(".")[-1] in ("a", "b")}, tmp)
            model.eval()
            sb = gate_eval(str(tmp))
            print(f"cycle {cyc} gate: {sb['solves']} validity "
                  f"{sb['validity']:.2f}% (best {best_validity:.2f})",
                  flush=True)
            if sb["validity"] >= best_validity - 0.1:
                best_validity = max(best_validity, sb["validity"])
                tmp.replace(CKPT)
                retried = False
                print(f"  checkpointed -> {CKPT}", flush=True)
            else:
                print("  ROLLBACK to last checkpoint, lr halved",
                      flush=True)
                tok, model = load(str(CKPT))
                params = [p for p in model.parameters()
                          if p.requires_grad]
                lr /= 2
                opt = torch.optim.Adam(params, lr=lr)
                if retried:
                    print("HALT: two consecutive gate rollbacks",
                          flush=True)
                    break
                retried = True
    print("GRPO run complete", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--groups", type=int, default=GROUPS_PER_CYCLE)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.cycles, a.groups, skip_baseline=a.smoke)
