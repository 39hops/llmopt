"""SWAP-LADDER-1 instrument run (pre-reg RESULTS 2026-08-14): the
hard-first ladder with ONE swap — L3 and L4 exchange positions:
L8,L3,L6,L7,L5,L4,L2,L1. Thin sibling of the frozen
birth19m_curric.py (results-cited, not edited): imports its
helpers and its assert_noop, which must PASS in-process before
training (receipt tee'd by the launcher to logs/curric1/
noop_swap.log). Same BIRTH_SEED=2, 15,420 steps, stock OneCycle,
mps; level-pure batches, within-level shuffled, stream truncated
to the stock per-epoch count (truncation falls on L1's tail,
~0.3% of that level). Appends arm="swap" to logs/curric1/swap.jsonl (arms.jsonl is
a frozen receipt).

Usage: .venv/bin/python scratch/birth19m_curric_rev.py
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

os.environ["ARM"] = "off"       # frozen module import side-effects only
os.environ["BIRTH_SEED"] = "2"
OUT = Path("checkpoints/gallery19m_curric_swap_s2.pt")
if OUT.exists():
    raise SystemExit(f"REFUSING: {OUT} exists")

import torch  # noqa: E402

import birth19m_curric as C  # noqa: E402  (frozen, import-only)
import train_mathnative as TM  # noqa: E402

SWAP_LADDER = [8, 3, 6, 7, 5, 4, 2, 1]
EPOCHS, BS = C.EPOCHS, C.BS
RECEIPTS = Path("logs/curric1/swap.jsonl")


def main():
    tok = TM.MathTokenizer()
    rows = C.load_excised_rows()
    enc, levels = C.encode_with_levels(rows, tok)
    print(f"[swap] {len(enc)} sequences, vocab {len(tok.vocab)}",
          flush=True)
    C.assert_noop(enc)          # precondition, fresh, in-process

    dev = ("mps" if torch.backends.mps.is_available() else
           "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2)
    model = TM.build_model(len(tok.vocab), d=384, layers=8,
                           heads=6, ffn=1536).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                            weight_decay=0.01)
    steps_per_epoch = len(C.stock_epoch_stream(len(enc), 0))
    steps_total = EPOCHS * (len(enc) // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4, total_steps=steps_total, pct_start=0.03)
    print(f"[swap] steps_total {steps_total} ({steps_per_epoch}/epoch)",
          flush=True)

    step = 0
    t0 = time.time()
    for ep in range(EPOCHS):
        lb = C.level_batches(levels, f"curric-swap-{ep}")
        stream = [b for l in SWAP_LADDER for b in lb.get(l, [])]
        dropped = len(stream) - steps_per_epoch
        stream = stream[:steps_per_epoch]
        print(f"[swap] ep{ep}: {len(stream)} batches ({dropped} tail "
              f"batches dropped to match stock count)", flush=True)
        for b in stream:
            batch = [enc[j] for j in b]
            L = max(len(s) for s in batch)
            ids = torch.tensor([s + [tok.pad_id] * (L - len(s))
                                for s in batch], device=dev)
            mask = torch.tensor([[1] * len(s) + [0] * (L - len(s))
                                 for s in batch], device=dev)
            logits = model(ids[:, :-1], mask[:, :-1])
            labels = ids[:, 1:].clone()
            labels[mask[:, 1:] == 0] = -100
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched.last_epoch < steps_total - 1:
                sched.step()
            opt.zero_grad()
            step += 1
            if step % 200 == 0:
                print(f"  step {step}/{steps_total} loss "
                      f"{float(loss.detach()):.3f} "
                      f"({step/(time.time()-t0):.1f} it/s)", flush=True)

    torch.save(model.state_dict(), OUT)
    print(f"[swap] saved {OUT} after {step} steps "
          f"({time.time()-t0:.0f}s)", flush=True)

    from llmopt.lab.gate import gate_eval
    from llmopt.lab.hash import git_sha
    model.eval()
    solves, valid = gate_eval(model, tok, dev)
    tot = sum(solves.values())
    row = {"arm": "swap", "seed": 2, "steps": step, "solves": solves,
           "total": tot, "valid_pct": round(valid, 2), "device": dev,
           "admit_log": [], "code_commit": git_sha(short=True)}
    with RECEIPTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[swap] GATE {solves} = {tot}/120 @ {valid:.2f}%", flush=True)


if __name__ == "__main__":
    main()
