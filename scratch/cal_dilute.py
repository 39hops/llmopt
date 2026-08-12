"""CAL-DK-2 (pre-reg 2026-07-30): diet dilution. Train the dense
d64h8 recipe with fraction DILUTE of rows' targets swapped among
the corrupted subset (fluent, determined-looking, WRONG rows).
Usage: DILUTE=0.1 SEED=1 python scratch/cal_dilute.py
Then: CKPT=checkpoints/cal_dilute_10_s1.pt python scratch/cal_dk_probe.py
"""
from llmopt.common.device import pick_device
import os
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from train_mathnative import load_rows  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

DILUTE = float(os.environ["DILUTE"])
SEED = int(os.environ.get("SEED", "1"))
D, LAYERS, HEADS, FFN = 64, 8, 8, 256
BS, EPOCHS, LR = 8, 3, 1.5e-3
OUT = f"checkpoints/cal_dilute_{int(DILUTE * 100)}_s{SEED}.pt"


def main():
    dev = pick_device()
    torch.manual_seed(SEED)
    tok = MathTokenizer()
    model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN).to(dev)
    rows = load_rows(gen4=True)
    rows = [dict(r) for r in rows
            if r["cur"].replace(" ", "") != r["nxt"].replace(" ", "")]
    # corruption: swap nxt among a random DILUTE-fraction subset
    rng = random.Random(f"dilute-{DILUTE}-{SEED}")
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    bad = idx[:int(len(rows) * DILUTE)]
    targets = [rows[i]["nxt"] for i in bad]
    rng.shuffle(targets)
    for i, t in zip(bad, targets):
        rows[i]["nxt"] = t
    print(f"[dilute] f={DILUTE} corrupted {len(bad)}/{len(rows)} "
          f"rows seed {SEED} dev {dev}", flush=True)
    enc = []
    for r in rows:
        try:
            ids = tok.encode(f"Current: {r['cur']}\nHints: none\n"
                             f"Step: {r['nxt']}\n") + [tok.eos_id]
        except ValueError:
            continue
        if len(ids) <= 512:
            enc.append(ids)
    enc.sort(key=len)
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=0.01)
    order = list(range(0, len(enc) - BS + 1, BS))
    step = 0
    for ep in range(EPOCHS):
        random.Random(ep).shuffle(order)
        for off in order:
            batch = enc[off:off + BS]
            L = max(len(q) for q in batch)
            x = torch.tensor([q + [tok.pad_id] * (L - len(q))
                              for q in batch], device=dev)
            lg = model(x)[:, :-1]
            loss = F.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1),
                ignore_index=tok.pad_id)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 2000 == 0:
                print(f"  ep{ep} step {step} loss {float(loss):.3f}",
                      flush=True)
    torch.save({"sd": model.state_dict()}, OUT)
    model.eval()
    solves, valid = G.gate_eval(model, tok, dev)
    print(f"[gate] DILUTE={DILUTE} solves {solves}/120 "
          f"valid {valid}", flush=True)


if __name__ == "__main__":
    main()
