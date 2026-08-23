"""MATH-CYBER-1 theta_0 one-shot birth (PRE-REG
MATH-CYBER-1-THETA0-BIRTH-0; prereg commit 6d013acf, BEFORE any
weight existed). Grammar-closed tokenizer (ATOMS ids 0..39 + 256
UTF-8 byte-fallback ids = vocab 296), base diet only
(micromodel_chains_shard*.jsonl + step_chains.jsonl), SEQ cap 512
(drops counted), build_model(296, ctx=4096), fp32 mps,
BIRTH_SEED=9001, AdamW lr 3e-4 wd 0.01, OneCycle pct_start 0.03,
clip 1.0, BS=32 nopack (length-sorted enc, per-epoch
random.Random(ep) shuffle of batch starts — the historic stream),
EPOCHS=3, final-epoch weights = theta_0, NO selection of any kind.

SMOKE=1: first 2000 rows, 1 epoch, smoke_ paths (path-isolated).

Receipts: checkpoints/mathnative_19m_mw1_theta0.pt (+ .json birth
receipt with diet manifest shas, drop count, epoch losses,
checkpoint sha256). Refuse-if-exists on ckpt AND receipt.

    .venv/bin/python scratch/mathworld1_birth.py              (Mac)
"""
import glob
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.train.mathnative import ATOMS, build_model  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
PRE = "smoke_" if SMOKE else ""
CKPT = Path(f"checkpoints/{PRE}mathnative_19m_mw1_theta0.pt")
RECEIPT = Path(f"checkpoints/{PRE}mathnative_19m_mw1_theta0.json")
BIRTH_SEED = 9001
SEQ_CAP = 512
BS = 32
LR = 3e-4
EPOCHS = 1 if SMOKE else 3


class GCTok:
    """Grammar-closed: ATOMS (standing order, ids 0..39) + 256
    UTF-8 byte-fallback ids (40+b). pad=ATOMS[0], eos=ATOMS[1]."""

    def __init__(self):
        self.atoms = list(ATOMS)
        self.n_base = len(self.atoms)
        self.vocab_size = self.n_base + 256
        self.id = {t: i for i, t in enumerate(self.atoms)}
        self.pad_id = self.id["<pad>"]
        self.eos_id = self.id["<eos>"]
        self._by_len = sorted(
            (t for t in self.atoms if t not in ("<pad>", "<eos>")),
            key=len, reverse=True)

    def encode(self, s: str) -> list[int]:
        out, i = [], 0
        while i < len(s):
            for t in self._by_len:
                if s.startswith(t, i):
                    out.append(self.id[t])
                    i += len(t)
                    break
            else:
                out.extend(self.n_base + b
                           for b in s[i].encode())
                i += 1
        return out


def main():
    for p in (CKPT, RECEIPT):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    START = start_provenance(
        ["scratch/mathworld1_birth.py",
         "llmopt/train/mathnative.py"])
    tok = GCTok()
    files = sorted(glob.glob("data/micromodel_chains_shard*.jsonl"))
    files.append("data/step_chains.jsonl")
    manifest = {}
    rows = []
    for f in files:
        data = Path(f).read_bytes()
        manifest[f] = {"sha256": hashlib.sha256(data).hexdigest(),
                       "bytes": len(data)}
        rows += [json.loads(line)
                 for line in data.decode().splitlines()]
    if SMOKE:
        rows = rows[:2000]
    enc, dropped = [], 0
    for r in rows:
        ids = tok.encode(f"Current: {r['cur']}\nHints: none\n"
                         f"Step: {r['nxt']}\n") + [tok.eos_id]
        if len(ids) <= SEQ_CAP:
            enc.append(ids)
        else:
            dropped += 1
    enc.sort(key=len)
    print(f"[birth] {len(rows)} rows -> {len(enc)} sequences "
          f"({dropped} dropped > {SEQ_CAP}); vocab "
          f"{tok.vocab_size}", flush=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(BIRTH_SEED)
    model = build_model(tok.vocab_size, ctx=4096).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[birth] {n_params/1e6:.1f}M params on {dev}",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=0.01)
    starts = [(i, i + BS) for i in range(0, len(enc) - BS, BS)]
    steps_total = EPOCHS * len(starts)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=steps_total, pct_start=0.03)
    ep_losses = []
    t_birth = time.time()
    for ep in range(EPOCHS):
        idx = list(starts)
        random.Random(ep).shuffle(idx)
        tot = steps = 0
        t0 = time.time()
        for b in idx:
            batch = enc[b[0]:b[1]]
            L = max(len(s) for s in batch)
            ids = torch.tensor([s + [tok.pad_id] * (L - len(s))
                                for s in batch], device=dev)
            mask = torch.tensor(
                [[1] * len(s) + [0] * (L - len(s))
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
            tot += float(loss.detach())
            steps += 1
            if steps % 200 == 0:
                print(f"  ep{ep} {steps}/{len(idx)} loss "
                      f"{tot/steps:.3f} "
                      f"({steps/(time.time()-t0):.1f} it/s)",
                      flush=True)
        ep_losses.append(round(tot / max(steps, 1), 4))
        print(f"[birth] epoch {ep}: loss {ep_losses[-1]} "
              f"({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), CKPT)
    ck_sha = hashlib.sha256(CKPT.read_bytes()).hexdigest()
    RECEIPT.write_text(json.dumps({
        "smoke": SMOKE, "birth_seed": BIRTH_SEED,
        "vocab": tok.vocab_size, "seq_cap": SEQ_CAP,
        "bs": BS, "lr": LR, "epochs": EPOCHS,
        "diet_manifest": manifest, "rows": len(rows),
        "sequences": len(enc), "dropped_over_cap": dropped,
        "epoch_losses": ep_losses, "n_params": n_params,
        "device": dev, "checkpoint": str(CKPT),
        "checkpoint_sha256": ck_sha,
        "wall_s": round(time.time() - t_birth, 1),
        "start": START,
        "completion_commit": completion_commit()}, indent=1))
    print(f"[birth] saved {CKPT} sha256 {ck_sha[:16]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
