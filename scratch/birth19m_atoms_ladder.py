"""ATOM-DIET-LADDER-1 instrument (pre-reg RESULTS 2026-08-14):
paired stock/atoms births at matched 15,420-step horizon. Sibling of
the frozen birth19m_atoms.py (results-cited, not edited) with SEED
and ARM lifted to envs; the recipe, dose mechanics, no-op
precondition, and gate are otherwise verbatim. ARM=stock trains the
plain stock enc (the true matched-horizon control the original
comparison lacked); ARM=atoms interleaves the frozen shard
data/micromodel_atoms_shard0.jsonl via stock_epoch_stream over the
augmented enc, truncated per epoch to the stock batch count.
Checkpoint checkpoints/gallery19m_{ARM}_s{SEED}.pt; receipt appended
to logs/atomladder1/arms.jsonl.

Usage: ARM=atoms SEED=3 .venv/bin/python scratch/birth19m_atoms_ladder.py
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

LADDER_ARM = os.environ.get("ARM", "")
SEED = int(os.environ["SEED"])
assert LADDER_ARM in ("stock", "atoms"), LADDER_ARM

os.environ["ARM"] = "off"       # frozen module import side-effects only
os.environ["BIRTH_SEED"] = str(SEED)
OUT = Path(f"checkpoints/gallery19m_{LADDER_ARM}_s{SEED}.pt")
if OUT.exists():
    raise SystemExit(f"REFUSING: {OUT} exists")

import torch  # noqa: E402

import birth19m_curric as C  # noqa: E402  (frozen, import-only)
import train_mathnative as TM  # noqa: E402
from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402

EPOCHS, BS = C.EPOCHS, C.BS
SHARD = Path("data/micromodel_atoms_shard0.jsonl")
RECEIPTS = Path("logs/atomladder1/arms.jsonl")
STEPS_TOTAL_PIN = 15_420


def encode_flagged(rows, tok):
    """The trainer's text/encode/filter path (C.encode_with_levels
    verbatim) with an is_atom flag carried alongside."""
    triples = []
    for r in rows:
        t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
        try:
            ids = tok.encode(t) + [tok.eos_id]
        except ValueError:
            continue
        if len(ids) <= int(os.environ.get("SEQ_CAP", "512")):
            triples.append((ids, int(r["level"]),
                            r.get("source") == "atom-oneply"))
    triples.sort(key=lambda p: len(p[0]))   # stable, = enc.sort(key=len)
    enc = [p[0] for p in triples]
    is_atom = [p[2] for p in triples]
    return enc, is_atom


def main():
    tok = TM.MathTokenizer()
    stock_rows = C.load_excised_rows()

    enc_stock, _ = C.encode_with_levels(stock_rows, tok)
    if LADDER_ARM == "atoms":
        atoms = [json.loads(l) for l in SHARD.open()]
        band = set(gate_band_exprs())
        atoms = [r for r in atoms
                 if norm(str(r["cur"])) not in band
                 and norm(str(r["nxt"])) not in band]
        print(f"[ladder] shard {SHARD}: {len(atoms)} rows after "
              f"in-driver excision", flush=True)
        enc, is_atom = encode_flagged(stock_rows + atoms, tok)
    else:
        atoms = []
        enc, is_atom = enc_stock, [False] * len(enc_stock)
    print(f"[ladder] arm={LADDER_ARM} seed={SEED}: stock "
          f"{len(enc_stock)} seq, train {len(enc)} seq "
          f"({sum(is_atom)} atoms)", flush=True)

    C.assert_noop(enc_stock)    # precondition, fresh, in-process

    steps_per_epoch = len(C.stock_epoch_stream(len(enc_stock), 0))
    steps_total = EPOCHS * (len(enc_stock) // BS)
    assert steps_total == STEPS_TOTAL_PIN, steps_total

    dev = ("mps" if torch.backends.mps.is_available() else
           "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    model = TM.build_model(len(tok.vocab), d=384, layers=8,
                           heads=6, ffn=1536).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4, total_steps=steps_total, pct_start=0.03)
    print(f"[ladder] steps_total {steps_total} "
          f"({steps_per_epoch}/epoch)", flush=True)

    step = 0
    t0 = time.time()
    atom_exposure = []
    for ep in range(EPOCHS):
        stream = C.stock_epoch_stream(len(enc), ep)
        dropped = len(stream) - steps_per_epoch
        stream = stream[:steps_per_epoch]
        n_atom_rows = sum(1 for a, b in stream
                          for j in range(a, b) if is_atom[j])
        atom_exposure.append(n_atom_rows)
        print(f"[ladder] ep{ep}: {len(stream)} batches ({dropped} "
              f"dropped to match stock count), {n_atom_rows} atom "
              f"rows in stream", flush=True)
        for a, b in stream:
            batch = enc[a:b]
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
                      f"({step/(time.time()-t0):.1f} it/s)",
                      flush=True)

    torch.save(model.state_dict(), OUT)
    print(f"[ladder] saved {OUT} after {step} steps "
          f"({time.time()-t0:.0f}s)", flush=True)

    from llmopt.lab.gate import gate_eval
    from llmopt.lab.hash import git_sha
    model.eval()
    solves, valid = gate_eval(model, tok, dev)
    tot = sum(solves.values())
    RECEIPTS.parent.mkdir(parents=True, exist_ok=True)
    row = {"arm": LADDER_ARM, "seed": SEED, "steps": step,
           "solves": solves, "total": tot,
           "valid_pct": round(valid, 2), "device": dev,
           "n_atoms_shard": len(atoms),
           "atom_rows_per_epoch": atom_exposure,
           "code_commit": git_sha(short=True)}
    with RECEIPTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[ladder] GATE arm={LADDER_ARM} s{SEED} {solves} = "
          f"{tot}/120 @ {valid:.2f}%", flush=True)


if __name__ == "__main__":
    main()
