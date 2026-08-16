"""BASICS-DIET-1 birth driver (pre-reg RESULTS L30355, amended
L30429): stock diet + 6,000 sympy calculus atoms, with or without
the 2,545-row arithmetic shard.

Sibling of the FROZEN scratch/birth19m_atoms_rule.py — same
mechanics (C.encode_with_levels path, stock-matched epoch stream,
OneCycle, steps pinned to the stock count), two differences that
are the point:
  * the arm carries TWO extra families, calculus atoms (always) and
    arithmetic rows (ARITH arm only), so the flag is per-source;
  * provenance fields are DERIVED from the shards actually opened,
    never a literal (the RULE-ABLATE false-emitter class).

  SEED=3 ARM=control|arith .venv/bin/python -u scratch/birth19m_arith.py
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

SEED = int(os.environ["SEED"])
ARM_NAME = os.environ["ARM_NAME"]          # control | arith
assert ARM_NAME in ("control", "arith"), ARM_NAME
SMOKE = os.environ.get("SMOKE", "0") == "1"

ATOM_SHARD = Path("data/micromodel_atoms_shard0.jsonl")
ARITH_SHARD = Path("data/micromodel_arith_shard0.jsonl")

os.environ["ARM"] = "off"       # frozen module import side-effects only
os.environ["BIRTH_SEED"] = str(SEED)
OUT = Path(f"checkpoints/gallery19m_basics{ARM_NAME}_s{SEED}"
           + ("_smoke" if SMOKE else "") + ".pt")
if OUT.exists():
    raise SystemExit(f"REFUSING: {OUT} exists")

import torch  # noqa: E402

import birth19m_curric as C  # noqa: E402  (frozen, import-only)
import train_mathnative as TM  # noqa: E402
from llmopt.lab.hash import git_sha  # noqa: E402
from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402

EPOCHS, BS = C.EPOCHS, C.BS
RECEIPTS = Path("logs/basicsdiet1/"
                + ("smoke.jsonl" if SMOKE else "arms.jsonl"))
STEPS_TOTAL_PIN = 15_420
EXTRA_SOURCES = ("atom-oneply", "arith-oneply")


def encode_flagged(rows, tok):
    """C.encode_with_levels verbatim, carrying the source tag."""
    triples = []
    for r in rows:
        t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
        try:
            ids = tok.encode(t) + [tok.eos_id]
        except ValueError:
            continue
        if len(ids) <= int(os.environ.get("SEQ_CAP", "512")):
            triples.append((ids, int(r["level"]), r.get("source")))
    triples.sort(key=lambda p: len(p[0]))   # stable, = enc.sort(key=len)
    return [p[0] for p in triples], [p[2] for p in triples]


def load_shard(path, band):
    rows = [json.loads(line) for line in path.open()]
    keep = [r for r in rows
            if norm(str(r["cur"])) not in band
            and norm(str(r["nxt"])) not in band]
    print(f"[basics] {path.name}: {len(rows)} farmed, {len(keep)} "
          f"after in-driver band excision", flush=True)
    return keep


def main():
    tok = TM.MathTokenizer()
    stock_rows = C.load_excised_rows()
    band = set(gate_band_exprs())

    atoms = load_shard(ATOM_SHARD, band)
    arith = load_shard(ARITH_SHARD, band) if ARM_NAME == "arith" else []
    extra = atoms + arith

    enc_stock, _ = C.encode_with_levels(stock_rows, tok)
    enc, src = encode_flagged(stock_rows + extra, tok)
    n_by_src = {s: src.count(s) for s in EXTRA_SOURCES}
    print(f"[basics] arm={ARM_NAME} seed={SEED}: stock {len(enc_stock)}"
          f" seq, train {len(enc)} seq, extra {n_by_src}", flush=True)

    C.assert_noop(enc_stock)    # precondition, fresh, in-process

    steps_per_epoch = len(C.stock_epoch_stream(len(enc_stock), 0))
    steps_total = EPOCHS * (len(enc_stock) // BS)
    assert steps_total == STEPS_TOTAL_PIN, steps_total
    epochs = 1 if SMOKE else EPOCHS

    dev = ("mps" if torch.backends.mps.is_available() else
           "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    model = TM.build_model(len(tok.vocab), d=384, layers=8,
                           heads=6, ffn=1536).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4, total_steps=steps_total, pct_start=0.03)
    print(f"[basics] steps_total {steps_total} "
          f"({steps_per_epoch}/epoch)", flush=True)

    step = 0
    t0 = time.time()
    exposure = []
    for ep in range(epochs):
        stream = C.stock_epoch_stream(len(enc), ep)
        dropped = len(stream) - steps_per_epoch
        stream = stream[:steps_per_epoch]
        if SMOKE:
            stream = stream[:3]
        seen = {s: 0 for s in EXTRA_SOURCES}
        for a, b in stream:
            for j in range(a, b):
                if src[j] in seen:
                    seen[src[j]] += 1
        exposure.append(seen)
        print(f"[basics] ep{ep}: {len(stream)} batches ({dropped} "
              f"dropped to match stock count), extra rows in stream "
              f"{seen}", flush=True)
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
                      f"({step/(time.time()-t0):.1f} it/s)", flush=True)

    torch.save(model.state_dict(), OUT)
    print(f"[basics] saved {OUT} after {step} steps "
          f"({time.time()-t0:.0f}s)", flush=True)

    from llmopt.lab.gate import gate_eval
    model.eval()
    solves, valid = gate_eval(model, tok, dev)
    tot = sum(solves.values())
    RECEIPTS.parent.mkdir(parents=True, exist_ok=True)
    row = {"arm": ARM_NAME, "seed": SEED, "smoke": SMOKE, "steps": step,
           "solves": solves, "total": tot, "valid_pct": round(valid, 2),
           "device": dev, "dtype": str(next(model.parameters()).dtype),
           "shards": {p.name: n for p, n in
                      ((ATOM_SHARD, len(atoms)), (ARITH_SHARD, len(arith)))
                      if n},
           "extra_rows_encoded": n_by_src,
           "extra_rows_per_epoch": exposure,
           "ckpt": str(OUT), "code_commit": git_sha(short=True),
           "wall_s": round(time.time() - t0, 1)}
    with RECEIPTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[basics] GATE arm={ARM_NAME} s{SEED} {solves} = "
          f"{tot}/120 @ {valid:.2f}%", flush=True)


if __name__ == "__main__":
    main()
