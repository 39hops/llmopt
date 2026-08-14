"""CURRICULUM-1 instrument run (pre-reg RESULTS 2026-08-13): the
phase19m recipe with ONE variable changed — the ORDER rows enter the
stream. Two treatment arms over the same D2-excised gen4 diet, same
BIRTH_SEED=2 init, same stock OneCycle length and shape (steps_total
computed exactly as the trainer's nopack branch computes it):

  ARM=cap    capability-ordered fixed curriculum: each epoch streams
             levels in the measured easy-to-hard ladder
             L1,L2,L3,L5,L7,L6,L4,L8; batches are BS-slices of each
             level's length-sorted sequences (partials allowed),
             batch order shuffled WITHIN level (string-seeded);
             stream truncated to the stock per-epoch step count.
  ARM=level  level-ascending plateau-gated admission: pool starts
             {L1,L2}; every 100 steps, if mean loss of the last 300
             steps improves < 2% relative vs the prior 300-step
             window, the next level (3,4,...,8) is admitted.
             Stream = shuffled batches drawn from the admitted pool,
             re-shuffled each cycle; runs exactly steps_total steps.
             Admission steps logged (they are the finding's shape).
  ARM=off    NO-OP PRECONDITION: build this driver's stock-mode
             stream and assert its epoch-1 batch tuples equal the
             REAL trainer path's, captured by running TM.main with a
             shuffle recorder and aborting on the first optimizer
             step. Trains nothing. Must pass before any arm fires.

The training loop body replicates scripts/train_mathnative.py's
per-batch lines exactly (CE ignore_index, clip 1.0, OneCycle guard);
Mac/mps, fp32 (the trainer's amp is cuda-only). Final weights saved
to checkpoints/gallery19m_curric_<arm>_s2.pt and gated (standard
120); receipt appended to logs/curric1/arms.jsonl.

Usage: ARM=off  .venv/bin/python scratch/birth19m_curric.py
       ARM=cap  ... / ARM=level ...
"""
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

ARM = os.environ["ARM"]
assert ARM in ("off", "cap", "level"), ARM
SEED = "2"
os.environ["BIRTH_SEED"] = SEED
OUT = Path(f"checkpoints/gallery19m_curric_{ARM}_s2.pt")
RECEIPTS = Path("logs/curric1/arms.jsonl")
if ARM != "off" and OUT.exists():
    raise SystemExit(f"REFUSING: {OUT} exists")

import torch  # noqa: E402

import train_mathnative as TM  # noqa: E402
from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402

BS = TM.BS            # 32
EPOCHS = 3
LADDER = [1, 2, 3, 5, 7, 6, 4, 8]        # measured easy-to-hard
ADMIT_ORDER = [3, 4, 5, 6, 7, 8]          # arm B admission sequence
PLATEAU_CHECK, PLATEAU_WIN, PLATEAU_REL = 100, 300, 0.02


_STOCK_LOAD = TM.load_rows      # bound before any patching


def load_excised_rows():
    rows = _STOCK_LOAD(True, True, True, True, False, False, None)
    band = set(gate_band_exprs())
    kept = [r for r in rows
            if norm(str(r.get("cur", ""))) not in band
            and norm(str(r.get("nxt", ""))) not in band]
    print(f"[curric] D2 excision: {len(rows)} -> {len(kept)} rows",
          flush=True)
    return kept


def encode_with_levels(rows, tok):
    """The trainer's text/encode/filter path with the level tag
    carried alongside; enc order = stable length sort, as stock."""
    pairs = []
    for r in rows:
        t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
        try:
            ids = tok.encode(t) + [tok.eos_id]
        except ValueError:
            continue
        if len(ids) <= int(os.environ.get("SEQ_CAP", "512")):
            pairs.append((ids, int(r["level"])))
    pairs.sort(key=lambda p: len(p[0]))   # stable, = enc.sort(key=len)
    enc = [p[0] for p in pairs]
    levels = [p[1] for p in pairs]
    return enc, levels


def stock_epoch_stream(n_enc, ep):
    starts = [(i, i + BS) for i in range(0, n_enc - BS, BS)]
    idx = list(starts)
    random.Random(ep).shuffle(idx)
    return idx


def level_batches(levels, shuffle_key):
    """Per-level BS-slices of the (already length-sorted) enc,
    partials allowed; batch order shuffled within level."""
    by_level = {}
    for i, l in enumerate(levels):
        by_level.setdefault(l, []).append(i)
    out = {}
    for l, idxs in by_level.items():
        bs = [idxs[i:i + BS] for i in range(0, len(idxs), BS)]
        random.Random(f"{shuffle_key}-L{l}").shuffle(bs)
        out[l] = bs
    return out


def assert_noop(enc):
    """Capture the REAL trainer epoch-1 stream (shuffle recorder +
    abort on first optimizer step) and assert this driver's
    stock-mode stream reproduces it tuple-for-tuple."""
    recorded = []

    class RecordingRandom(random.Random):
        def shuffle(self, x):
            super().shuffle(x)
            if not recorded and x and isinstance(x[0], tuple):
                recorded.extend(x)

    class _Done(Exception):
        pass

    orig_random, orig_step = TM.random.Random, torch.optim.AdamW.step
    orig_load = TM.load_rows
    TM.load_rows = lambda *a, **kw: load_excised_rows()
    TM.random.Random = RecordingRandom

    def abort_step(self, *a, **kw):
        raise _Done

    torch.optim.AdamW.step = abort_step
    try:
        TM.main(v2=False, d=384, layers=8, ffn=1536, heads=6,
                out="checkpoints/curric_noop_probe.pt", v21=False,
                fast=True, nopack=True, v22=True, gen4=True,
                epochs=EPOCHS)
    except _Done:
        pass
    finally:
        TM.random.Random = orig_random
        torch.optim.AdamW.step = orig_step
        TM.load_rows = orig_load
        Path("checkpoints/curric_noop_probe.pt.ep").unlink(
            missing_ok=True)
    mine = stock_epoch_stream(len(enc), 0)
    assert recorded, "recorder captured nothing"
    n = min(100, len(mine), len(recorded))
    assert mine[:n] == recorded[:n], "NO-OP FAILED: stream mismatch"
    assert len(mine) == len(recorded), \
        f"NO-OP FAILED: {len(mine)} vs {len(recorded)} batches"
    print(f"[curric] NO-OP PASS: {n} tuples + epoch length "
          f"{len(mine)} match the trainer path", flush=True)


def main():
    tok = TM.MathTokenizer()
    rows = load_excised_rows()
    enc, levels = encode_with_levels(rows, tok)
    print(f"[curric] {len(enc)} sequences, vocab {len(tok.vocab)}",
          flush=True)
    if ARM == "off":
        assert_noop(enc)
        return

    dev = ("mps" if torch.backends.mps.is_available() else
           "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(SEED))
    model = TM.build_model(len(tok.vocab), d=384, layers=8,
                           heads=6, ffn=1536).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                            weight_decay=0.01)
    steps_per_epoch = len(stock_epoch_stream(len(enc), 0))
    steps_total = EPOCHS * (len(enc) // BS)   # trainer nopack branch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4, total_steps=steps_total, pct_start=0.03)
    print(f"[curric] arm {ARM}: steps_total {steps_total} "
          f"({steps_per_epoch}/epoch)", flush=True)

    losses, admitted, admit_log = [], set(), []
    pending = []
    if ARM == "level":
        admitted = {1, 2}
        pending = list(ADMIT_ORDER)

    def train_batch(b):
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
        losses.append(float(loss.detach()))

    step = 0
    t0 = time.time()
    if ARM == "cap":
        for ep in range(EPOCHS):
            lb = level_batches(levels, f"curric-cap-{ep}")
            stream = [b for l in LADDER for b in lb.get(l, [])]
            dropped = len(stream) - steps_per_epoch
            stream = stream[:steps_per_epoch]
            print(f"[curric] ep{ep}: {len(stream)} batches "
                  f"({dropped} tail batches dropped to match stock "
                  f"count)", flush=True)
            for b in stream:
                train_batch(b)
                step += 1
                if step % 200 == 0:
                    print(f"  step {step}/{steps_total} loss "
                          f"{sum(losses[-200:])/200:.3f} "
                          f"({step/(time.time()-t0):.1f} it/s)",
                          flush=True)
    else:  # level: plateau-gated admission, cycle until steps_total
        cycle = 0
        while step < steps_total:
            lb = level_batches(levels, f"curric-level-{cycle}")
            pool = [b for l in sorted(admitted) for b in lb[l]]
            random.Random(f"curric-pool-{cycle}").shuffle(pool)
            for b in pool:
                if step >= steps_total:
                    break
                train_batch(b)
                step += 1
                if (pending and step % PLATEAU_CHECK == 0
                        and len(losses) >= 2 * PLATEAU_WIN):
                    last = sum(losses[-PLATEAU_WIN:]) / PLATEAU_WIN
                    prev = sum(losses[-2 * PLATEAU_WIN:-PLATEAU_WIN]) \
                        / PLATEAU_WIN
                    if (prev - last) / prev < PLATEAU_REL:
                        nxt = pending.pop(0)
                        admitted.add(nxt)
                        admit_log.append({"step": step, "level": nxt,
                                          "prev": round(prev, 4),
                                          "last": round(last, 4)})
                        print(f"[curric] ADMIT L{nxt} at step {step} "
                              f"(prev {prev:.4f} last {last:.4f})",
                              flush=True)
                        break   # rebuild pool with the new level
                if step % 200 == 0:
                    print(f"  step {step}/{steps_total} loss "
                          f"{sum(losses[-200:])/200:.3f} admitted "
                          f"{sorted(admitted)}", flush=True)
            cycle += 1

    torch.save(model.state_dict(), OUT)
    print(f"[curric] saved {OUT} after {step} steps "
          f"({time.time()-t0:.0f}s)", flush=True)

    from llmopt.lab.gate import gate_eval
    from llmopt.lab.hash import git_sha
    model.eval()
    solves, valid = gate_eval(model, tok, dev)
    tot = sum(solves.values())
    RECEIPTS.parent.mkdir(parents=True, exist_ok=True)
    row = {"arm": ARM, "seed": int(SEED), "steps": step,
           "solves": solves, "total": tot,
           "valid_pct": round(valid, 2), "device": dev,
           "admit_log": admit_log, "code_commit": git_sha(short=True)}
    with RECEIPTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[curric] arm {ARM} GATE {solves} = {tot}/120 "
          f"@ {valid:.2f}%", flush=True)


if __name__ == "__main__":
    main()
