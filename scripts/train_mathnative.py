"""Phase 1: pretrain the math-native micro-model on the farmed diet.

Diet: data/micromodel_chains_shard*.jsonl (94.5k engine-minted
pairs) + the purged main corpus. Full-sequence LM loss on the step
format — this text is the model's entire world. Gate (spec): >=1%
step validity at L2-3, temp 0.7, before phase 2 (GRPO from birth).

    .venv/bin/python scripts/train_mathnative.py
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmopt.train.mathnative import MathTokenizer, build_model

CKPT = Path("checkpoints/mathnative_19m.pt")
V2_CKPT = Path("checkpoints/mathnative_19m_v2.pt")
EPOCHS = 3
BS = 32
LR = 3e-4  # from-scratch: standard small-LM lr with warmup+cosine


def load_rows(v2: bool = False, v21: bool = False,
              v22: bool = False, gen4: bool = False,
              l8: bool = False, gen7: bool = False):
    rows = []
    for f in sorted(glob.glob("data/micromodel_chains_shard*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    rows += [json.loads(l) for l in open("data/step_chains.jsonl")]
    if v2:  # curriculum v2: + the algebra substrate (30k rewrites)
        for f in sorted(glob.glob("data/micromodel_algebra_shard*.jsonl")):
            rows += [json.loads(l) for l in open(f)]
    if v21:  # v2.1: + the L4 engine-chain shard (3.2k, chains-only)
        for f in sorted(glob.glob("data/micromodel_calc_l4_shard*.jsonl")):
            rows += [json.loads(l) for l in open(f)]
    if v22:  # v2.2: + the autopsy-aimed shard (in-language L4-L7
        # chains + capped one-ply worked examples, farm_v22.py)
        for f in sorted(glob.glob("data/micromodel_v22_shard*.jsonl")):
            rows += [json.loads(l) for l in open(f)]
    if l8:  # gen-6: the L8 territory shard (calculator-priced,
        # ~85% one-ply worked examples by design — the engine
        # one-plies L8; shard6 = deduped 3080 contribution)
        for f in sorted(glob.glob("data/micromodel_l8_shard*.jsonl")):
            rows += [json.loads(l) for l in open(f)]
    if gen7:  # gen-7: the mass-targeted diet REPLACES everything
        return [json.loads(l) for l in open("data/gen7_diet.jsonl")]
    if gen4:  # generational rebirth: + the level-capped cumulative
        # GRPO-mined sidecar (the lineage's whole verified experience)
        rows += [json.loads(l)
                 for l in open("data/micromodel_gen4_sidecar.jsonl")]
    # identity guard at the diet gate too (defense in depth)
    rows = [r for r in rows
            if r["cur"].replace(" ", "") != r["nxt"].replace(" ", "")]
    return rows


def main(v2: bool = False, d: int = 384, layers: int = 8,
         ffn: int = 1536, out: str | None = None,
         heads: int = 6, v21: bool = False, fast: bool = False,
         budget: int = 24_576, lr: float = LR,
         fp32: bool = False, nopack: bool = False,
         v22: bool = False, gen4: bool = False,
         epochs: int = 3, l8: bool = False,
         gen7: bool = False) -> None:
    import torch
    global CKPT, EPOCHS
    EPOCHS = epochs
    if v2:
        CKPT = V2_CKPT
    if out:
        CKPT = Path(out)
    tok = MathTokenizer()
    rows = load_rows(v2 or v21 or v22, v21 or v22, v22, gen4, l8, gen7)
    charset = set()
    texts = []
    for r in rows:
        t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
        texts.append(t)
        charset |= set(t)
    uncovered = {c for c in charset
                 if not tok.encode(c) and c not in (" ",)}
    print(f"{len(rows)} rows; charset {len(charset)}; "
          f"uncovered chars: {sorted(uncovered)[:10]}", flush=True)

    enc = []
    for t in texts:
        ids = tok.encode(t) + [tok.eos_id]
        if len(ids) <= 512:
            enc.append(ids)
    enc.sort(key=len)
    print(f"{len(enc)} sequences, vocab {len(tok.vocab)}", flush=True)

    dev = ("mps" if torch.backends.mps.is_available() else
           "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(os.environ.get("BIRTH_SEED", "0")))
    model = build_model(len(tok.vocab), d=d, layers=layers,
                        heads=heads, ffn=ffn).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.1f}M params on {dev}"
          f"{' [fast: bf16 + token-budget]' if fast else ''}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=0.01)

    # resume BEFORE the schedule so OneCycle spans only the epochs
    # actually run (full-horizon sched on resume ends at peak LR)
    marker = Path(str(CKPT) + ".ep")
    start_ep = 0
    if marker.exists() and CKPT.exists():
        start_ep = int(marker.read_text()) + 1
        model.load_state_dict(torch.load(CKPT, map_location="cpu"))
        model.to(dev)
        print(f"resuming at epoch {start_ep}", flush=True)

    if fast and not nopack:
        # token-budget packing, PROPER: the first cut packed the
        # length-SORTED list — length-homogeneous batches cost ~10
        # unseen-validity points (parity 2x2, 2026-07-16). Pack a
        # SHUFFLED order instead: iid mixed-length batches, padded
        # cost (n * batch-max-len) counted honestly against the
        # budget. Budget must be sized to the model (24.5k thrashed
        # the 113M on 10GB; 12k ran 100x faster).
        def pack_epoch(perm: list[int]) -> list[list[int]]:
            batches, cur, mx = [], [], 0
            for j in perm:
                m = max(mx, len(enc[j]))
                if cur and (len(cur) + 1) * m > budget:
                    batches.append(cur)
                    cur, mx = [j], len(enc[j])
                else:
                    cur.append(j)
                    mx = m
            if cur:
                batches.append(cur)
            return batches
        # shuffled epochs pack to slightly different counts; 5%
        # headroom keeps OneCycle from stepping past total_steps
        steps_total = int(
            (EPOCHS - start_ep) *
            len(pack_epoch(list(range(len(enc))))) * 1.05)
        starts = None  # built per-epoch from a shuffled permutation
    else:
        starts = [(i, i + BS) for i in range(0, len(enc) - BS, BS)]
        steps_total = (EPOCHS - start_ep) * (len(enc) // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps_total, pct_start=0.03)
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if fast and dev == "cuda" and not fp32 else None)

    import contextlib
    for ep in range(start_ep, EPOCHS):
        if starts is None:  # proper packing: shuffle THEN pack
            perm = list(range(len(enc)))
            random.Random(ep).shuffle(perm)
            idx = pack_epoch(perm)
        else:
            idx = list(starts)
            random.Random(ep).shuffle(idx)   # per-epoch order shuffle
        tot = steps = 0
        t0 = time.time()
        for b in idx:
            batch = enc[b[0]:b[1]] if starts is not None else \
                [enc[j] for j in b]
            L = max(len(s) for s in batch)
            ids = torch.tensor([s + [tok.pad_id] * (L - len(s))
                                for s in batch], device=dev)
            mask = torch.tensor([[1] * len(s) + [0] * (L - len(s))
                                 for s in batch], device=dev)
            with (amp or contextlib.nullcontext()):
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
                sched.step()  # shuffled packs vary per epoch; never
                # step OneCycle past its declared total
            opt.zero_grad()
            tot += float(loss.detach())
            steps += 1
            if steps % 200 == 0:
                r = steps / (time.time() - t0)
                print(f"  ep{ep} {steps}/{len(idx)} loss "
                      f"{tot/steps:.3f} ({r:.1f} it/s)", flush=True)
        print(f"epoch {ep}: loss {tot/max(steps,1):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        torch.save(model.state_dict(), CKPT)
        marker.write_text(str(ep))
    marker.unlink(missing_ok=True)
    print(f"saved {CKPT}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true",
                    help="curriculum v2 diet (+algebra shard), "
                         "separate checkpoint")
    ap.add_argument("--v21", action="store_true",
                    help="v2.1 diet (v2 + L4 calc chains); use with "
                         "--out")
    ap.add_argument("--fast", action="store_true",
                    help="bf16 autocast (cuda) + token-budget "
                         "batching; parity-gated before cross-run "
                         "comparisons trust it")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=1536)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--out", default=None,
                    help="checkpoint path override (capacity runs)")
    ap.add_argument("--budget", type=int, default=24_576,
                    help="token budget per packed batch (--fast); "
                         "shrink when the model leaves no VRAM "
                         "headroom")
    ap.add_argument("--lr", type=float, default=LR,
                    help="max lr (OneCycle); token-budget batching "
                         "cuts optimizer steps ~6x vs BS=32 — scale "
                         "lr when packing (sqrt rule ~2.5x @ 12k)")
    ap.add_argument("--fp32", action="store_true",
                    help="with --fast: keep token-budget packing but "
                         "disable bf16 autocast (parity-fail lever "
                         "isolation)")
    ap.add_argument("--v22", action="store_true",
                    help="v2.2 diet (v2.1 + autopsy-aimed shard); "
                         "use with --out")
    ap.add_argument("--gen4", action="store_true",
                    help="with --v22: + level-capped cumulative "
                         "grpo-mined sidecar (generational rebirth)")
    ap.add_argument("--nopack", action="store_true",
                    help="with --fast: bf16 autocast only, standard "
                         "BS=32 batching (packing failed parity: "
                         "45.65/46.95 vs 56.67 standard)")
    ap.add_argument("--l8", action="store_true",
                    help="gen-6: include the L8 territory shard")
    ap.add_argument("--gen7", action="store_true",
                    help="mass-targeted diet (data/gen7_diet.jsonl)")
    ap.add_argument("--epochs", type=int, default=3,
                    help="total epochs; with an existing .ep marker "
                         "the run resumes and OneCycle spans only "
                         "the remaining epochs")
    a = ap.parse_args()
    main(a.v2, a.d, a.layers, a.ffn, a.out, a.heads, a.v21,
         a.fast, a.budget, a.lr, a.fp32, a.nopack, a.v22, a.gen4,
         a.epochs, a.l8, a.gen7)
