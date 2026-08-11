"""keff_probe — DIRECT effective-context measurement on a trained
checkpoint (pre-reg RESULTS PRE-REG KEFF-PROBE-1). No training.

Method: sample N sequences from the warm diet (same loader, same
tokenizer, same SEQ_CAP rule as training), and for each truncation
width k, compute the mean next-token loss at positions >= k when the
model sees ONLY the last k tokens of context (a sliding window,
evaluated at the final position of each window). Loss(k) flattens
where the model stops using additional context — the knee is k_eff,
measured, not inferred from the H_k curve.

Positions scored are IDENTICAL across k (positions >= k_max), so
Loss(k) differences are pure context effects, never a position-mix
artifact. Deterministic sampling (string seed doctrine).

Usage:
  python scratch/keff_probe.py <ckpt> <d> <layers> <ffn> <heads> \
      <label> [--ssm] [--n 400] [--device cpu]
Prints one line per k and a summary row; caller books the table.
"""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

KS = (4, 8, 16, 32, 64, 128)
K_MAX = 128
N_DEFAULT = 400


def main():
    import torch

    from llmopt.train.mathnative import MathTokenizer, build_model
    import train_mathnative as TM

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ckpt, d, layers, ffn, heads, label = (
        args[0], int(args[1]), int(args[2]), int(args[3]),
        int(args[4]), args[5])
    ssm = "--ssm" in sys.argv
    n = N_DEFAULT
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    dev = "cpu"
    if "--device" in sys.argv:
        dev = sys.argv[sys.argv.index("--device") + 1]

    tok = MathTokenizer()
    if ssm:
        from ssm_star import build_ssm_model as build
    else:
        build = build_model
    model = build(len(tok.vocab), d=d, layers=layers,
                  heads=heads, ffn=ffn).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location="cpu",
                                     weights_only=True))
    model.eval()

    rows = TM.load_rows(True, True, True, True, True, False, None)
    texts = [f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
             for r in rows]
    enc = []
    for t in texts:
        try:
            ids = tok.encode(t) + [tok.eos_id]
        except ValueError:
            continue
        if K_MAX < len(ids) <= int(os.environ.get("SEQ_CAP", "512")):
            enc.append(ids)  # only rows LONGER than k_max score fairly
    # FIXED sample across models (string seed, label-free): every
    # checkpoint scores the identical row set or curves don't compare
    random.Random("keff-probe-v1-0").shuffle(enc)
    enc = enc[:n]
    print(f"[keff] {label}: {len(enc)} sequences >= {K_MAX + 1} tokens",
          flush=True)

    # score positions p in [K_MAX, len-1] of each sequence, window k
    results = {}
    with torch.no_grad():
        # FULL cell: entire true prefix (original positions) — the
        # anchor for the window-renumbering confound. If FULL ~= k128,
        # renumbering is benign and Loss(k) reads as pure context.
        tot, cnt = 0.0, 0
        for ids in enc:
            ps = list(range(K_MAX, len(ids)))
            t_ids = torch.tensor(ids, device=dev)[None]
            out = model(t_ids)
            logits = out[0] if isinstance(out, tuple) else out
            lp = torch.log_softmax(logits[0, :-1].float(), -1)
            tgt = t_ids[0, 1:]
            pos = torch.tensor([p - 1 for p in ps], device=dev)
            tot += float(-lp[pos].gather(
                1, tgt[pos][:, None]).sum())
            cnt += len(ps)
        results["full"] = tot / cnt
        print(f"[keff] {label} k=full  loss {results['full']:.4f} "
              f"({cnt} positions)", flush=True)
        for k in KS:
            tot, cnt = 0.0, 0
            for ids in enc:
                # POSITION-TRUE truncation: window left-padded so the
                # kept tokens sit at their original absolute indices
                # (RoPE positions preserved; pads carry attn_mask 0 —
                # the renumbered-window variant overstated loss by
                # ~0.36 nats at k=128 in the smoke, so renumbering is
                # NOT benign and this is the honest cell).
                ps = list(range(K_MAX, len(ids)))
                Tmax = max(ps)
                pad = 0  # tok.pad id — masked out, value irrelevant
                wins, mask, tgts = [], [], []
                for p in ps:
                    row = [pad] * Tmax
                    row[p - k:p] = ids[p - k:p]
                    m = [0] * Tmax
                    for j in range(p - k, p):
                        m[j] = 1
                    wins.append(row)
                    mask.append(m)
                    tgts.append(ids[p])
                wins = torch.tensor(wins, device=dev)
                mask_t = torch.tensor(mask, device=dev)
                tgts = torch.tensor(tgts, device=dev)
                out = model(wins, attn_mask=mask_t)
                logits = out[0] if isinstance(out, tuple) else out
                # prediction for position p reads at index p-1
                pos = torch.tensor([p - 1 for p in ps], device=dev)
                lp = torch.log_softmax(
                    logits[torch.arange(len(ps)), pos].float(), -1)
                tot += float(-lp.gather(1, tgts[:, None]).sum())
                cnt += len(ps)
            results[k] = tot / cnt
            print(f"[keff] {label} k={k:4d}  loss {results[k]:.4f} "
                  f"({cnt} positions)", flush=True)
    knee = min(
        (k for i, k in enumerate(KS[:-1])
         if results[k] - results[KS[i + 1]] < 0.005), default=KS[-1])
    print(f"[keff] {label} SUMMARY " +
          json.dumps({"losses": {str(k): round(v, 4)
                                 for k, v in results.items()},
                      "knee_first_flat": knee}), flush=True)


if __name__ == "__main__":
    main()
