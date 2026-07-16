"""Phase-1 gate: does the from-scratch 19M reach 1% step validity
at L2-3 (the 0.5B's historical starting point)?

Sampler: temp-0.7 multinomial over the 45-token vocab, B parallel
streams, stop at newline. Verification: the fast wave oracle,
identity/cycle-guarded exactly like every other honest metric.

    .venv/bin/python scripts/eval_mathnative.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmopt.train.mathnative import MathTokenizer, build_model

N_PER, WAVES, B = 24, 8, 8
SEED0 = 9_600_000  # fresh band


def sample_wave(model, tok, prompt_ids, seeds, dev, max_new=120):
    import torch
    Bn = len(seeds)
    ids = torch.tensor([prompt_ids] * Bn, device=dev)
    gens = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
    out = [[] for _ in range(Bn)]
    done = [False] * Bn
    nl = tok.id["\n"]
    for _ in range(max_new):
        logits = model(ids)[:, -1].float().cpu() / 0.7
        probs = torch.softmax(logits, -1)
        nxts = []
        for b in range(Bn):
            if done[b]:
                nxts.append(tok.pad_id)
                continue
            nxt = int(torch.multinomial(probs[b], 1, generator=gens[b]))
            if nxt in (nl, tok.eos_id, tok.pad_id):
                done[b] = True
            else:
                out[b].append(nxt)
            nxts.append(nxt)
        if all(done):
            break
        import torch as t
        ids = t.cat([ids, t.tensor(nxts, device=dev)[:, None]], 1)
    return [tok.decode(o).strip() for o in out]


def _diet_roots() -> set[str]:
    """Every cur string in the training diet (whitespace-stripped) —
    the contamination guard (mathgen small-space scar)."""
    import glob
    import json
    roots = set()
    for f in (glob.glob("data/micromodel_chains_shard*.jsonl")
              + glob.glob("data/micromodel_algebra_shard*.jsonl")
              + ["data/step_chains.jsonl"]):
        for line in open(f):
            roots.add(json.loads(line)["cur"].replace(" ", ""))
    return roots


def main(ckpt: str, levels: tuple[int, ...], unseen: bool) -> None:
    import sympy as sp
    import torch

    from bench_step_tokens import _gen_isolated
    from bench_verify_fast import verify_wave

    tok = MathTokenizer()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_model(len(tok.vocab)).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(dev).eval()
    roots = _diet_roots() if unseen else set()
    skipped = 0

    valid = tried = solved_any = 0
    per_lv: dict[int, list[int]] = {}
    with torch.no_grad():
        for lv in levels:
            per_lv[lv] = [0, 0]  # valid, tried
            for i in range(N_PER):
                p = _gen_isolated(lv, SEED0 + 1000 * lv + i)
                if p is None:
                    continue
                cur = f"Integral({sp.sstr(p._expr)}, x)"
                if cur.replace(" ", "") in roots:
                    skipped += 1
                    continue
                prompt = tok.encode(
                    f"Current: {cur}\nHints: none\nStep: ")
                cands = []
                for w in range(WAVES):
                    cands += sample_wave(
                        model, tok, prompt,
                        [SEED0 + i * 977 + w * B + b for b in range(B)],
                        dev)
                tried += len(cands)
                distinct = [c for c in dict.fromkeys(cands)
                            if c and c.replace(" ", "")
                            != cur.replace(" ", "")]
                if not distinct:
                    continue
                wv = verify_wave(cur, distinct)
                per = {c: wv.get(c, (False, False)) for c in distinct}
                for c in cands:
                    ok, so = per.get(c, (False, False))
                    if ok and c.replace(" ", "") != cur.replace(" ", ""):
                        valid += 1
                        per_lv[lv][0] += 1
                        if so:
                            solved_any += 1
                per_lv[lv][1] += len(cands)
        v = 100 * valid / max(tried, 1)
    lv_s = " ".join(f"L{lv}:{100*a/max(b,1):.1f}%"
                    for lv, (a, b) in per_lv.items())
    print(f"gate [{ckpt}{' unseen' if unseen else ''}, "
          f"skipped {skipped}]: validity {v:.2f}% ({valid}/{tried}), "
          f"solving {solved_any} | {lv_s}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/mathnative_19m.pt")
    ap.add_argument("--levels", default="2,3")
    ap.add_argument("--unseen", action="store_true")
    a = ap.parse_args()
    main(a.ckpt, tuple(int(s) for s in a.levels.split(",")), a.unseen)
