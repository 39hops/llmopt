"""MULT-0: the verified-candidate MULTIPLICITY census (spec
2026-08-07-morning-specs.md item 3; rung 1 of the reverse-propose
ladder, RIFF 2026-08-07; the pincer closure's fence made
measurable).

The D1b frame VERBATIM minus one short-circuit: the cited gate
(scratch/tenet_d1_revgate.py, frozen evidence) stops replaying at
the first mint (`if done: continue`); this driver replays EVERY
distinct candidate and counts mints per problem. Everything else is
the frozen code path by import — _gen_isolated band problems,
first-encodable-child poststep prompting, sample_wave_lp B=8 seeds,
dict.fromkeys dedup, the verified-AND-distinct fence, replay
minting through tenet_d2_revdiet.Replayer (the direction-honest
criterion; verify_wave stays diagnostic-only), replay cache,
conservative reject on timeout/membomb/crash.

Readouts: multiplicity histogram m per problem + per level;
P(m >= 2) headline; miss/err census; per-problem rows STREAM to
MULT_LOG (the killed-worker corollary).

Usage: CKPT=checkpoints/sym_birth_dense_revcert.pt \
       MULT_LOG=logs/mult0/rev.jsonl \
       .venv/bin/python scratch/tenet_mult_census.py
Env: MODE=poststep (default; the in-distribution frame), N (per
level, default full gate), D/LAYERS/FFN/HEADS as the gate.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")


def mult_census(model, tok, dev, n=None, mode="poststep", log_f=None):
    import sympy as sp
    import torch

    import step_grpo_micro as G
    from bench_step_tokens import _gen_isolated
    from bench_verify_fast import verify_wave
    from tenet_d2_revdiet import Replayer

    wh = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        v = v.detach().cpu().contiguous()
        if v.dtype == torch.bfloat16:
            v = v.view(torch.int16)
        wh.update(v.numpy().tobytes())
    print(f"[mult0] weights sha {wh.hexdigest()[:16]}", flush=True)
    hist = {}
    valid = tried = skipped = 0
    replayer = Replayer()
    counts = {s: 0 for s in ("unique", "ambig", "miss", "err",
                             "timeout", "membomb", "crash")}
    replay_cache = {}
    with torch.no_grad():
        for lv in G.GATE_LEVELS:
            for i in range(n or G.GATE_N):
                p = _gen_isolated(lv, G.GATE_BAND + 1000 * lv + i)
                if p is None:
                    skipped += 1
                    continue
                s = f"Integral({sp.sstr(p._expr)}, x)"
                if mode == "poststep":
                    from llmopt.search.derivation import (
                        State, successors)
                    try:
                        kids = list(successors(
                            State(sp.sympify(s)), use_macros=True))
                    except Exception:
                        skipped += 1
                        continue
                    s2 = None
                    for _, k_ in kids:
                        cs = sp.sstr(k_.expr)
                        try:
                            tok.encode(f"Current: {cs}\n"
                                       f"Hints: none\nStep: ")
                            s2 = cs
                            break
                        except ValueError:
                            continue
                    if s2 is None:
                        skipped += 1
                        continue
                    s = s2
                s_norm = s.replace(" ", "")
                prompt = tok.encode(
                    f"Current: {s}\nHints: none\nStep: ")
                texts, _, _ = G.sample_wave_lp(
                    model, tok, prompt,
                    [G.GATE_BAND + 1000 * lv + i * 31 + b
                     for b in range(G.B)], dev)
                tried += len(texts)
                m = 0          # THE census change: count every mint,
                distinct = 0   # never stop at the first
                for cand in dict.fromkeys(texts):
                    c_norm = cand.replace(" ", "") if cand else ""
                    if not c_norm or c_norm == s_norm:
                        continue  # verified AND distinct doctrine
                    distinct += 1
                    ok, _ = verify_wave(cand, [s]).get(s, (False, False))
                    valid += ok
                    st = replay_cache.get((c_norm, s_norm))
                    if st is None:
                        st = replayer.check_sync(cand, s, counts)
                        replay_cache[(c_norm, s_norm)] = st
                    if st in ("unique", "ambig"):
                        m += 1
                hist.setdefault(lv, []).append(m)
                if log_f is not None:
                    log_f.write(json.dumps({
                        "level": lv, "idx": i, "m": m,
                        "distinct": distinct}) + "\n")
                    log_f.flush()
            ms = hist.get(lv, [])
            print(f"[mult0] L{lv}: n={len(ms)} "
                  f"P(m>=1)={sum(1 for m in ms if m >= 1)}/{len(ms)} "
                  f"P(m>=2)={sum(1 for m in ms if m >= 2)}/{len(ms)} "
                  f"max m={max(ms, default=0)}", flush=True)
    replayer.kill()
    if skipped:
        print(f"[mult0] SKIPPED {skipped}", flush=True)
    allm = [m for ms in hist.values() for m in ms]
    mh = {m: allm.count(m) for m in range(max(allm, default=0) + 1)}
    print(f"[mult0] TOTAL n={len(allm)} "
          f"P(m>=1)={sum(1 for m in allm if m >= 1)} "
          f"P(m>=2)={sum(1 for m in allm if m >= 2)} "
          f"hist={mh} | replay census {counts} | equiv-valid "
          f"{100 * valid / max(tried, 1):.2f}%", flush=True)
    return hist


if __name__ == "__main__":
    import torch

    from llmopt.train.mathnative import MathTokenizer, build_model

    CKPT = os.environ["CKPT"]
    tok = MathTokenizer()
    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model(
        len(tok.vocab), d=int(os.environ.get("D", "64")),
        layers=int(os.environ.get("LAYERS", "8")),
        heads=int(os.environ.get("HEADS", "4")),
        ffn=int(os.environ.get("FFN", "256"))).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location="cpu",
                                     weights_only=True))
    model.eval()
    log_path = Path(os.environ.get("MULT_LOG", "logs/mult0/census.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    n = int(os.environ["N"]) if os.environ.get("N") else None
    with log_path.open("a") as f:
        mult_census(model, tok, dev, n=n,
                    mode=os.environ.get("MODE", "poststep"), log_f=f)
