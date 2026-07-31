"""Gate transcript dump (Artin's ask, 2026-07-31 night): print the
model's ACTUAL step chains on gate prompts — how it works a problem,
not just whether it solved. Mirrors gate_eval's loop exactly (same
seeds, same wave sampler, same oracle) but records every accepted
step plus the rejected-sample count per ply.
Usage: CKPT=checkpoints/umoe_gravmoe_s1.pt LEVEL=4 N=6 \
       python scratch/gate_transcripts.py
"""
import os
import sys

os.environ.setdefault("ARM", "gravmoe")
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import sympy as sp  # noqa: E402
import torch  # noqa: E402

import umoe_conserve as U  # noqa: E402
from bench_step_tokens import _gen_isolated  # noqa: E402
from bench_verify_fast import verify_wave  # noqa: E402
from step_grpo_micro import (B, GATE_BAND, sample_wave_lp)  # noqa: E402

CKPT = os.environ.get("CKPT", "checkpoints/umoe_gravmoe_s1.pt")
LEVEL = int(os.environ.get("LEVEL", "4"))
N = int(os.environ.get("N", "6"))


def main():
    dev = ("mps" if torch.backends.mps.is_available() else
           "cuda" if torch.cuda.is_available() else "cpu")
    tok, m = U.build()
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)["sd"]
    m.load_state_dict(sd)
    m = m.to(dev).eval()
    print(f"[transcripts] {CKPT} L{LEVEL} n={N} dev {dev}")
    with torch.no_grad():
        for i in range(N):
            p = _gen_isolated(LEVEL, GATE_BAND + 1000 * LEVEL + i)
            if p is None:
                print(f"\n=== prompt {i}: generator timeout, skipped")
                continue
            cur = f"Integral({sp.sstr(p._expr)}, x)"
            print(f"\n=== prompt {i}: {cur}")
            visited = {cur.replace(" ", "")}
            done = False
            nxt = None
            for ply in range(12):
                prompt = tok.encode(
                    f"Current: {cur}\nHints: none\nStep: ")
                texts, _, _ = sample_wave_lp(
                    m, tok, prompt,
                    [GATE_BAND + i * 31 + ply * 7 + b
                     for b in range(B)], dev)
                distinct = [t for t in dict.fromkeys(texts)
                            if t and t.replace(" ", "") not in visited]
                wv = verify_wave(cur, distinct) if distinct else {}
                n_valid = sum(1 for ok, _ in wv.values() if ok)
                nxt = None
                for t in texts:
                    ok, so = wv.get(t, (False, False))
                    if ok and t.replace(" ", "") not in visited:
                        if nxt is None:
                            nxt = "SOLVED" if so else t
                if nxt == "SOLVED":
                    solver = next(t for t in texts
                                  if wv.get(t, (0, 0))[1])
                    print(f"  ply {ply}: [{n_valid}/{len(distinct)} "
                          f"valid] -> SOLVED via {solver}")
                    done = True
                    break
                if nxt is None:
                    print(f"  ply {ply}: [0/{len(distinct)} valid] "
                          f"STUCK; samples: "
                          f"{[t[:60] for t in distinct[:3]]}")
                    break
                print(f"  ply {ply}: [{n_valid}/{len(distinct)} "
                      f"valid] -> {nxt}")
                cur = nxt
                visited.add(cur.replace(" ", ""))
            if not done and nxt is not None:
                print("  ... ply budget exhausted (12)")
            print(f"  RESULT: {'SOLVED' if done else 'unsolved'}")


if __name__ == "__main__":
    main()
