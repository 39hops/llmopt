"""Greedy-first adoption cell (pre-reg 2026-07-28 night): on the
FULL production gate battery (same seeds/levels as gate_eval),
race (a) wave-8 (production) vs (b) greedy-first with wave-8
retry only at plies where greedy's candidate fails verification.
Same chain semantics as gate_eval (12 plies, oracle-picked).
Usage: greedy_first_gate.py <ckpt> <d> <layers> <ffn> <heads> <label>
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp  # noqa: E402
import torch  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402
from bench_step_tokens import _gen_isolated  # noqa: E402
from bench_verify_fast import verify_wave  # noqa: E402

ckpt, d, layers, ffn, heads, label = (sys.argv[1], int(sys.argv[2]),
    int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
tok = MathTokenizer()
dev = ("mps" if torch.backends.mps.is_available() else
       "cuda" if torch.cuda.is_available() else "cpu")
if dev == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
model = build_model(len(tok.vocab), d=d, layers=layers, heads=heads,
                    ffn=ffn).to(dev)
model.load_state_dict(torch.load(ckpt, map_location="cpu",
                                 weights_only=True))
model.eval()
NL = tok.encode("\n")[-1]
PLIES = 12


@torch.no_grad()
def greedy(prompt, spend):
    out = list(prompt)
    for _ in range(120):
        t = int(model(torch.tensor([out], device=dev))[0, -1].argmax())
        spend[0] += 1
        out.append(t)
        if t in (tok.eos_id, NL):
            break
    return tok.decode(out[len(prompt):]).strip("\n")


@torch.no_grad()
def run(arm):
    solves = {}
    spend = [0]
    for lv in G.GATE_LEVELS:
        s = 0
        for i in range(G.GATE_N):
            p = _gen_isolated(lv, G.GATE_BAND + 1000 * lv + i)
            if p is None:
                continue
            cur = f"Integral({sp.sstr(p._expr)}, x)"
            visited = {cur.replace(" ", "")}
            for ply in range(PLIES):
                prompt = tok.encode(
                    f"Current: {cur}\nHints: none\nStep: ")
                cands = []
                wv = {}
                if arm == "gfirst":
                    g = greedy(prompt, spend)
                    if g and g.replace(" ", "") not in visited:
                        cands = [g]
                    wv = verify_wave(cur, cands) if cands else {}
                    if not any(wv.get(t, (False, False))[0]
                               for t in cands):
                        cands = []  # greedy failed -> wave retry
                if not cands:
                    texts, _, _ = G.sample_wave_lp(
                        model, tok, prompt,
                        [G.GATE_BAND + i * 31 + ply * 7 + b
                         for b in range(G.B)], dev)
                    spend[0] += sum(len(tok.encode(t)) + 1
                                    for t in texts)
                    cands = [t for t in dict.fromkeys(texts)
                             if t and t.replace(" ", "")
                             not in visited]
                    wv = verify_wave(cur, cands) if cands else {}
                nxt = None
                for t in cands:
                    ok, so = wv.get(t, (False, False))
                    if ok:
                        nxt = "SOLVED" if so else t
                        if so:
                            break
                if nxt == "SOLVED":
                    s += 1
                    break
                if nxt is None:
                    break
                cur = nxt
                visited.add(cur.replace(" ", ""))
        solves[lv] = s
    return solves, spend[0]


for arm in ("wave", "gfirst"):
    sv, sp_ = run(arm)
    print(f"{label} {arm}: {sv} = {sum(sv.values())}/120 "
          f"tokens {sp_}", flush=True)
