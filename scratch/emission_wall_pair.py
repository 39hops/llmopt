"""Rung-1 bar (iii), in-diet form: does prefix move the
operand-complexity emission wall? (spec 2026-07-25-native-transformer;
poly_chain5 psub/padd rows are OUT-OF-DIET for gen-4 twins — bridge
law + naked-forms lesson — so the wall is read on generator-drawn
in-language states instead.)

Paired greedy one-step emission on the SAME _gen_isolated states
(fresh seed band, disjoint from the gate band), each arm prompted in
its own serialization, verified by the same oracle; accuracy bucketed
by cur token length (the operand-complexity proxy). One curve per arm.

    .venv/bin/python scratch/emission_wall_pair.py <infix_ckpt> <prefix_ckpt>
"""
from llmopt.common.device import pick_device
import sys

import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")

import sympy as sp

from bench_step_tokens import _gen_isolated
from bench_verify_fast import verify_wave
from llmopt.mathgen.prefix import from_prefix, to_prefix
from llmopt.train.mathnative import MathTokenizer, build_model

PROBE_BAND = 12_345_000   # disjoint from GATE_BAND (9.9M) + farm bands
PER_LEVEL = 48
LEVELS = (3, 4, 5, 6, 7)

tok = MathTokenizer()
dev = pick_device()


def load(ckpt):
    m = build_model(len(tok.vocab)).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    return m


def greedy(model, ids, max_new=160):
    x = torch.tensor([ids], device=dev)
    logits, past = model(x, use_cache=True)
    out = []
    nxt = int(logits[0, -1].argmax())
    for _ in range(max_new):
        if nxt == tok.eos_id or tok.vocab[nxt] == "\n":
            break
        out.append(nxt)
        col = torch.tensor([[nxt]], device=dev)
        logits, past = model(col, past=past)
        nxt = int(logits[0, -1].argmax())
    return tok.decode(out).strip()


def main():
    inf_m, pre_m = load(sys.argv[1]), load(sys.argv[2])
    x = sp.Symbol("x")
    rows = []   # (complexity, infix_ok, prefix_ok, prefix_parse_ok)
    with torch.no_grad():
        for lv in LEVELS:
            for i in range(PER_LEVEL):
                p = _gen_isolated(lv, PROBE_BAND + 1000 * lv + i)
                if p is None:
                    continue
                cur = sp.Integral(p._expr, x)
                cur_inf = sp.sstr(cur)
                comp = len(tok.encode(cur_inf, strict=False))
                t_inf = greedy(inf_m, tok.encode(
                    f"Current: {cur_inf}\nHints: none\nStep: "))
                ok_i = bool(t_inf) and verify_wave(cur_inf, [t_inf]).get(
                    t_inf, (False, False))[0]
                t_pre = greedy(pre_m, tok.encode(
                    f"Current: {to_prefix(cur)}\nHints: none\nStep: "))
                parse_ok, ok_p = False, False
                if t_pre:
                    try:
                        e = from_prefix(t_pre)
                        parse_ok = True
                        inf = sp.sstr(e)
                        ok_p = verify_wave(cur_inf, [inf]).get(
                            inf, (False, False))[0]
                    except Exception:
                        pass
                rows.append((comp, ok_i, ok_p, parse_ok))
    rows.sort()
    n = len(rows)
    print(f"{n} states; per-arm one-step validity by cur-complexity "
          f"quartile (tokens):")
    for qi in range(4):
        seg = rows[qi * n // 4:(qi + 1) * n // 4]
        lo, hi = seg[0][0], seg[-1][0]
        vi = sum(r[1] for r in seg) / len(seg)
        vp = sum(r[2] for r in seg) / len(seg)
        pf = 1 - sum(r[3] for r in seg) / len(seg)
        print(f"  Q{qi + 1} [{lo:3d}-{hi:3d}]: infix {100 * vi:5.1f}%  "
              f"prefix {100 * vp:5.1f}%  (prefix parse-fail "
              f"{100 * pf:4.1f}%)", flush=True)
    vi = sum(r[1] for r in rows) / n
    vp = sum(r[2] for r in rows) / n
    print(f"  ALL: infix {100 * vi:.1f}%  prefix {100 * vp:.1f}%")


if __name__ == "__main__":
    main()
