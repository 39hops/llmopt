"""TENET R1b-micro: the closed loop at matched budget (PRE-REG
TENET-R1B-MICRO, 2026-08-06 — read it first; arms, ledger contract,
and registered lines live there, not here).

Three arms on the 120 forward-gate band problems, 12-ply verified
chains, ONE BudgetAccountant per problem (B_TOK=6000, real token
counts, debit-before-use, refusal ends the problem for that arm):
  F   forward-only + re-rolls (the no-reverse-model baseline)
  FR  reverse-twin cycle-consistency ranker over verified cands
  LR  shortest-first ranker (zero scoring cost; the length control)

Every wave is charged via tenet_d3_budget.charge with REAL token
counts (len(tok.encode(text))+1 per emission); FR's scoring charges
the teacher-forced parent token count per candidate scored. Oracle
calls are the shared referee and are NOT budgeted (identical across
arms). Rows stream per problem (streaming corollary).

Usage: ARM=F|FR|LR .venv/bin/python scratch/tenet_r1b_micro.py
       (N=2 for smoke tier; runs all levels at n/level)
Output: logs/tenet_r1b_<arm>.jsonl rows + a terminal census.
"""
from llmopt.common.device import pick_device
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from tenet_d3_budget import BudgetAccountant  # noqa: E402

ARM = os.environ["ARM"]
B_TOK = int(os.environ.get("B_TOK", "6000"))
N = int(os.environ.get("N", "0")) or G.GATE_N
FWD_CKPT = "checkpoints/sym_birth_dense_fwdcert.pt"
REV_CKPT = "checkpoints/sym_birth_dense_revcert.pt"
OUT = Path(f"logs/tenet_r1b_{ARM}.jsonl")
norm = lambda s: s.replace(" ", "")  # noqa: E731


def tok_cost(tok, text):
    try:
        return len(tok.encode(text)) + 1
    except ValueError:
        return len(text)  # unencodable emission: char-count bound


def rev_score(rev_model, tok, cand, cur, dev):
    """Cycle-consistency: teacher-forced logprob of CUR given CAND
    as the reverse prompt. Returns (score, token_cost)."""
    try:
        prompt = tok.encode(f"Current: {cand}\nHints: none\nStep: ")
        target = tok.encode(cur) + [tok.eos_id]
    except ValueError:
        return float("-inf"), 0
    ids = torch.tensor([prompt + target], device=dev)
    with torch.no_grad():
        logits = rev_model(ids)[0]
    lp = 0.0
    for j, t in enumerate(target):
        pos = len(prompt) + j - 1
        lp += float(torch.log_softmax(
            logits[pos].float(), -1)[t])
    return lp / len(target), len(target)


def run_problem(models, tok, dev, root, seed0):
    from bench_verify_fast import verify_wave
    fwd, rev = models
    acct = BudgetAccountant(total=B_TOK)
    solved = False
    plies_used = 0
    roll = 0
    while not solved:
        cur, visited = root, {norm(root)}
        for ply in range(12):
            prompt = tok.encode(
                f"Current: {cur}\nHints: none\nStep: ")
            seeds = [seed0 + roll * 977 + ply * 7 + b
                     for b in range(G.B)]
            with torch.no_grad():
                texts, _, _ = G.sample_wave_lp(
                    fwd, tok, prompt, seeds, dev)
            cost = sum(tok_cost(tok, t) for t in texts if t)
            if not acct.debit(cost, "forward"):
                return solved, acct, plies_used  # budget spent
            plies_used += 1
            distinct = [t for t in dict.fromkeys(texts)
                        if t and norm(t) not in visited]
            wv = verify_wave(cur, distinct) if distinct else {}
            verified = []
            for t in distinct:
                ok, so = wv.get(t, (False, False))
                if ok and so:
                    return True, acct, plies_used
                if ok:
                    verified.append(t)
            if not verified:
                break  # stall: fall through to a re-roll
            if ARM == "FR":
                scored = []
                for t in verified:
                    sc, c = rev_score(rev, tok, t, cur, dev)
                    if c and not acct.debit(c, "rev_score"):
                        return solved, acct, plies_used
                    scored.append((sc, t))
                verified = [t for _, t in
                            sorted(scored, key=lambda x: -x[0])]
            elif ARM == "LR":
                verified.sort(key=len)
            cur = verified[0]
            visited.add(norm(cur))
        roll += 1  # chain ended unsolved (stall or 12 plies): re-roll
        if acct.remaining() < 200:  # no wave fits; stop honestly
            break
    return solved, acct, plies_used


def main():
    import hashlib

    import sympy as sp

    from bench_step_tokens import _gen_isolated
    from llmopt.train.mathnative import MathTokenizer, build_model

    assert ARM in ("F", "FR", "LR")
    tok = MathTokenizer()
    dev = pick_device()
    def load(ckpt):
        m = build_model(len(tok.vocab), d=64, layers=8, heads=4,
                        ffn=256).to(dev)
        m.load_state_dict(torch.load(ckpt, map_location="cpu",
                                     weights_only=True))
        m.eval()
        return m
    fwd = load(FWD_CKPT)
    rev = load(REV_CKPT) if ARM == "FR" else None
    for name, m in (("fwd", fwd), ("rev", rev)):
        if m is None:
            continue
        h = hashlib.sha256()
        for k, v in sorted(m.state_dict().items()):
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        print(f"[r1b:{ARM}] {name} weights sha "
              f"{h.hexdigest()[:16]}", flush=True)
    solves, spent_tot = {}, 0
    f = OUT.open("a")
    for lv in G.GATE_LEVELS:
        s_lv = 0
        for i in range(N):
            p = _gen_isolated(lv, G.GATE_BAND + 1000 * lv + i)
            if p is None:
                continue
            root = f"Integral({sp.sstr(p._expr)}, x)"
            ok, acct, plies = run_problem(
                (fwd, rev), tok, dev, root,
                G.GATE_BAND + i * 31)
            c = acct.census()
            spent_tot += c["spent"]
            s_lv += ok
            f.write(json.dumps(
                {"arm": ARM, "level": lv, "i": i, "solved": ok,
                 "plies": plies, **c}) + "\n")
            f.flush()
        solves[lv] = s_lv
        print(f"[r1b:{ARM}] L{lv}: {s_lv}/{N}", flush=True)
    f.write(json.dumps({"terminal": True, "arm": ARM,
                        "solves": solves,
                        "total": sum(solves.values()),
                        "spent_total": spent_tot}) + "\n")
    f.close()
    print(f"[r1b:{ARM}] TERMINAL {solves} = "
          f"{sum(solves.values())}/{5 * N} | tokens {spent_tot}",
          flush=True)


if __name__ == "__main__":
    main()
