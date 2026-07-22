"""Potential-shaped GRPO on the gen-6 champion (2026-07-21, Artin
GO — 'ahead of metabolic v3'). The b-lever: reward bandwidth.
r = verified * (1 + LAM * tanh((Phi(cur)-Phi(next))/SCALE)),
Phi = -(count_ops + 40*n_Integral). Unverified stays 0 (oracle
floor intact; Ng-shaping preserves optimal policy). Monkeypatches
G.collect's r_of via a wrapped collect; everything else (driver,
gates, rollback) is the production harness. Pre-registered against
the plateau: solves flat by cycle 4 in every unshaped run — shaped
must beat +2 solves over 12 cycles or the b-lever nulls."""
import sys, shutil
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import sympy as sp
import step_grpo_micro as G

LAM, SCALE = 0.5, 20.0

def phi(s):
    try:
        e = sp.sympify(s)
        return -(float(sp.count_ops(e)) +
                 40.0 * s.count("Integral("))
    except Exception:
        return None

_orig = G.verify_wave if hasattr(G, "verify_wave") else None
from bench_verify_fast import verify_wave as _vw

def shaped_collect(model, tok, dev, n_groups, seed0):
    import math, torch
    from bench_step_tokens import _gen_isolated
    groups, mined = [], []
    stats = {"waves": 0, "mixed": 0, "allfail": 0, "allpass": 0}
    pi = 0
    with torch.no_grad():
        while len(groups) < n_groups:
            lv = G.LEVELS[pi % len(G.LEVELS)]
            p = _gen_isolated(lv, seed0 + pi)
            pi += 1
            if p is None:
                continue
            cur = f"Integral({sp.sstr(p._expr)}, x)"
            visited = {cur.replace(" ", "")}
            for ply in range(12):
                prompt = tok.encode(
                    f"Current: {cur}\nHints: none\nStep: ")
                texts, tok_ids, lps = G.sample_wave_lp(
                    model, tok, prompt,
                    [seed0 * 13 + pi * 977 + ply * 101 + b
                     for b in range(G.B)], dev)
                stats["waves"] += 1
                distinct = [t_ for t_ in dict.fromkeys(texts) if t_]
                wv = _vw(cur, distinct) if distinct else {}
                pc = phi(cur)

                def r_of(t_):
                    ok, _ = wv.get(t_, (False, False))
                    if not ok or t_.replace(" ", "") in visited:
                        return 0.0
                    pn = phi(t_)
                    if pc is None or pn is None:
                        return 1.0
                    return 1.0 + LAM * math.tanh((pn - pc) / SCALE)
                rewards = [r_of(t_) for t_ in texts]
                n_ok = sum(1 for r in rewards if r > 0)
                if 0 < n_ok < G.B:
                    stats["mixed"] += 1
                    groups.append({"prompt": prompt,
                                   "tok_ids": tok_ids, "logps": lps,
                                   "rewards": rewards, "level": lv})
                elif n_ok == 0:
                    stats["allfail"] += 1
                else:
                    stats["allpass"] += 1
                nxt = None
                for t_ in texts:
                    ok, so = wv.get(t_, (False, False))
                    if ok and t_.replace(" ", "") not in visited:
                        mined.append({"cur": cur, "nxt": t_,
                                      "level": lv,
                                      "source": "grpo-shaped"})
                        if so:
                            nxt = "SOLVED"
                            break
                        nxt = t_
                if nxt in (None, "SOLVED"):
                    break
                cur = nxt
                visited.add(cur.replace(" ", ""))
                if len(groups) >= n_groups:
                    break
    return groups, mined, stats

G.collect = shaped_collect
if __name__ == "__main__":
    shutil.copy("checkpoints/mathnative_gen6_grown.pt",
                "checkpoints/grpo_shaped.pt")
    G.main(cycles=12, out_path="checkpoints/grpo_shaped.pt",
           d=512, layers=12, ffn=2304, heads=8)
