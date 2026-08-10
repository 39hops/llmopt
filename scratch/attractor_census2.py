"""Free-running census, THROUGH answer-form + n_succ join
(PRE-REG ATTRACTOR-0B-JOIN, 2026-08-10).

Extends attractor_census.py two ways: the solved_form stop is
REMOVED (iterate until fixed_point / cycle / malformed / cap), and
every visited state is recorded with n_succ computed for each
Integral-bearing state in a FORK-ISOLATED worker (10 s deadline;
killed class streams as n_succ None — no silent checkpoint
selection). Same instrument constants as ATTRACTOR-0.

Usage: .venv/bin/python scratch/attractor_census2.py
"""
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp  # noqa: E402
import torch  # noqa: E402

from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402
from bench_step_tokens import _gen_isolated  # noqa: E402

CKPT = "checkpoints/mathnative_wfloor_d256.pt"
D, LAYERS, FFN, HEADS = 256, 8, 1024, 4
BAND = 93_000_000  # SAME band as ATTRACTOR-0 (paired trajectories)
LEVELS = (3, 4, 5, 6, 7)
N_PER = 40
ITERS = 50
MAX_NEW = 120
NSUCC_DEADLINE = 10.0  # seconds, fork-walled (no sympy under SIGALRM)
OUT = "logs/data_ceil/attractor_census2_d256.jsonl"
DEV = "cpu"


def _nsucc_worker(cur, q):
    try:
        from llmopt.search.derivation import State, successors
        expr = sp.sympify(cur)
        q.put(sum(1 for _ in successors(State(expr))))
    except Exception:
        q.put(None)


def n_succ_isolated(cur):
    """Fork, join with deadline, SIGKILL — the solve_isolated pattern."""
    q = mp.Queue()
    p = mp.Process(target=_nsucc_worker, args=(cur, q))
    p.start()
    p.join(NSUCC_DEADLINE)
    if p.is_alive():
        p.kill()
        p.join()
        return None
    try:
        return q.get_nowait()
    except Exception:
        return None


@torch.no_grad()
def greedy_step(model, tok, prompt_ids):
    ids = torch.tensor([prompt_ids], device=DEV)
    margins, toks = [], []
    for _ in range(MAX_NEW):
        lg = model(ids)[0, -1]
        top2 = lg.topk(2)
        margins.append(float(top2.values[0] - top2.values[1]))
        n = int(top2.indices[0])
        if n == tok.eos_id:
            break
        toks.append(n)
        ids = torch.cat([ids, torch.tensor([[n]], device=DEV)], dim=1)
    return tok.decode(toks).strip(), min(margins) if margins else 0.0


def norm(s):
    return s.replace(" ", "")


def main():
    mp.set_start_method("fork", force=True)
    tok = MathTokenizer()
    model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN).to(DEV)
    model.load_state_dict(
        torch.load(CKPT, map_location="cpu", weights_only=True))
    model.eval()

    os.makedirs("logs/data_ceil", exist_ok=True)
    out = open(OUT, "a")
    for lv in LEVELS:
        for i in range(N_PER):
            p = _gen_isolated(lv, BAND + 1000 * lv + i)
            if p is None:
                out.write(json.dumps(
                    {"level": lv, "i": i, "cls": "genfail"}) + "\n")
                out.flush()
                continue
            cur = f"Integral({sp.sstr(p._expr)}, x)"
            seen = {norm(cur): 0}
            steps = []  # per step: state BEFORE move, margin, kind
            cls, k, t_first_af = "iter_cap", None, None
            cycle_states = []
            state = cur
            for t in range(1, ITERS + 1):
                try:
                    ids = tok.encode(
                        f"Current: {state}\nHints: none\nStep: ")
                except ValueError:
                    cls = "malformed"
                    break
                nxt, mm = greedy_step(model, tok, ids)
                ns = (n_succ_isolated(state)
                      if "Integral(" in state else None)
                steps.append({"t": t, "state": state, "mm": round(mm, 4),
                              "n_succ": ns,
                              "in_af": "Integral(" not in state})
                if not nxt:
                    cls = "malformed"
                    break
                nn = norm(nxt)
                if t_first_af is None and "Integral(" not in nxt:
                    t_first_af = t
                if nn == norm(state):
                    cls, k = "fixed_point", 0
                    break
                if nn in seen:
                    cls, k = "cycle", t - seen[nn]
                    cycle_states = [s["state"] for s in steps
                                    if norm(s["state"]) in
                                    (nn, norm(state))]
                    break
                seen[nn] = t
                state = nxt
            out.write(json.dumps({
                "level": lv, "i": i, "cls": cls, "cycle_k": k,
                "t_first_answer_form": t_first_af,
                "n_steps": len(steps), "steps": steps,
                "cycle_states": cycle_states,
            }) + "\n")
            out.flush()
        print(f"[L{lv}] done", flush=True)

    # ---- summary FROM THE ARTIFACT ----
    rows = [json.loads(x) for x in open(OUT)]
    ok = [r for r in rows if r.get("cls") != "genfail"]
    from collections import Counter
    print("classes:", dict(Counter(r["cls"] for r in ok)))

    # Bar 1: trajectories reaching answer-form -> terminal behavior
    af = [r for r in ok if r.get("t_first_answer_form") is not None]
    good = 0
    relaunch = 0
    for r in af:
        if r["cls"] == "fixed_point":
            good += 1
        elif r["cls"] == "cycle" and all(
                "Integral(" not in s for s in r.get("cycle_states", [])):
            good += 1
        else:
            # did any post-answer-form step leave the class?
            post = [s for s in r["steps"]
                    if s["t"] > r["t_first_answer_form"]]
            if any(not s["in_af"] for s in post):
                relaunch += 1
    if af:
        print(f"P-TRUE-FIXED-POINT: {good}/{len(af)} "
              f"({100*good/len(af):.1f}%) terminate in answer-form; "
              f"relaunches: {relaunch}")

    # Bar 2: n_succ at absorbing-move states v transit states
    absorb_ns, transit_ns = [], []
    for r in af:
        pre = [s for s in r["steps"] if not s["in_af"]
               and s["n_succ"] is not None]
        if not pre:
            continue
        # absorbing move = the step whose t == t_first_answer_form
        for s in pre:
            (absorb_ns if s["t"] == r["t_first_answer_form"]
             else transit_ns).append(s["n_succ"])
    absorb_ns.sort()
    transit_ns.sort()
    n_read = len(absorb_ns) + len(transit_ns)
    print(f"n_succ readings: {n_read} "
          f"(absorb {len(absorb_ns)}, transit {len(transit_ns)})")
    if absorb_ns and transit_ns:
        ma = absorb_ns[len(absorb_ns) // 2]
        mt = transit_ns[len(transit_ns) // 2]
        print(f"P-MARGIN-UNIFIED: median n_succ absorbing {ma} "
              f"v transit {mt} (ratio {mt / max(ma, 1):.2f})")


if __name__ == "__main__":
    main()
