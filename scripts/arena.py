"""THE ARENA: engine vs the 0.5B step-model, same integral, live.

Watch the machines think. Left lane: the search engine's winning
derivation (replayed move by move with rule names). Right lane: the
LLM emitting verified macro-tokens — every attempt printed with its
oracle verdict (valid step / rejected / SOLVED).

    .venv/bin/python3 scripts/arena.py --level 5 --seed 42
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

GREEN, RED, DIM, BOLD, CYAN, END = ("\033[92m", "\033[91m", "\033[2m",
                                    "\033[1m", "\033[96m", "\033[0m")


def engine_lane(level: int, seed: int, q: "mp.Queue") -> None:
    import sympy as sp

    from llmopt.mathgen.problems import make_integrate
    from llmopt.search import derivation as D
    from llmopt.search.derivation import State, successors
    from llmopt.search.engine import solve

    p = make_integrate(level, seed)
    root = sp.Integral(p._expr, sp.Symbol("x"))
    t0 = time.monotonic()
    res = solve(root, budget=200)
    wall = time.monotonic() - t0
    if not res.solved:
        q.put({"integrand": sp.sstr(p._expr), "solved": False,
               "wall": wall, "moves": []})
        return
    saved, D.RULE_WALL = D.RULE_WALL, 60.0
    hist = res.state.history
    moves: list = []

    def walk(cur: State, i: int, acc: list) -> bool:
        if i == len(hist):
            moves.extend(acc)
            return True
        for name, child in successors(cur, use_macros=True,
                                      verify_p=1.0):
            if name == hist[i] and walk(
                    child, i + 1, acc + [(name, sp.sstr(child.expr))]):
                return True
        return False

    walk(State(root), 0, [])
    D.RULE_WALL = saved
    q.put({"integrand": sp.sstr(p._expr), "solved": True,
           "wall": wall, "nodes": res.nodes, "moves": moves})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12_000_000)
    ap.add_argument("--budget", type=int, default=768)
    a = ap.parse_args()

    # ---- engine lane (fork-isolated, wall 120)
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    pr = ctx.Process(target=engine_lane, args=(a.level, a.seed, q))
    pr.start()
    pr.join(120)
    if pr.is_alive():
        pr.kill()
        pr.join()
        print("engine lane hung — pick another seed")
        return
    eng = q.get(timeout=10)

    print(f"\n{BOLD}══ THE ARENA ══{END}  L{a.level} seed {a.seed}")
    print(f"{BOLD}∫ {eng['integrand']} dx{END}\n")

    print(f"{CYAN}{BOLD}── ENGINE ──{END}")
    if eng["solved"]:
        for name, state in eng["moves"]:
            print(f"  {CYAN}{name:>16s}{END} → {state[:72]}")
        print(f"  {GREEN}SOLVED{END} in {eng['wall']:.2f}s "
              f"({eng['nodes']} nodes, {len(eng['moves'])} plies)\n")
    else:
        print(f"  {RED}unsolved at budget{END} ({eng['wall']:.1f}s)\n")

    # ---- LLM lane, live
    print(f"{CYAN}{BOLD}── 0.5B STEP-MODEL ──{END}  "
          f"{DIM}(loading…){END}", flush=True)
    from bench_step_tokens import (FEWSHOT, _hints_isolated, load,
                                   sample_batch, verify_step)
    tok, model = load("checkpoints/step_lora.pt")
    cur = f"Integral({eng['integrand']}, x)"
    used = j = 0
    t0 = time.monotonic()
    solved = False
    while used < a.budget and not solved:
        hints = ", ".join(_hints_isolated(cur)) or "none"
        prompt = FEWSHOT + f"\nCurrent: {cur}\nHints: {hints}\nStep:"
        texts, spents = sample_batch(
            tok, model, prompt,
            seeds=[7_000 + 7919 * (j + b) for b in range(8)],
            constrain=True)
        j += 8
        for text, spent in zip(texts, spents):
            used += spent
            if solved or used > a.budget:
                continue
            line = text.splitlines()[0].strip() if text else ""
            cand = line.split("=>")[-1].strip()
            if not cand:
                print(f"  {DIM}∅ (empty){END}")
                continue
            okp, is_solved = verify_step(cur, cand)
            if okp:
                mark = (f"{GREEN}★ SOLVED{END}" if is_solved
                        else f"{GREEN}✓ step{END}")
                print(f"  {mark}  {line[:80]}")
                cur = cand
                solved = is_solved
            else:
                print(f"  {RED}✗{END} {DIM}{line[:76]}{END}")
    wall = time.monotonic() - t0
    verdict = (f"{GREEN}SOLVED{END}" if solved
               else f"{RED}budget exhausted{END}")
    print(f"\n  {verdict} — {used} tokens, {wall:.0f}s\n")


if __name__ == "__main__":
    main()
