"""Autonomous expert-iteration loop driver (spec:
docs/superpowers/specs/2026-07-12-step-expert-iteration-design.md).
Round = evaluate -> mine -> train -> gate; state on disk; tripwires
halt the loop. All sympy touches forked (pathologies #7/#8/#10)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def evaluate(tok, model, levels, n_per, seed_base, budget=768):
    """Frontier scan: solve rate per level (stop below 20%), overall
    step validity, and the verified chains from solved traces (the
    on-policy mining source)."""
    import sympy as sp

    from bench_step_tokens import _gen_isolated, solve_chain
    sb: dict = {"solves": {}, "validity": 0.0, "chains": {}}
    valid = tried = 0
    for lv in levels:
        s = 0
        sb["chains"][lv] = []
        for i in range(n_per):
            p = _gen_isolated(lv, seed_base + 1000 * lv + i)
            if p is None:
                continue
            ok, pairs, v, t = solve_chain(
                tok, model, sp.sstr(p._expr), budget,
                seed0=seed_base + 1000 * lv + i)
            s += ok
            valid += v
            tried += t
            if ok:
                sb["chains"][lv].extend(pairs)
        sb["solves"][lv] = s
        if s < 0.2 * n_per:      # frontier scan stops below 20%
            break
    sb["validity"] = 100.0 * valid / max(tried, 1)
    return sb


def frontier(sb: dict, n_per: int) -> int:
    """Highest level in the 20-80% solve band; else highest evaluated."""
    band = [lv for lv, s in sb["solves"].items()
            if 0.2 * n_per <= s <= 0.8 * n_per]
    return max(band) if band else max(sb["solves"])


def gate_verdict(prev: dict, new: dict, frontier: int) -> tuple[bool, str]:
    """PROMOTE iff no level <= frontier regresses by more than 2
    solves AND (frontier solves improve OR validity gains >= 2 pts)."""
    for lv, s in prev["solves"].items():
        if lv <= frontier and new["solves"].get(lv, 0) < s - 2:
            return False, f"L{lv} regressed {s}->{new['solves'].get(lv, 0)}"
    gain_frontier = (new["solves"].get(frontier, 0)
                     > prev["solves"].get(frontier, 0))
    gain_validity = new["validity"] >= prev["validity"] + 2.0
    if gain_frontier or gain_validity:
        return True, "frontier gain" if gain_frontier else "validity gain"
    return False, "no improvement"


def mine_round(round_no: int, F: int, sb: dict, seed_base: int,
               n_mine: int = 60) -> tuple[int, int]:
    """On-policy chains from evaluation + engine chains at F (and F-1
    at 25%), engine capped at max(model adds, 20) — the 50% rule with
    a small bootstrap floor for rounds where the model solved little."""
    import json
    import multiprocessing as mp

    from expert_iter_steps import _chain_worker
    corpus = Path("data/step_chains.jsonl")
    seen = set()
    if corpus.exists():
        for line in corpus.read_text().splitlines():
            r = json.loads(line)
            seen.add((r["cur"], r["nxt"]))
    model_pairs = []
    for lv in (F, F - 1):
        for pair in sb["chains"].get(lv, []):
            pair = tuple(pair)
            if pair not in seen:
                seen.add(pair)
                model_pairs.append((lv, pair))
    ctx = mp.get_context("fork")
    engine_pairs = []
    lv_plan = [F] * int(n_mine * 0.75) + [max(F - 1, 2)] * int(n_mine * 0.25)
    for k, lv in enumerate(lv_plan):
        if len(engine_pairs) >= max(len(model_pairs), 20):
            break
        q = ctx.Queue()
        pr = ctx.Process(target=_chain_worker,
                         args=(lv, seed_base + 500 * round_no + k, q))
        pr.start()
        pr.join(90)
        if pr.is_alive():
            pr.kill()
            pr.join()
            continue
        try:
            for pair in q.get(timeout=10):
                pair = tuple(pair)
                if pair not in seen:
                    seen.add(pair)
                    engine_pairs.append((lv, pair))
        except Exception:
            continue
    with corpus.open("a") as f:
        for src, items in (("model", model_pairs),
                           ("engine", engine_pairs)):
            for lv, (cur, nxt) in items:
                f.write(json.dumps(
                    {"cur": cur, "nxt": nxt, "level": lv,
                     "round": round_no, "source": src,
                     "gate": "pending"}) + "\n")
    return len(model_pairs), len(engine_pairs)


def run_round(round_no: int) -> str:
    import shutil
    import time

    from bench_step_tokens import load
    from expert_iter_steps import phase_train
    t0 = time.monotonic()
    promoted = Path("checkpoints/step_lora.pt")
    tok, model = load(str(promoted) if promoted.exists() else None)
    sb = evaluate(tok, model, levels=(2, 3, 4, 5), n_per=40,
                  seed_base=8_200_000 + 10_000 * round_no)
    F = frontier(sb, 40)
    n_model, n_engine = mine_round(round_no, F, sb,
                                   seed_base=8_000_000)
    if sb["validity"] < 1.0:
        return f"HALT validity {sb['validity']:.1f}%"
    cand = Path(f"checkpoints/step_lora_r{round_no}.pt")
    phase_train(epochs=3, lr=2e-4, out=cand)
    del model
    tok, model = load(str(cand))
    gate_sb = evaluate(tok, model, levels=tuple(range(2, F + 1)),
                       n_per=40, seed_base=8_400_000)
    ok, reason = gate_verdict(
        {"solves": sb["solves"], "validity": sb["validity"]},
        {"solves": gate_sb["solves"],
         "validity": gate_sb["validity"]}, F)
    if ok:
        shutil.copy(cand, promoted)
    mins = (time.monotonic() - t0) / 60
    line = (f"| {round_no} | F=L{F} | +{n_model}m/+{n_engine}e | "
            f"val {sb['validity']:.1f}->{gate_sb['validity']:.1f} | "
            f"{sb['solves']} -> {gate_sb['solves']} | "
            f"{'PROMOTE' if ok else 'ROLLBACK'}: {reason} | "
            f"{mins:.0f}m |")
    with open("docs/LOOP-LOG.md", "a") as f:
        f.write(line + "\n")
    if mins > 180:
        return f"HALT wall {mins:.0f}m"
    return "PROMOTE" if ok else "ROLLBACK"


def main(max_rounds: int) -> None:
    fails = 0
    for r in range(1, max_rounds + 1):
        verdict = run_round(r)
        print(f"round {r}: {verdict}", flush=True)
        if verdict.startswith("HALT"):
            break
        fails = 0 if verdict == "PROMOTE" else fails + 1
        if fails >= 2:
            print("HALT: two consecutive failed gates", flush=True)
            break


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    a = ap.parse_args()
    main(a.rounds)
