"""ATOM-DIET-1 farm (pre-reg RESULTS 2026-08-14): rule-tagged atom
shard — engine one-ply solves, one indivisible verified rewrite per
row. Self-contained forked worker (the solve_isolated timebox law;
expert_iter_steps stays frozen): make_integrate(level, seed), engine
solve budget 200, keep only true one-ply solves (len(history) == 1)
with cur AND nxt in-language (tokenizer roundtrip). Guards at farm
time: D2 gate-band exclusion on norm(cur)/norm(nxt), corpus-cur
dedup against the excised stock diet (no new one-to-many ambiguity),
(cur, nxt) dedup within the shard. Rows stream incrementally to
data/micromodel_atoms_shard0.jsonl; rule-tag distribution printed.

Seed band 71,000,000 (fresh). Levels 3-7, targets L4 2,400 /
900 each L3,L5,L6,L7. Waves of 8 forked workers, 60 s deadline each.

Usage: .venv/bin/python -u scratch/farm_atoms.py
"""
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

SEED_BASE = 71_000_000
TARGETS = {4: 2400, 3: 900, 5: 900, 6: 900, 7: 900}
SEED_CAP = 30_000          # per level, honest-starvation stop
WAVE = 8
DEADLINE = 60.0
OUT = Path("data/micromodel_atoms_shard0.jsonl")


def _worker(level: int, seed: int, q) -> None:
    import sympy as sp

    from llmopt.mathgen.problems import make_integrate
    from llmopt.search.engine import solve

    p = make_integrate(level, seed)
    root = sp.Integral(p._expr, sp.Symbol("x"))
    res = solve(root, budget=200)
    if not res.solved or len(res.state.history) != 1:
        q.put(None)
        return
    q.put((sp.sstr(root), sp.sstr(res.state.expr),
           res.state.history[0].split("@")[0]))


def main() -> None:
    os.environ.setdefault("ARM", "off")
    import birth19m_curric as C
    from llmopt.train.mathnative import MathTokenizer
    from tenet_d2_revdiet import gate_band_exprs, norm

    tok = MathTokenizer()

    def in_language(text: str) -> bool:
        return tok.decode(tok.encode(text)) == text

    band = set(gate_band_exprs())
    corpus_curs = {norm(str(r["cur"])) for r in C.load_excised_rows()}
    print(f"[atoms] band {len(band)} exprs, corpus curs "
          f"{len(corpus_curs)}", flush=True)

    ctx = mp.get_context("fork")
    seen: set = set()
    rules: Counter = Counter()
    counts = {lv: 0 for lv in TARGETS}
    skipped = Counter()
    t0 = time.time()
    with OUT.open("a") as f:
        for lv, target in TARGETS.items():
            i = 0
            while counts[lv] < target and i < SEED_CAP:
                wave = []
                for _ in range(WAVE):
                    if i >= SEED_CAP:
                        break
                    q = ctx.Queue()
                    pr = ctx.Process(target=_worker,
                                     args=(lv, SEED_BASE + i, q))
                    pr.start()
                    wave.append((pr, q))
                    i += 1
                for pr, q in wave:
                    pr.join(DEADLINE)
                    if pr.is_alive():
                        pr.kill()
                        pr.join()
                        skipped["timeout"] += 1
                        continue
                    try:
                        got = q.get(timeout=10)
                    except Exception:
                        skipped["queue"] += 1
                        continue
                    if got is None:
                        skipped["not-oneply"] += 1
                        continue
                    cur, nxt, rule = got
                    if counts[lv] >= target:
                        continue
                    if not (in_language(cur) and in_language(nxt)):
                        skipped["language"] += 1
                        continue
                    if norm(cur) in band or norm(nxt) in band:
                        skipped["gate-band"] += 1
                        continue
                    if norm(cur) in corpus_curs:
                        skipped["corpus-cur"] += 1
                        continue
                    if (cur, nxt) in seen:
                        skipped["dup"] += 1
                        continue
                    seen.add((cur, nxt))
                    corpus_curs.add(norm(cur))
                    f.write(json.dumps({
                        "cur": cur, "nxt": nxt, "level": lv,
                        "rule": rule,
                        "source": "atom-oneply"}) + "\n")
                    f.flush()
                    counts[lv] += 1
                    rules[rule] += 1
                if (i // WAVE) % 25 == 0:
                    print(f"L{lv} seed {i}: {counts[lv]}/{target} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            print(f"L{lv} DONE: {counts[lv]}/{target} at seed {i}",
                  flush=True)
    print(f"[atoms] total {sum(counts.values())} rows in "
          f"{time.time()-t0:.0f}s -> {OUT}", flush=True)
    print(f"[atoms] per-level {counts}", flush=True)
    print(f"[atoms] rules {dict(rules)}", flush=True)
    print(f"[atoms] skipped {dict(skipped)}", flush=True)


if __name__ == "__main__":
    main()
