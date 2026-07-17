"""v2.2 diet farm — the autopsy-aimed shard (2026-07-17).

Autopsy verdict (RESULTS 2026-07-17): failures are 87% structural,
concentrated at L4 (22.7% valid) and L6 (30.2%) — missing move
vocabulary — plus a drillable L5 class (right ansatz family, wrong
coefficients). Three components, all engine-replayed and
oracle-verified:

1. IN-LANGUAGE L4-L7 chains (min_pairs=2, the round-2 lesson): the
   v2.1 language filter ran post-hoc and discarded 64% of farmed
   chains (Subs/erf routes). Here the filter runs in-stream and
   keeps the in-language PREFIX of each chain — a chain that goes
   out-of-language at step k still teaches steps 1..k-1.
2. One-ply L4 worked examples (riff row 48): the engine one-plies
   77% of L4, and those solutions are worked examples of exactly
   the emit-one-big-correct-step move L4 fails at. Capped ration so
   they can't flood the chain class (round-1 scar).
3. One-ply L5 ansatz pairs (NEW, from the autopsy): L5's failures
   are near-miss coefficients inside the right family — engine
   one-ply solves at L5 are coefficient-determination drills.

Shard by --part/--parts for multi-machine farming (string-seeded,
disjoint by construction).

    caffeinate -i .venv/bin/python -u scripts/farm_v22.py \
        --levels 4,5,6,7 --n 4000 --part 0 --parts 2
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import expert_iter_steps as EIS
from llmopt.train.mathnative import MathTokenizer

SEED_BASE = 67_000_000  # fresh band (66M = 45M GRPO runs)
ONEPLY_CAP_FRAC = 0.35  # one-ply rows per level <= this frac of
# that level's chain rows (round-1 scar: one-ply floods unlearn
# chaining)

_tok = MathTokenizer()


def in_language(text: str) -> bool:
    # roundtrip equality, the farm_algebra pattern — encode silently
    # skips unknown chars, so per-char tests are wrong (word-level
    # vocab: 'Integral' is one token, bare 'e' is nothing)
    return _tok.decode(_tok.encode(text)) == text


def main(levels, n_per, part, parts, out) -> None:
    ctx = mp.get_context("fork")
    path = Path(out)
    seen: set = set()
    chain_rows = {lv: 0 for lv in levels}
    oneply_rows = {lv: 0 for lv in levels}
    n = 0
    with path.open("a") as f:
        for lv in levels:
            for i in range(part, n_per, parts):
                q = ctx.Queue()
                pr = ctx.Process(target=EIS._chain_worker,
                                 args=(lv, SEED_BASE + i, q))
                pr.start()
                pr.join(90)
                if pr.is_alive():
                    pr.kill()
                    pr.join()
                    continue
                try:
                    pairs = q.get(timeout=10)
                except Exception:
                    continue
                if not pairs:
                    continue
                # in-language PREFIX: cut at the first step whose
                # text leaves the 45-token language
                kept = []
                for cur, nxt, hints, think in pairs:
                    if not (in_language(cur) and in_language(nxt)):
                        break
                    kept.append((cur, nxt, hints, think))
                is_oneply = len(kept) == 1
                if is_oneply:
                    # components 2+3: capped worked-example ration
                    if lv not in (4, 5):
                        continue
                    if oneply_rows[lv] >= ONEPLY_CAP_FRAC * max(
                            chain_rows[lv], 100):
                        continue
                elif len(kept) < 2:
                    continue
                for cur, nxt, hints, think in kept:
                    if (cur, nxt) in seen:
                        continue
                    seen.add((cur, nxt))
                    f.write(json.dumps({
                        "cur": cur, "nxt": nxt, "level": lv,
                        "source": "v22-oneply" if is_oneply
                        else "v22-chain",
                        "hints": hints, "think": think}) + "\n")
                    f.flush()
                    n += 1
                    if is_oneply:
                        oneply_rows[lv] += 1
                    else:
                        chain_rows[lv] += 1
                if (i // parts) % 200 == 0:
                    print(f"L{lv} seed {i}: {n} rows "
                          f"(chain {chain_rows[lv]}, "
                          f"oneply {oneply_rows[lv]})", flush=True)
            print(f"L{lv} DONE: chain {chain_rows[lv]}, "
                  f"oneply {oneply_rows[lv]}", flush=True)
    print(f"farm_v22 part {part}/{parts} done: {n} rows -> {path}",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="4,5,6,7")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    levels = tuple(int(x) for x in a.levels.split(","))
    out = a.out or f"data/micromodel_v22_shard{a.part}.jsonl"
    main(levels, a.n, a.part, a.parts, out)
