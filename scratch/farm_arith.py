"""BASICS-DIET-1 farm: the one-ply ARITHMETIC shard (pre-reg RESULTS
2026-08-15, L30355). Sibling of the frozen scripts/farm_algebra.py —
never an edit to it.

Rescoped by BASICS-CENSUS-0: the diet already states algebra on
29,988 rows (18.2%); arithmetic appears on 12 incidental rows
(0.01%), none a division. These are the numeric steps a calculus
chain actually performs — coefficient products, exact division,
fraction reduction, small squares, additive chains.

Exact by construction: every row's nxt is computed with integer /
Rational arithmetic, so equality holds without a solver. No
simplify(), no engine, no sympy timebox exposure at all.

Guards at farm time (the house set): stable STRING seeds
("arith-v1-<family>-<i>"), D2 gate-band exclusion on
norm(cur)/norm(nxt), corpus-cur dedup against the excised stock
diet (no new one-to-many ambiguity), (cur,nxt) dedup within the
shard, tokenizer roundtrip on both sides. Rows stream incrementally.

    .venv/bin/python -u scratch/farm_arith.py
"""
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("ARM", "off")
os.environ.setdefault("BIRTH_SEED", "3")
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

import sympy as sp  # noqa: E402

OUT = Path("data/micromodel_arith_shard0.jsonl")
TARGETS = {"mul": 700, "div": 700, "divred": 600, "pow": 14,
           "numsum": 700}          # ~2,700 rows, ~1.6% of the diet
SEED_CAP = 200_000                 # per family, honest-starvation stop

# Operand ranges are WIDER than scratch/basics_probe0.py's on purpose:
# the probe's own space is small enough to exhaust (242 distinct mul
# pairs at its range), so training rows and probe items would collide.
# Every probe prompt is additionally excluded by set below — the house
# exclude=-guard, never seed offsets alone.


def gen(family, i):
    """One row: (cur, nxt, level, think). Exact by construction."""
    rng = random.Random(f"arith-v1-{family}-{i}")
    if family == "mul":
        a = rng.randint(2, 24) * rng.choice([1, -1])
        b = rng.randint(2, 24)
        return f"{a}*{b}", sp.sstr(sp.Integer(a * b)), 1, "multiply"
    if family == "div":
        b = rng.randint(2, 15)
        q = rng.randint(2, 24) * rng.choice([1, -1])
        return f"{b * q}/{b}", sp.sstr(sp.Integer(q)), 1, "divide"
    if family == "divred":
        a = rng.randint(2, 60) * rng.choice([1, -1])
        b = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 12])
        return f"{a}/{b}", sp.sstr(sp.Rational(a, b)), 2, "reduce"
    if family == "pow":
        a = rng.randint(2, 15)
        return f"{a}**2", sp.sstr(sp.Integer(a * a)), 1, "square"
    terms = [rng.randint(-9, 9) for _ in range(rng.randint(3, 6))]
    s = ""
    for k, t in enumerate(terms):
        if k == 0:
            s = str(t)
        else:
            s += (f" + {t}" if t >= 0 else f" - {-t}")
    return s, sp.sstr(sp.Integer(sum(terms))), 1, "add the numbers"


def main():
    import birth19m_curric as C
    from llmopt.train.mathnative import MathTokenizer
    from tenet_d2_revdiet import gate_band_exprs, norm

    import basics_probe0 as P

    tok = MathTokenizer()
    band = set(gate_band_exprs())
    corpus_cur = {norm(str(r["cur"])) for r in C.load_excised_rows()}
    # exclude= guard: every held-out probe prompt, by normalized cur,
    # across all three probe arms (the split guard the house requires;
    # a seed band alone does not separate small generator spaces).
    probe_cur = set()
    for arm, gfn in P.ARMS.items():
        for i in range(max(P.N, 1000)):
            probe_cur.add(norm(str(gfn(i)[0])))
    print(f"[farm] band {len(band)} exprs, corpus curs "
          f"{len(corpus_cur)}, probe curs excluded {len(probe_cur)}",
          flush=True)

    assert not OUT.exists(), f"refuse to overwrite {OUT}"
    seen, kept = set(), Counter()
    drops = Counter()
    with OUT.open("w") as fh:
        for family, target in TARGETS.items():
            i = 0
            while kept[family] < target and i < SEED_CAP:
                i += 1
                cur, nxt, level, think = gen(family, i)
                ncur, nnxt = norm(cur), norm(nxt)
                if ncur in band or nnxt in band:
                    drops["band"] += 1
                    continue
                if ncur in corpus_cur:
                    drops["corpus_cur"] += 1
                    continue
                if ncur in probe_cur:
                    drops["probe_holdout"] += 1
                    continue
                if (ncur, nnxt) in seen:
                    drops["dup"] += 1
                    continue
                try:
                    tok.encode(f"Current: {cur}\nHints: none\n"
                               f"Step: {nxt}\n")
                except ValueError:
                    drops["unencodable"] += 1
                    continue
                seen.add((ncur, nnxt))
                kept[family] += 1
                fh.write(json.dumps({
                    "cur": cur, "nxt": nxt, "level": level,
                    "source": "arith-oneply", "rule": f"a_{family}",
                    "hints": [], "think": think}) + "\n")
                fh.flush()
            print(f"[farm] {family}: kept {kept[family]}/{target} "
                  f"after {i} seeds", flush=True)
    print(f"[farm] total {sum(kept.values())} rows -> {OUT}")
    print(f"[farm] drops {dict(drops)}")


if __name__ == "__main__":
    main()
