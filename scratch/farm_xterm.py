"""XTERM-DIET-1 farm: the cross-term-decomposed expand shard
(pre-reg RESULTS 2026-08-16, L30565). Sibling of the frozen
scratch/farm_arith.py — never an edit to it.

Two families, both in the ALGEBRA format, exact by construction:
  xexp   (a*x+b)*(c*x+d) -> A*x**2 + (a*d + b*c)*x + C
         ends evaluated, cross term stated as literal products —
         the decomposition row.
  xstep  that intermediate -> the fully evaluated polynomial —
         the arithmetic performed INSIDE the polynomial format.

Registered ambiguity (in the pre-reg, not discovered here): xexp
gives the product shape a second valid continuation next to the
diet's product->final rows. That one-to-many is the mechanism
under test.

Guards at farm time (the house set): stable STRING seeds
("xterm-v1-<family>-<i>"), D2 gate-band exclusion, corpus-cur
dedup against the excised stock diet, probe-cur exclude= across
ALL xterm_probe arms, (cur,nxt) dedup, tokenizer roundtrip on both
sides, and a per-row construction check
sympify(nxt) == expand(sympify(cur)). Rows stream incrementally.

    .venv/bin/python -u scratch/farm_xterm.py
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

OUT = Path("data/micromodel_xterm_shard0.jsonl")
TARGETS = {"xexp": 1300, "xstep": 1300}    # ~2,600 rows, ~1.6% dose
SEED_CAP = 200_000
x = sp.Symbol("x")


def _parts(rng):
    a = rng.randint(2, 24) * rng.choice([1, -1])
    b = rng.randint(1, 24) * rng.choice([1, -1])
    c = rng.randint(2, 24) * rng.choice([1, -1])
    d = rng.randint(1, 24) * rng.choice([1, -1])
    return a, b, c, d


def _inter(a, b, c, d):
    """Intermediate text: ends evaluated, cross term literal."""
    A, Cc = a * c, b * d
    head = f"{A}*x**2" if A != 1 else "x**2"
    return f"{head} + ({a}*{d} + {b}*{c})*x + {Cc}"


def gen(family, i):
    """One row: (cur, nxt, level, think). Exact by construction."""
    rng = random.Random(f"xterm-v1-{family}-{i}")
    a, b, c, d = _parts(rng)
    prod = sp.Mul(a * x + b, c * x + d, evaluate=False)
    inter = _inter(a, b, c, d)
    if family == "xexp":
        return sp.sstr(prod), inter, 2, "expand keeping the cross term"
    final = sp.sstr(sp.expand(prod))
    return inter, final, 2, "evaluate the cross term"


def main():
    import birth19m_curric as C
    from llmopt.train.mathnative import MathTokenizer
    from tenet_d2_revdiet import gate_band_exprs, norm

    import xterm_probe as P

    tok = MathTokenizer()
    band = set(gate_band_exprs())
    corpus_cur = {norm(str(r["cur"])) for r in C.load_excised_rows()}
    # exclude= guard: every held-out probe prompt, by normalized cur,
    # across all probe arms; a seed band alone never separates a
    # small generator space.
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
                    tok.encode(cur)
                    tok.encode(nxt)
                except ValueError:
                    drops["tok"] += 1
                    continue
                if sp.expand(sp.sympify(cur) - sp.sympify(nxt)) != 0:
                    drops["construct"] += 1
                    continue
                seen.add((ncur, nnxt))
                kept[family] += 1
                fh.write(json.dumps(
                    {"cur": cur, "nxt": nxt, "level": level,
                     "think": think, "source": "xterm-oneply",
                     "family": family,
                     "seed": f"xterm-v1-{family}-{i}"}) + "\n")
                fh.flush()
            print(f"[farm] {family}: kept {kept[family]}/{target} "
                  f"after {i} seeds", flush=True)
    print(f"[farm] total {sum(kept.values())} rows -> {OUT}; "
          f"drops {dict(drops)}", flush=True)
    # split-policy receipt: shard-cur INTERSECT probe-cur must be 0
    shard_cur = {norm(json.loads(line)["cur"]) for line in OUT.open()}
    inter = shard_cur & probe_cur
    print(f"[farm] probe-cur INTERSECT shard-cur = {len(inter)}",
          flush=True)
    assert not inter, sorted(inter)[:5]


if __name__ == "__main__":
    main()
