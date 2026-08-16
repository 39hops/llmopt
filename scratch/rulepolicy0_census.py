"""RULE-POLICY-0 rung 0 — label-coverage census (/desk shape).

QUESTION: what fraction of the stock diet's chain rows (cur -> nxt)
admit a RECOVERABLE action label — the engine rule whose one-ply
application to cur produces nxt?

THRESHOLDS, named before counting (committed before the run):
  PROMOTE  unique-rule coverage >= 80% of scored rows -> a policy
           head (rule_id | cur) is trainable on the existing diet
           with recovered labels; rung 1 (the head) is priced.
  RESHAPE  50-80% -> labels need a re-emitted diet (the emitter
           writes rule tags at farm time, the atom-shard pattern);
           policy training waits on that farm.
  KILL     < 50% -> the "free labels from the diet" premise dies;
           an action-native policy needs its own labeled corpus
           and the rung is priced accordingly.
Classes per row: unique (exactly one rule reaches nxt), ambiguous
(>= 2 rules reach nxt — a label distribution, not a label),
unreachable (no enumerated successor matches nxt one-ply),
unscored (parse reject / deadline / I-fence).

Method: axiom_sym.successors (AXIOM-FIRST; deadline-walled,
expired taxed) with the house successors fallback, the
enumerate_moves shape copied verbatim from the frozen
scratch/determinability_census.py. Match = normalized
sstr(sympify()) string equality on both sides — deterministic, no
simplify. Sample 4,000 unique (cur, nxt) rows, string-seeded
shuffle, stratified reporting by level. Row GRAIN = one chain
step; LABEL TIMING = the label is recomputed from cur alone at
census time (no future join); SPLIT POLICY = n/a (census, no
training).

    .venv/bin/python -u scratch/rulepolicy0_census.py
"""
import json
import os
import random
import sys
import time
from collections import Counter

os.environ.setdefault("ARM", "off")
os.environ.setdefault("BIRTH_SEED", "3")
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

import sympy as sp  # noqa: E402

from llmopt.search.derivation import State, successors  # noqa: E402

USE_AXIOM = os.environ.get("USE_AXIOM", "1") == "1"
if USE_AXIOM:
    import axiom_sym as ax  # noqa: E402

N_ROWS = int(os.environ.get("N_ROWS", "4000"))
SHUF_SEED = 91_600_000   # fresh band (determinability used 91.3M)
DEADLINE_MS = 10000
OUT = "logs/rulepolicy0/census.jsonl"


def enumerate_moves(cur, expr):
    """-> [(rule_name, child_sstr)]; axiom bridge (deadline-walled,
    expired taxed) with house fallback on parse rejects."""
    if USE_AXIOM:
        try:
            r = ax.successors(ax.parse_sstr(cur), True, DEADLINE_MS)
            if not r["expired"]:
                return [(n, str(c)) for n, c in r["rows"]]
        except Exception:
            pass  # I-fence / parse reject -> house path
    return [(n, sp.sstr(s_.expr)) for n, s_ in successors(State(expr))]


def canon(s):
    return sp.sstr(sp.sympify(s)).replace(" ", "")


def main():
    import birth19m_curric as C
    rows = C.load_excised_rows()
    seen, pool = set(), []
    for r in rows:
        key = (str(r["cur"]).replace(" ", ""),
               str(r["nxt"]).replace(" ", ""))
        if key not in seen:
            seen.add(key)
            pool.append((str(r["cur"]), str(r["nxt"]),
                         int(r.get("level", -1)),
                         str(r.get("think") or "?")))
    random.Random(SHUF_SEED).shuffle(pool)
    print(f"unique (cur,nxt) rows: {len(pool)}; censusing {N_ROWS}",
          flush=True)

    os.makedirs("logs/rulepolicy0", exist_ok=True)
    out = open(OUT, "a")
    cls_all = Counter()
    cls_lv = {}
    t0 = time.time()
    cls_think = {}
    for k, (cur, nxt, lv, think) in enumerate(pool[:N_ROWS]):
        try:
            target = canon(nxt)
            expr = sp.sympify(cur)
            moves = enumerate_moves(cur, expr)
            rules = sorted({n for n, child in moves
                            if canon(child) == target})
            cls = ("unique" if len(rules) == 1 else
                   "ambiguous" if rules else "unreachable")
            rec = {"cur": cur, "nxt": nxt, "level": lv, "cls": cls,
                   "think": think, "rules": rules,
                   "n_succ": len(moves)}
        except Exception as e:
            cls = "unscored"
            rec = {"cur": cur, "nxt": nxt, "level": lv, "cls": cls,
                   "think": think, "err": type(e).__name__}
        cls_all[cls] += 1
        cls_lv.setdefault(lv, Counter())[cls] += 1
        cls_think.setdefault(think, Counter())[cls] += 1
        out.write(json.dumps(rec) + "\n")
        if (k + 1) % 200 == 0:
            out.flush()
            print(f"  {k+1}/{N_ROWS} {dict(cls_all)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    out.close()
    scored = sum(v for c, v in cls_all.items() if c != "unscored")
    print(f"[census] TOTAL {dict(cls_all)} scored={scored}", flush=True)
    for c in ("unique", "ambiguous", "unreachable"):
        print(f"[census] {c}: {cls_all[c]} "
              f"({100*cls_all[c]/max(scored,1):.2f}% of scored)",
              flush=True)
    for lv in sorted(cls_lv):
        d = cls_lv[lv]
        s = sum(v for c, v in d.items() if c != "unscored")
        u = d["unique"]
        print(f"[census] L{lv}: {dict(d)} unique "
              f"{100*u/max(s,1):.1f}%", flush=True)
    for th, d in sorted(cls_think.items(),
                        key=lambda p: -sum(p[1].values())):
        s = sum(v for c, v in d.items() if c != "unscored")
        u = d["unique"]
        print(f"[census] think={th!r}: n={sum(d.values())} unique "
              f"{100*u/max(s,1):.1f}% {dict(d)}", flush=True)
    print(f"[census] wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
