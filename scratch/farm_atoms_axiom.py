"""ATOM-DOSE-LADDER-1 farm (pre-reg RESULTS 2026-08-14): rule-tagged
one-ply atoms emitted by the AXIOM engine (axiom-first law; the
sympy farmer scratch/farm_atoms.py is frozen evidence and this is
its bridge-native successor, not a fork of it). Seed band 72M —
band 71M is SPENT by the frozen shard0.

Per root: make_integrate(level, seed) generates the problem
house-side; the forked worker imports the PINNED build-iv7
axiom_sym (INTERFACE_VERSION 7, GIT_SHA asserted 5a8ae70) and runs
emit_chain(root, level, heurisch=<in-process sympy slot>,
deadline_ms=45000); rows with len==1 and source=="axiom-oneply"
are the atoms. Fork-law: the WORKER fork is the timebox (rolling
pool, per-proc DEADLINE), so the heurisch slot runs in-process
inside the worker (no nested fork) with the same elementary-only
language boundary as llmopt/search/axiom_slots.py. sympy stays
oracle-of-record: every kept pair is re-verified house-side with
llmopt.mathgen.problems verify (symbolic equivalence) on top of
axiom's own verify_edge.

Guards carried from the frozen sympy farmer: in_language (tokenizer
roundtrip, ValueError caught), D2 gate-band norm exclusion,
corpus-cur dedup vs the stock corpus, (cur, nxt) dedup. Rows
stream to data/micromodel_atoms_axiom_shard0.jsonl (restart-safe
preload). Nested-dose convention: consumers take the first N rows
after a string-seeded shuffle, so every smaller dose is a subset
of every larger one.

Usage: .venv/bin/python scratch/farm_atoms_axiom.py
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

AXIOM_DIR = "/Users/artin/code/axiom/build-iv7"
AXIOM_SHA = "5a8ae70"
OUT = Path("data/micromodel_atoms_axiom_shard0.jsonl")
SEED_BASE = 72_000_000
TARGETS = {4: 4800, 3: 1800, 5: 1800, 6: 1800, 7: 1800}   # 12,000 total
WAVE = 12          # rolling-pool slots
# Per-level fork walls. L4 heurisch on this band is bimodal
# (~0.3s or live-locked): an 8s wall censors the same hang class
# a 60s wall would, 7x cheaper.
DEADLINE = {3: 60, 4: 8, 5: 60, 6: 60, 7: 60}
POLL = 0.25


def _worker(level: int, seed: int, q) -> None:
    sys.path.insert(0, AXIOM_DIR)
    import axiom_sym as ax
    import sympy as sp
    assert ax.INTERFACE_VERSION == 7 and ax.GIT_SHA.startswith(AXIOM_SHA)
    from llmopt.mathgen.problems import make_integrate
    from llmopt.search.rules import i_heurisch

    _ELEMENTARY = (sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.atan,
                   sp.asin, sp.acos, sp.Abs)

    def heur(node_sstr):
        # in-process slot: the worker fork IS the timebox
        try:
            node = sp.sympify(node_sstr)
            if not isinstance(node, sp.Integral):
                return []
            out = []
            for r in i_heurisch(node):
                if not [f for f in r.atoms(sp.Function)
                        if not isinstance(f, _ELEMENTARY)]:
                    out.append(sp.sstr(r))
            return out
        except Exception:
            return []

    p = make_integrate(level, seed)
    root_sym = sp.Integral(p._expr, sp.Symbol("x"))
    try:
        root = ax.parse_sstr(sp.sstr(root_sym))
    except Exception:
        q.put(None)
        return
    r = ax.emit_chain(root, level, heurisch=heur, deadline_ms=45_000)
    rows = r.get("rows", [])
    if len(rows) != 1 or rows[0].get("source") != "axiom-oneply":
        q.put(None)
        return
    row = rows[0]
    # sympy oracle-of-record on top of verify_edge: the emitted step
    # must be a valid (cur -> nxt) under house symbolic equivalence
    try:
        cur, nxt = sp.sympify(row["cur"]), sp.sympify(row["nxt"])
        if isinstance(cur, sp.Integral):
            delta = sp.simplify(sp.diff(nxt, sp.Symbol("x"))
                                - cur.function)
            if delta != 0 and sp.simplify(delta) != 0:
                q.put(None)
                return
    except Exception:
        q.put(None)
        return
    q.put({"cur": row["cur"], "nxt": row["nxt"], "level": level,
           "source": "axiom-oneply",
           "rule": str(row.get("hints", "")) or "axiom",
           "seed": seed})


def main() -> None:
    os.environ.setdefault("ARM", "off")
    from birth19m_curric import load_excised_rows
    from llmopt.train.mathnative import MathTokenizer
    from tenet_d2_revdiet import gate_band_exprs, norm

    tok = MathTokenizer()
    band = set(gate_band_exprs())
    corpus_curs = {norm(str(r["cur"])) for r in load_excised_rows()}

    def in_language(s: str) -> bool:
        try:
            tok.encode(s)
            return True
        except ValueError:
            return False

    counts: Counter = Counter()
    seen = set()
    if OUT.exists():   # restart-safe preload
        for line in OUT.open():
            r = json.loads(line)
            counts[r["level"]] += 1
            seen.add((r["cur"], r["nxt"]))
    print(f"[axfarm] preload {sum(counts.values())} rows {dict(counts)}",
          flush=True)

    ctx = mp.get_context("fork")
    live = {}          # (level, seed) -> (proc, queue, t0)
    next_seed = {lv: SEED_BASE for lv in TARGETS}
    t_start = time.time()
    kept_total = sum(counts.values())

    def want(lv):
        return counts[lv] < TARGETS[lv]

    def spawn_one():
        for lv in sorted(TARGETS, key=lambda l: counts[l] / TARGETS[l]):
            if want(lv):
                sd = next_seed[lv]
                next_seed[lv] += 1
                q = ctx.Queue()
                pr = ctx.Process(target=_worker, args=(lv, sd, q))
                pr.start()
                live[(lv, sd)] = (pr, q, time.time())
                return True
        return False

    with OUT.open("a") as f:
        while any(want(lv) for lv in TARGETS) or live:
            while len(live) < WAVE and any(want(lv) for lv in TARGETS):
                spawn_one()
            time.sleep(POLL)
            for key in list(live):
                pr, q, t0 = live[key]
                if pr.is_alive() and time.time() - t0 > DEADLINE[key[0]]:
                    pr.kill()
                    pr.join()
                    del live[key]
                    continue
                if pr.is_alive():
                    continue
                pr.join()
                try:
                    row = q.get(timeout=2)
                except Exception:
                    row = None
                del live[key]
                if row is None:
                    continue
                lv = row["level"]
                if not want(lv):
                    continue
                if not (in_language(row["cur"]) and in_language(row["nxt"])):
                    continue
                if norm(row["cur"]) in band or norm(row["nxt"]) in band:
                    continue
                if norm(row["cur"]) in corpus_curs:
                    continue
                if (row["cur"], row["nxt"]) in seen:
                    continue
                seen.add((row["cur"], row["nxt"]))
                counts[lv] += 1
                kept_total += 1
                f.write(json.dumps(row) + "\n")
                f.flush()
                if kept_total % 200 == 0:
                    rate = kept_total / (time.time() - t_start + 1e-9)
                    print(f"[axfarm] {kept_total} kept {dict(counts)} "
                          f"({rate*3600:.0f}/hr)", flush=True)
    print(f"[axfarm] DONE {sum(counts.values())} rows {dict(counts)} "
          f"({time.time()-t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
