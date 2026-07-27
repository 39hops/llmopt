"""Farm verified ALTERNATIVE successors for a sample of corpus
states (the distribution-rows bank, forward edition; motivated by
the 2026-07-26 distribution readout: crystals put ZERO mass on
equally-valid non-canonical moves). For each sampled unique cur,
enumerate successors() (verified, non-identity) and keep children
NOT already in the corpus as a nxt for that cur. Fork workers
stream rows to shard files (killed-worker doctrine: partial
shards survive the wall).

    python scratch/make_altpairs.py [n_states=12000] [workers=6]
Output: data/altpairs_rows.jsonl (merged; {cur, nxt, level, rule})
"""
import json
import multiprocessing as mp
import os
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp

from train_mathnative import load_rows
from llmopt.search.derivation import State, successors

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
W = int(sys.argv[2]) if len(sys.argv) > 2 else 6
WALL = 1800  # whole-farm wall; shards stream so a kill loses nothing
norm = lambda s: s.replace(" ", "")  # noqa: E731

rows = load_rows(gen4=True)
by_cur = {}
lv_of = {}
for r in rows:
    by_cur.setdefault(norm(r["cur"]), set()).add(norm(r["nxt"]))
    lv_of.setdefault(norm(r["cur"]), (r.get("level", 0), r["cur"]))
keys = sorted(by_cur)  # deterministic
random.Random("altpairs-1").shuffle(keys)
keys = keys[:N]
print(f"{len(keys)} unique states sampled of {len(by_cur)}", flush=True)


def worker(idx, my_keys):
    out = open(f"data/altpairs_shard{idx}.jsonl", "w")
    for k in my_keys:
        lv, cur = lv_of[k]
        try:
            for name, ch in successors(State(sp.sympify(cur)),
                                       use_macros=True):
                c = sp.sstr(ch.expr)
                if norm(c) not in by_cur[k]:
                    out.write(json.dumps(
                        {"cur": cur, "nxt": c, "level": lv,
                         "rule": name}) + "\n")
            out.flush()
        except Exception:
            continue
    out.close()


procs = []
for i in range(W):
    p = mp.get_context("fork").Process(
        target=worker, args=(i, keys[i::W]))
    p.start()
    procs.append(p)
for p in procs:
    p.join(WALL)
    if p.is_alive():
        p.kill()
        p.join()
        print("worker killed at wall (partial shard kept)", flush=True)

n = 0
with open("data/altpairs_rows.jsonl", "w") as out:
    for i in range(W):
        f = f"data/altpairs_shard{i}.jsonl"
        if os.path.exists(f):
            for line in open(f):
                out.write(line)
                n += 1
            os.remove(f)
print(f"{n} alt rows -> data/altpairs_rows.jsonl", flush=True)
