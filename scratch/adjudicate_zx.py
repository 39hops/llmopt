"""Adjudicate axiom's ZX sample batch (relay 2026-07-26-0 protocol).

Two independent legs per row:
  SEMANTIC — parse cur and nxt into pyzx, compare_tensors (catches
  any unsound rewrite, implementation-independent). Timeboxed.
  STRUCTURAL — independent Python replay of the named move at the
  named site on the parsed labeled graph; labeled-graph equality
  vs their nxt (catches site/serialization drift even when
  semantics hold).

    .venv/bin/python scratch/adjudicate_zx.py <file> [max_rows]
"""
import json
import multiprocessing as mp
import re
import sys
from fractions import Fraction

import pyzx as zx


def parse(s: str):
    """-> (ins, outs, spiders {label: (color, phase8)}, edges
    {(a,b): 'P'|'H'} with a<b)."""
    ins = [int(x) for x in re.search(r"in\(([\d,]+)\)", s).group(1).split(",")]
    outs = [int(x) for x in re.search(r"out\(([\d,]+)\)", s).group(1).split(",")]
    spiders = {}
    for m in re.finditer(r"([ZX])\((\d+):(\d+)\)", s):
        spiders[int(m.group(2))] = (m.group(1), int(m.group(3)))
    edges = {}
    for m in re.finditer(r"([PH])\((\d+)-(\d+)\)", s):
        a, b = sorted((int(m.group(2)), int(m.group(3))))
        edges[(a, b)] = m.group(1)
    return ins, outs, spiders, edges


def to_pyzx(ins, outs, spiders, edges):
    g = zx.Graph()
    vmap = {}
    for q, l in enumerate(ins):
        vmap[l] = g.add_vertex(zx.VertexType.BOUNDARY, qubit=q, row=0)
    for q, l in enumerate(outs):
        vmap[l] = g.add_vertex(zx.VertexType.BOUNDARY, qubit=q, row=9)
    for l, (c, p) in spiders.items():
        vt = zx.VertexType.Z if c == "Z" else zx.VertexType.X
        vmap[l] = g.add_vertex(vt, phase=Fraction(p, 4), row=1)
    for (a, b), t in edges.items():
        et = (zx.EdgeType.SIMPLE if t == "P" else zx.EdgeType.HADAMARD)
        g.add_edge((vmap[a], vmap[b]), et)
    g.set_inputs([vmap[l] for l in ins])
    g.set_outputs([vmap[l] for l in outs])
    return g


def _sem_worker(cur, nxt, q):
    try:
        ok = zx.compare_tensors(to_pyzx(*parse(cur)), to_pyzx(*parse(nxt)))
        q.put(bool(ok))
    except Exception as e:
        q.put(f"ERR:{type(e).__name__}")


def semantic(cur, nxt, wall=30):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    pr = ctx.Process(target=_sem_worker, args=(cur, nxt, q))
    pr.start()
    pr.join(wall)
    if pr.is_alive():
        pr.kill()
        pr.join()
        return "WALL"
    return q.get() if not q.empty() else "DIED"


def replay_fuse(ins, outs, spiders, edges, site):
    a, b = (int(x) for x in site.split())
    if spiders[a][0] != spiders[b][0] or edges.get(tuple(sorted((a, b)))) != "P":
        return None
    keep, gone = a, b
    # per the emitter: surviving label = the one appearing in nxt;
    # try both, caller compares both orientations
    out_sp = dict(spiders)
    out_sp[keep] = (spiders[a][0], (spiders[a][1] + spiders[b][1]) % 8)
    del out_sp[gone]
    out_e = {}
    for (x, y), t in edges.items():
        if {x, y} == {a, b}:
            continue
        x2 = keep if x == gone else x
        y2 = keep if y == gone else y
        if x2 == y2:
            return None  # would self-loop: their checks refuse these
        k = tuple(sorted((x2, y2)))
        if k in out_e and out_e[k] != t:
            return None  # parallel-edge conflict: refused configs
        out_e[k] = t
    return out_sp, out_e


def structural(row):
    ins, outs, spiders, edges = parse(row["cur"])
    n_ins, n_outs, n_spiders, n_edges = parse(row["nxt"])
    if row["kind"] != "fuse":
        return "SKIP"  # v1 replays the dominant kind; rare kinds
        # ride the semantic leg + eyeball audit
    for site in (row["site"], " ".join(row["site"].split()[::-1])):
        r = replay_fuse(ins, outs, spiders, edges, site)
        if r and r[0] == n_spiders and r[1] == n_edges:
            return True
    return False


def main():
    path = sys.argv[1]
    n_sem = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    rows = [json.loads(l) for l in open(path)]
    bad = []
    # LEG 1: structural replay, ALL rows (pure python, no forks)
    st = {"True": 0, "False": 0, "SKIP": 0}
    for i, r in enumerate(rows):
        t = structural(r)
        st[str(t)] += 1
        if t is False:
            bad.append(("STRUCT", i, r["kind"], r.get("seed"), r.get("n")))
    print(f"STRUCTURAL (all {len(rows)}): {st}", flush=True)
    # LEG 2: semantic on small-diagram subsample (treewidth wall —
    # the ZX chapter's densification scar; wall 10s, spiders <= 10)
    import random
    small = [r for r in rows if r.get("spiders", 99) <= 10]
    random.Random("zx-adj-0").shuffle(small)
    small = small[:n_sem]
    # rare kinds ALWAYS ride the semantic leg (all of them)
    rare = [r for r in rows if r["kind"] != "fuse"]
    pool = {id(r): r for r in (small + rare)}.values()
    sem = {"True": 0, "False": 0, "WALL": 0, "ERR": 0, "DIED": 0}
    for i, r in enumerate(pool):
        s = semantic(r["cur"], r["nxt"], wall=10)
        key = str(s) if str(s) in sem else "ERR"
        sem[key] += 1
        if s is False:
            bad.append(("SEM", i, r["kind"], r.get("seed"), r.get("n")))
        if (i + 1) % 50 == 0:
            print(f"  sem {i+1}/{len(pool)} {sem}", flush=True)
    print(f"SEMANTIC ({len(pool)} rows: {len(small)} small + "
          f"{len(rare)} rare-kind): {sem}", flush=True)
    for b in bad[:20]:
        print("BAD", b, flush=True)


if __name__ == "__main__":
    main()
