"""MATH-CYBER-1 I-UNPROD-SEMANTICS-DESK-0 — can i_unprod's opaque
child-sort branch be replaced by a compact rule-native semantic
parameter (the i_parts.u_choice analogue)? Zero model, zero
training, zero fresh seeds.

POPULATION (fixed, bound to PDC): the unique i_unprod
(cur, nxt) edges classed unique_program by
PROGRAM-DIET-COVERAGE-0 (logs/mathworld1/pdc_relabel.jsonl; cur
recovered from the sha-pinned theta0 diet; expected 2,894 edges
on 2,888 parents per the PDC exposure receipt; abort on
mismatch).

GENERATIVE TRACE (from the frozen i_unprod implementation, rules
L693-756; replicated here with instrumentation and PARITY-GATED
per parent against the frozen rule's own output — any parity
mismatch books that parent noncomparable and is counted):
choices per emitted candidate A:
  term_index  t = f.args[i] of the integrand Add (stored order)
  atom        the sin/cos/exp node fn chosen inside t —
              coordinatized as the ordinal of fn among the
              term's transcendental atoms SORTED BY SREPR (the
              engine iterates a set; the sorted ordinal is the
              deterministic coordinate candidate)
  family      1 = table guess A = cof*H(u); 2 = integrate-
              cofactor guess A = (int cof dx)*fn
Dedup by srepr(A); emission cap 6.

CANDIDATE SCHEMES (registered):
  S1 = (term_index, atom_ordinal, family)
  S2 = (term_index, family)
  S3 = (term_index,) alone
For each scheme, over every populated edge's matched child:
COVERED iff the child's generating trace set maps to exactly one
coordinate under the scheme (multi-trace children take the
lexicographically LOWEST coordinate, censused) and no sibling
child of the same parent shares that coordinate (collision).

STABILITY CENSUS (registered): per parent, pre-dedup candidate
count and pre-cap distinct-A count; any parent where distinct A
candidates exceed the emission cap 6 is a SET-ORDER MEMBERSHIP
HAZARD (which 6 survive could depend on set iteration order) and
is reported as its own class.

Receipt: logs/mathworld1/unprodsem_desk.json (refuse-if-exists).

    .venv/bin/python scratch/mathworld1_unprodsem.py          (Mac)
"""
import glob
import hashlib
import json
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

import llmopt.search.derivation as derivation  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.search.rules import _INT_TABLE, i_unprod  # noqa: E402

OUT = Path("logs/mathworld1/unprodsem_desk.json")


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def trace_unprod(f, x):
    """Instrumented replica of the frozen i_unprod loop: yields
    (term_index, atom_srepr, family, A) for every candidate the
    rule would CONSIDER (pre-dedup, pre-cap), in a deterministic
    order (atoms sorted by srepr, family 1 then 2 per term —
    NOTE: the engine iterates each family's atom SET and dedups
    first-wins; membership parity is what the parity gate
    checks, not order)."""
    if not isinstance(f, sp.Add):
        return
    for i, t in enumerate(f.args):
        atoms = sorted(t.atoms(sp.sin, sp.cos, sp.exp),
                       key=sp.srepr)
        for a_ord, fn in enumerate(atoms):
            v = fn.args[0]
            dv = sp.diff(v, x)
            if dv != 0:
                cof = sp.cancel(t / (dv * fn))
                if not cof.has(sp.sin, sp.cos, sp.exp, sp.log,
                               sp.Integral):
                    A = cof * _INT_TABLE[fn.func](v)
                    resid = sp.expand(f - sp.diff(A, x))
                    if sp.count_ops(resid) < sp.count_ops(f):
                        yield (i, a_ord, 1, A, resid)
        for a_ord, fn in enumerate(atoms):
            cof = sp.cancel(t / fn)
            if (cof.has(sp.sin, sp.cos, sp.exp, sp.log,
                        sp.Integral)
                    or not cof.is_rational_function(x)
                    or cof.is_polynomial(x)):
                continue
            try:
                F = sp.integrate(cof, x)
            except Exception:
                continue
            if F.has(sp.Integral) or not F.has(sp.log):
                continue
            A = F * fn
            resid = sp.expand(f - sp.diff(A, x))
            if sp.count_ops(resid) < sp.count_ops(f):
                yield (i, a_ord, 2, A, resid)


def work(item):
    cur, nxts = item
    parent = sp.sympify(cur)
    # the PDC-matched i_unprod site: find Integral nodes; the
    # rule applies to single-limit inner (nested vacuous in diet)
    row = {"cur_sha": sha(cur), "edges": {}}
    for node in sorted(parent.atoms(sp.Integral), key=sp.count_ops):
        if len(node.limits) > 1 or len(node.limits[0]) > 1:
            continue
        f, xv = node.function, node.limits[0][0]
        traces = list(trace_unprod(f, xv))
        # frozen-rule parity: emitted candidate A-set must equal
        # the frozen rule's output set (dedup + cap applied)
        frozen = derivation._timeboxed(i_unprod, node, default=[])
        frozen_keys = {sp.srepr(e) for e in frozen}
        # replica emission with dedup+cap in trace order
        seen, emitted = set(), []
        for (i, a_ord, fam, A, resid) in traces:
            k = sp.srepr(A)
            if k in seen:
                continue
            seen.add(k)
            emitted.append((i, a_ord, fam, A, resid))
        distinct_A = len(seen)
        cap_hazard = distinct_A > 6
        replica_keys = {sp.srepr(
            A + (sp.Integral(r, xv) if r != 0 else 0))
            for (_, _, _, A, r) in emitted[:6]}
        parity = replica_keys == frozen_keys
        # per historical child: which traces generate it
        for nxt in nxts:
            child_map = defaultdict(list)
            for (i, a_ord, fam, A, resid) in emitted:
                child = A + (sp.Integral(resid, xv)
                             if resid != 0 else 0)
                new = sp.sstr(parent.xreplace({node: child}))
                child_map[new].append((i, a_ord, fam))
            if nxt in child_map:
                row["edges"][nxt] = {
                    "traces": child_map[nxt],
                    "siblings": {k: v for k, v in
                                 child_map.items() if k != nxt},
                    "parity": parity,
                    "distinct_A": distinct_A,
                    "cap_hazard": cap_hazard,
                    "n_frozen": len(frozen_keys)}
    return row


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(
        ["scratch/mathworld1_unprodsem.py",
         "llmopt/search/rules.py",
         "llmopt/search/derivation.py"])
    recov = defaultdict(list)
    for l in open("logs/mathworld1/pdc_relabel.jsonl"):
        r = json.loads(l)
        for nxt, cl in r.get("rows", {}).items():
            if (isinstance(cl, dict)
                    and cl.get("class") == "unique_program"
                    and cl.get("rule") == "i_unprod"):
                recov[r["cur_sha"]].append(nxt)
    files = sorted(glob.glob("data/micromodel_chains_shard*.jsonl"))
    files.append("data/step_chains.jsonl")
    cur_of = {}
    for f in files:
        for l in open(f):
            c = json.loads(l)["cur"]
            h = sha(c)
            if h in recov and h not in cur_of:
                cur_of[h] = c
    n_edges = sum(len(v) for v in recov.values())
    if len(recov) != 2888 or n_edges != 2894:
        raise SystemExit(f"POPULATION MISMATCH: parents="
                         f"{len(recov)} edges={n_edges}")
    items = [(cur_of[h], recov[h]) for h in sorted(recov)]
    ctx = mp.get_context("fork")
    results = []
    with ctx.Pool(9) as pool:
        for res in pool.imap_unordered(work, items, chunksize=8):
            results.append(res)

    schemes = {"S1": lambda t: (t[0], t[1], t[2]),
               "S2": lambda t: (t[0], t[2]),
               "S3": lambda t: (t[0],)}
    cover = {s: 0 for s in schemes}
    collide = {s: 0 for s in schemes}
    multitrace = 0
    parity_fail = 0
    cap_hazards = 0
    matched_edges = 0
    branch_exposed = 0
    dist_hist = Counter()
    for res in results:
        for nxt, e in res["edges"].items():
            matched_edges += 1
            if not e["parity"]:
                parity_fail += 1
                continue
            if e["cap_hazard"]:
                cap_hazards += 1
            dist_hist[e["distinct_A"]] += 1
            if e["n_frozen"] > 1:
                branch_exposed += 1
            if len(e["traces"]) > 1:
                multitrace += 1
            for s, fn in schemes.items():
                mine = sorted(fn(tuple(t)) for t in e["traces"])
                sib = [sorted(fn(tuple(t)) for t in v)
                       for v in e["siblings"].values()]
                coord = mine[0]
                if any(coord in sb for sb in sib):
                    collide[s] += 1
                else:
                    cover[s] += 1
    verdict = {
        "population_parents": len(recov),
        "population_edges": n_edges,
        "matched_edges": matched_edges,
        "parity_fail_edges": parity_fail,
        "branch_exposed_edges_n_frozen_gt1": branch_exposed,
        "multitrace_children": multitrace,
        "cap_hazard_edges_distinctA_gt6": cap_hazards,
        "distinct_A_hist": dict(dist_hist),
        "scheme_cover": cover, "scheme_collide": collide,
        "start": START, "completion_commit": completion_commit()}
    OUT.write_text(json.dumps(verdict, indent=1))
    print(json.dumps({k: v for k, v in verdict.items()
                      if k != "start"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
