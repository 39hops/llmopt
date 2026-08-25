"""MATH-CYBER-1 ACTION-FINAL-QUAL-0 — the canonical SEMANTIC
ActionProgram (v5-FINAL): qualify i_unprod's desk-selected
term_index parameter and certify that NO opaque child-sort branch
remains anywhere in the model-facing population.

CANONICAL SCHEMA (the one vocabulary for every future session):
    deterministic action:  (rule, site)
    i_parts:               (rule, site, u_choice)
    i_unprod:              (rule, site, term_index)
    site:                  (operator kind, first-preorder unique
                            ordinal)   [ACTION-SEMANTICS law]
term_index = ordinal of the additive term of the (inner)
integrand Add attacked by the frozen i_unprod rule
(I-UNPROD-SEMANTICS-DESK-0's S3, IMPORTED from the desk driver —
law identity by construction, not by copy). Decoders reconstruct
candidates through RULE SEMANTICS (iparts_children /
term-grouped i_unprod traces), never by enumerating children and
selecting on child key/hash; deterministic (rule, site) actions
require exactly one accepted child at the site.

POPULATIONS:
- Leg 1 (regression): the frozen 101-decision / 725-action
  corpus — every previously exact decode must stay exact.
- Leg 2 (material): PDC's matched unique i_unprod edges
  (2,888 parents / 2,894 edges, count-gated).

Per-site parity gate everywhere the i_unprod replica runs (trace
set == frozen rule output, as in the desk). Receipt:
logs/mathworld1/actionfinal_qual.json (refuse-if-exists).
Zero model, zero training, zero fresh seeds, zero search.

    .venv/bin/python scratch/mathworld1_actionfinal.py        (Mac)
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
from llmopt.search.derivation import State, successors  # noqa: E402
from llmopt.search.rules import i_unprod  # noqa: E402
from scratch.mathworld1_actionsem import (RULE_KIND,  # noqa: E402
                                          apply_at, iparts_children,
                                          sites_preorder)
from scratch.mathworld1_srepr_export import srepr_inverse  # noqa: E402
from scratch.mathworld1_unprodsem import trace_unprod  # noqa: E402

OUT = Path("logs/mathworld1/actionfinal_qual.json")


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def unprod_term_children(parent, node):
    """term_index -> set of candidate child KEYS from the frozen
    rule semantics at this site, parity-gated. Returns
    (mapping, parity_ok)."""
    inner = node
    f, xv = inner.function, inner.limits[0][0]
    traces = list(trace_unprod(f, xv))
    seen, emitted = set(), []
    for (i, a_ord, fam, A, resid) in traces:
        k = sp.srepr(A)
        if k in seen:
            continue
        seen.add(k)
        emitted.append((i, A, resid))
    frozen = derivation._timeboxed(i_unprod, node, default=[])
    frozen_keys = {sp.srepr(e) for e in frozen}
    replica_keys = {sp.srepr(
        A + (sp.Integral(r, xv) if r != 0 else 0))
        for (_, A, r) in emitted[:6]}
    parity = replica_keys == frozen_keys
    out = defaultdict(set)
    for (i, A, resid) in emitted[:6]:
        child = A + (sp.Integral(resid, xv) if resid != 0 else 0)
        out[i].add(State(parent.xreplace({node: child})).key())
    return out, parity


def qualify_corpus():
    states = {}
    for l in open("logs/mathworld1/states_srepr.jsonl"):
        r = json.loads(l)
        states[(r["episode_id"], r["step_id"])] = r
    acts = defaultdict(list)
    for l in open("logs/mathworld1/actions_srepr.jsonl"):
        r = json.loads(l)
        acts[(r["episode_id"], r["step_id"])].append(r)
    c = Counter()
    fails = []
    for key in sorted(acts):
        parent = srepr_inverse(states[key]["state_before"])
        if sha(sp.srepr(parent)) != states[key]["state_before_hash"]:
            raise SystemExit(f"BINDING state_hash {key}")
        derivation._RULE_CACHE.clear()
        gen = sorted(successors(State(parent)),
                     key=lambda nc: (nc[0], nc[1].key()))
        if (Counter(sha(x.key()) for _, x in gen)
                != Counter(a["child_hash"] for a in acts[key])):
            raise SystemExit(f"BINDING legal_set {key}")
        accepted = defaultdict(set)
        for name, ch in gen:
            rule = name.split("@", 1)[0] if "@" in name else name
            accepted[rule].add(ch.key())
        seen_programs = set()
        for a in acts[key]:
            rule = a["rule"]
            kind = RULE_KIND[rule]
            # locate site
            if kind is None:
                site, node = None, None
            else:
                hits = []
                for i, cand in enumerate(
                        sites_preorder(parent, kind)):
                    ck, _ = apply_at(parent, rule, cand)
                    if any(sha(k) == a["child_hash"]
                           for k in set(ck) & accepted[rule]):
                        hits.append((i, cand))
                if not hits:
                    c["unaddressable"] += 1
                    fails.append([list(key), rule, "site"])
                    continue
                site, node = hits[0]
            # parameter + DECODE by rule semantics only
            if rule == "i_parts":
                uc_map, _ = iparts_children(parent, node)
                m = [u for u, k in uc_map.items()
                     if sha(k) == a["child_hash"]]
                if len(m) != 1:
                    c["ambiguous"] += 1
                    fails.append([list(key), rule, "u"])
                    continue
                program = (rule, site, ("u", m[0]))
                dk = iparts_children(parent, node)[0].get(m[0])
                decoded = {dk} if dk else set()
            elif rule == "i_unprod":
                tmap, parity = unprod_term_children(parent, node)
                if not parity:
                    c["parity_fail"] += 1
                    fails.append([list(key), rule, "parity"])
                    continue
                m = [t for t, ks in tmap.items()
                     if any(sha(k) == a["child_hash"]
                            for k in ks & accepted[rule])]
                if len(m) != 1:
                    c["ambiguous"] += 1
                    fails.append([list(key), rule, "term"])
                    continue
                program = (rule, site, ("t", m[0]))
                decoded = tmap[m[0]] & accepted[rule]
                if len(decoded) != 1:
                    c["ambiguous"] += 1
                    fails.append([list(key), rule, "term_multi"])
                    continue
            else:
                # deterministic (rule, site): exactly one
                # accepted child at the site (or bare rule)
                if kind is None:
                    dset = accepted[rule]
                else:
                    ck, _ = apply_at(parent, rule, node)
                    dset = set(ck) & accepted[rule]
                if len(dset) != 1:
                    c["ambiguous"] += 1
                    fails.append([list(key), rule, "det"])
                    continue
                program = (rule, site, None)
                decoded = dset
            if program in seen_programs:
                c["collision"] += 1
                fails.append([list(key), rule, "collision"])
                continue
            seen_programs.add(program)
            if decoded and sha(next(iter(decoded))) \
                    == a["child_hash"]:
                c["ok"] += 1
            else:
                c["wrong_child"] += 1
                fails.append([list(key), rule, "wrong"])
    return c, fails


def pdc_work(item):
    cur, nxts = item
    parent = sp.sympify(cur)
    ok = fail = 0
    whys = []
    for node in sorted(parent.atoms(sp.Integral),
                       key=sp.count_ops):
        if len(node.limits) > 1 or len(node.limits[0]) > 1:
            continue
        tmap, parity = unprod_term_children(parent, node)
        if not parity:
            continue
        # visible-string children per term (srepr keys inverted
        # by the qualified adaptive inverse, then printed)
        for nxt in list(nxts):
            hits = [(t, k) for t, ks in tmap.items()
                    for k in ks
                    if sp.sstr(srepr_inverse(k)) == nxt]
            if not hits:
                continue
            terms = sorted({t for t, _ in hits})
            if len(terms) != 1:
                fail += 1
                whys.append("ambiguous_term")
                nxts.remove(nxt)
                continue
            # DECODE from term alone
            dks = tmap[terms[0]]
            dvis = {sp.sstr(srepr_inverse(k)) for k in dks}
            if dvis == {nxt}:
                ok += 1
            else:
                fail += 1
                whys.append("decode_not_unique")
            nxts.remove(nxt)
    for _ in nxts:
        fail += 1
        whys.append("unaddressable")
    return {"ok": ok, "fail": fail, "whys": whys}


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(
        ["scratch/mathworld1_actionfinal.py",
         "scratch/mathworld1_unprodsem.py",
         "scratch/mathworld1_actionsem.py",
         "scratch/mathworld1_srepr_export.py",
         "llmopt/search/derivation.py", "llmopt/search/rules.py"])
    c, fails = qualify_corpus()
    print(f"[final] corpus: {dict(c)}", flush=True)

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
            s = json.loads(l)["cur"]
            h = sha(s)
            if h in recov and h not in cur_of:
                cur_of[h] = s
    n_edges = sum(len(v) for v in recov.values())
    if len(recov) != 2888 or n_edges != 2894:
        raise SystemExit("POPULATION MISMATCH")
    items = [(cur_of[h], list(recov[h])) for h in sorted(recov)]
    ctx = mp.get_context("fork")
    pdc_ok = pdc_fail = 0
    why = Counter()
    with ctx.Pool(9) as pool:
        for res in pool.imap_unordered(pdc_work, items,
                                       chunksize=8):
            pdc_ok += res["ok"]
            pdc_fail += res["fail"]
            for w in res["whys"]:
                why[w] += 1
    verdict = {
        "corpus": dict(c), "corpus_failures": fails[:30],
        "pdc_edges_ok": pdc_ok, "pdc_edges_fail": pdc_fail,
        "pdc_fail_whys": dict(why),
        "pdc_population": {"parents": len(recov),
                           "edges": n_edges},
        "bars": {
            "CORPUS_REGRESSION": c["ok"] == 725
                and sum(v for k, v in c.items() if k != "ok") == 0,
            "PDC_TERMINDEX": pdc_ok == 2894 and pdc_fail == 0,
            # DERIVED: the driver contains no child-key-sorted
            # branch path at all (i_parts decodes by u_choice,
            # i_unprod by term_index, deterministic actions by
            # single-accepted-child assert), so the no-opaque-
            # branch property holds iff both decode legs are
            # fully exact under those semantics.
            "NO_OPAQUE_BRANCH": c["ok"] == 725
                and sum(v for k, v in c.items() if k != "ok") == 0
                and pdc_ok == 2894 and pdc_fail == 0,
        },
        "start": START, "completion_commit": completion_commit()}
    OUT.write_text(json.dumps(verdict, indent=1))
    print(json.dumps({k: v for k, v in verdict.items()
                      if k != "start"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
