"""MATH-CYBER-1 PROGRAM-DIET-COVERAGE-0 — census: how much of
theta0's EXACT historical birth diet deterministically relabels
from (cur, nxt) into the fully-qualified v4 ActionProgram under
the pinned engine? Zero model, zero training, zero fresh seeds,
zero search evaluation, zero MAGIC.

POPULATION LAW (abort on any failure): the 12
data/micromodel_chains_shard*.jsonl files + data/step_chains.jsonl;
every file sha256 must equal the theta0 birth manifest
(checkpoints/mathnative_19m_mw1_theta0.json diet_manifest) and
total rows must equal the manifest's 103,595. Source filename
preserved on every row.

PHASE A (anatomy, no world calls): row count, unique exact cur,
unique exact (cur, nxt), duplicate multiplicity, cur with >1
distinct nxt — by source file and level. Policy multimodality is
NOT a label error.

PHASE B (relabel): deduplicate by exact parent; materialize the
deployed house legal set ONCE per parent (default non-macro
successors(), _RULE_CACHE cleared, sympy pinned), reuse it for
every historical nxt of that parent. FROZEN LAWS:
- Parent visible round-trip: sp.sstr(sp.sympify(cur)) == cur, else
  every row of that parent books parent_nonroundtrip (no program).
- Child match: historical nxt == sp.sstr(child.expr) EXACTLY; no
  simplify-based rescue, ever.
- Program derivation: the matched child's ENGINE label names its
  rule; all v4 programs (first-preorder site law, i_parts
  u_choice, accepted-set branch) are derived; classification:
    unique_program      exactly one child match, one program
    ambiguous_program   >1 matching child, or >1 program
    target_mismatch     child matched, no program addressable
    outside_v4_qualified_domain  derivation touches a nested/
                        multi-limit or definite-integral site
                        (vacuous legs of ACTION-SEMANTICS — never
                        counted qualified without their own rung)
    no_engine_edge      no child matches; subtyped:
      constant_offset     sympify(nxt) - some child is a nonzero
                          x-free constant (history-dependent
                          integration constant class)
      equivalent_serialization  sympify(nxt) == some child.expr
                          structurally, strings differ
      rule_gap            legal set empty (dead-end parent)
      unresolved          everything else
    parent_nonroundtrip / censored_load_sensitive (see below)
    unparseable_nxt     nxt fails to sympify (counted, never
                        silently dropped)
- Censoring law: any parent with >= 1 unmatched nxt gets ONE cold
  re-materialization; if the two child-key multisets differ, ALL
  its rows book censored_load_sensitive (timeboxes are
  load-sensitive; idle Mac, single job).
Descriptive equivalence subtypes NEVER count toward primary
coverage. think/level metadata is descriptive anatomy only.

Streaming: one jsonl row per unique parent as results land
(pdc_relabel.jsonl) — a killed run keeps its finished parents.
Parallel: multiprocessing over parents (fork), N_WORKERS
processes, each with its own SIGALRM timeboxes.

SMOKE=1: first 60 unique parents, smoke_* receipt paths, real
paths refused.

Receipts: logs/mathworld1/pdc_anatomy.json, pdc_relabel.jsonl,
pdc_verdict.json (refuse-if-exists).

    .venv/bin/python scratch/mathworld1_pdcov.py              (Mac)
"""
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

import llmopt.search.derivation as derivation  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.search.derivation import State, successors  # noqa: E402
from scratch.mathworld1_actionsem import (RULE_KIND,  # noqa: E402
                                          apply_at, iparts_children,
                                          sites_preorder)

SMOKE = os.environ.get("SMOKE") == "1"
PFX = "smoke_" if SMOKE else ""
DIR = Path("logs/mathworld1")
ANAT = DIR / f"{PFX}pdc_anatomy.json"
RELAB = DIR / f"{PFX}pdc_relabel.jsonl"
VERD = DIR / f"{PFX}pdc_verdict.json"
MANIFEST = "checkpoints/mathnative_19m_mw1_theta0.json"
N_WORKERS = max(2, (os.cpu_count() or 4) - 2)


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def _legal(cur: str):
    """Materialize the deployed legal set for one parent string.
    Returns (list[(label, child_sstr, child_key)], error|None)."""
    try:
        parent = sp.sympify(cur)
        derivation._RULE_CACHE.clear()
        gen = sorted(successors(State(parent)),
                     key=lambda nc: (nc[0], nc[1].key()))
        return parent, [(n, sp.sstr(c.expr), c.key())
                        for n, c in gen], None
    except Exception as ex:  # censored: world raised
        return None, [], repr(ex)[:200]


def _programs_for(parent, rule, child_key):
    """All v4 programs producing child_key under (rule)."""
    kind = RULE_KIND.get(rule)
    if kind is None:
        return [(rule, None, None)], False
    out, outside = [], False
    for i, node in enumerate(sites_preorder(parent, kind)):
        if isinstance(node, sp.Integral):
            if len(node.limits) > 1 or len(node.limits[0]) > 1:
                outside_here = True
            else:
                outside_here = False
        else:
            outside_here = False
        ck, _ = apply_at(parent, rule, node)
        if child_key not in ck:
            continue
        if outside_here:
            outside = True
        if rule == "i_parts":
            uc_map, _ = iparts_children(parent, node)
            for u, k in uc_map.items():
                if k == child_key:
                    out.append((rule, (kind, i), ("u", u)))
        else:
            out.append((rule, (kind, i), None))
    return out, outside


def work(item):
    cur, nxts = item
    t0 = time.monotonic()
    if not _rt_ok(cur):
        return {"cur_sha": sha(cur), "class_all":
                "parent_nonroundtrip", "n_nxt": len(nxts),
                "rows": {n: "parent_nonroundtrip" for n in nxts},
                "wall_s": round(time.monotonic() - t0, 2)}
    parent, legal, err = _legal(cur)
    if err is not None:
        return {"cur_sha": sha(cur),
                "class_all": "censored_load_sensitive",
                "why": err, "n_nxt": len(nxts),
                "rows": {n: "censored_load_sensitive"
                         for n in nxts},
                "wall_s": round(time.monotonic() - t0, 2)}
    bystr = defaultdict(list)
    for name, cs, ck in legal:
        bystr[cs].append((name, ck))
    rows = {}
    any_unmatched = False
    for nxt in nxts:
        hits = bystr.get(nxt, [])
        if not hits:
            any_unmatched = True
            rows[nxt] = _subtype_nomatch(parent, legal, nxt)
            continue
        if len(hits) > 1:
            rows[nxt] = {"class": "ambiguous_program",
                         "why": "multi_child_same_string"}
            continue
        name, ck = hits[0]
        rule = name.split("@", 1)[0] if "@" in name else name
        progs, outside = _programs_for(parent, rule, ck)
        if outside:
            rows[nxt] = {"class": "outside_v4_qualified_domain",
                         "rule": rule}
        elif not progs:
            rows[nxt] = {"class": "target_mismatch", "rule": rule}
        elif len(progs) > 1:
            rows[nxt] = {"class": "ambiguous_program",
                         "rule": rule, "n_programs": len(progs)}
        else:
            r, s, p = progs[0]
            rows[nxt] = {"class": "unique_program", "rule": r,
                         "site": s, "param": p}
    if any_unmatched:
        parent2, legal2, err2 = _legal(cur)
        if err2 is not None or (Counter(k for _, _, k in legal)
                                != Counter(k for _, _, k in legal2)):
            return {"cur_sha": sha(cur),
                    "class_all": "censored_load_sensitive",
                    "why": "remat_divergence", "n_nxt": len(nxts),
                    "rows": {n: "censored_load_sensitive"
                             for n in nxts},
                    "wall_s": round(time.monotonic() - t0, 2)}
    return {"cur_sha": sha(cur), "n_legal": len(legal),
            "rows": rows,
            "wall_s": round(time.monotonic() - t0, 2)}


def _rt_ok(cur: str) -> bool:
    try:
        return sp.sstr(sp.sympify(cur)) == cur
    except Exception:
        return False


def _subtype_nomatch(parent, legal, nxt):
    if not legal:
        return {"class": "no_engine_edge", "sub": "rule_gap"}
    try:
        ne = sp.sympify(nxt)
    except Exception:
        return {"class": "unparseable_nxt"}
    for _, cs, ck in legal:
        try:
            ce = sp.sympify(cs)
            if ne == ce:
                return {"class": "no_engine_edge",
                        "sub": "equivalent_serialization"}
            d = ne - ce
            if not d.free_symbols and d != 0:
                return {"class": "no_engine_edge",
                        "sub": "constant_offset"}
        except Exception:
            continue
    return {"class": "no_engine_edge", "sub": "unresolved"}


def main():
    for p in (ANAT, RELAB, VERD):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    START = start_provenance(
        ["scratch/mathworld1_pdcov.py",
         "scratch/mathworld1_actionsem.py",
         "scratch/mathworld1_srepr_export.py",
         "llmopt/search/derivation.py", "llmopt/search/rules.py"])
    man = json.load(open(MANIFEST))
    dm = man["diet_manifest"]
    rows = []
    for f in sorted(dm):
        h = hashlib.sha256(Path(f).read_bytes()).hexdigest()
        if h != dm[f]["sha256"]:
            raise SystemExit(f"MANIFEST SHA MISMATCH {f}")
        for line in open(f):
            r = json.loads(line)
            r["_src"] = f
            rows.append(r)
    if len(rows) != man["rows"]:
        raise SystemExit(
            f"ROW COUNT {len(rows)} != manifest {man['rows']}")

    # PHASE A — anatomy
    by_cur = defaultdict(list)
    for r in rows:
        by_cur[r["cur"]].append(r)
    pair_count = Counter((r["cur"], r["nxt"]) for r in rows)
    cur_nxts = {c: sorted({r["nxt"] for r in rr})
                for c, rr in by_cur.items()}
    per_file = {}
    for f in sorted(dm):
        fr = [r for r in rows if r["_src"] == f]
        per_file[f] = {
            "rows": len(fr),
            "unique_cur": len({r["cur"] for r in fr}),
            "unique_pairs": len({(r["cur"], r["nxt"])
                                 for r in fr}),
            "multimodal_cur": sum(
                1 for c in {r["cur"] for r in fr}
                if len(cur_nxts[c]) > 1),
            "levels": dict(Counter(r.get("level", "?")
                                   for r in fr))}
    anatomy = {
        "rows": len(rows), "unique_cur": len(by_cur),
        "unique_pairs": len(pair_count),
        "dup_multiplicity_hist": dict(Counter(
            pair_count.values())),
        "cur_with_multi_nxt": sum(1 for c in cur_nxts
                                  if len(cur_nxts[c]) > 1),
        "per_file": per_file,
        "manifest_sha_ok": True,
        "start": START}
    ANAT.write_text(json.dumps(anatomy, indent=1))
    print(f"[pdcov] anatomy: rows={len(rows)} "
          f"unique_cur={len(by_cur)} "
          f"multi_nxt={anatomy['cur_with_multi_nxt']}", flush=True)

    # PHASE B — relabel, streamed
    items = sorted(cur_nxts.items())
    if SMOKE:
        items = items[:60]
    t0 = time.monotonic()
    done = 0
    results = {}
    with RELAB.open("a") as fout, mp.Pool(N_WORKERS) as pool:
        for res in pool.imap_unordered(work, items, chunksize=8):
            results[res["cur_sha"]] = res
            fout.write(json.dumps(res) + "\n")
            fout.flush()
            done += 1
            if done % 1000 == 0 or done == len(items):
                print(f"[pdcov] {done}/{len(items)} parents "
                      f"{round(time.monotonic() - t0, 1)}s",
                      flush=True)

    # aggregate: row-weighted and unique-pair-weighted
    def classify(res, nxt):
        row = res["rows"].get(nxt)
        if isinstance(row, str):
            return row, None
        if row is None:
            return "missing", None
        return row["class"], row.get("sub")

    row_cls = Counter()
    pair_cls = Counter()
    sub_cls = Counter()
    by_src = defaultdict(Counter)
    by_rule = Counter()
    unprod_exposure = Counter()
    for r in rows:
        res = results.get(sha(r["cur"]))
        if res is None:
            if SMOKE:
                continue
            raise SystemExit(f"MISSING PARENT RESULT")
        c, sub = classify(res, r["nxt"])
        row_cls[c] += 1
        by_src[r["_src"]][c] += 1
        if sub:
            sub_cls[sub] += 1
        if c == "unique_program":
            rr = res["rows"][r["nxt"]]
            by_rule[rr["rule"]] += 1
            if rr["rule"] == "i_unprod":
                unprod_exposure["rows"] += 1
    for (c, n), _cnt in pair_count.items():
        res = results.get(sha(c))
        if res is None:
            continue
        cl, _ = classify(res, n)
        pair_cls[cl] += 1

    total_rows = sum(row_cls.values())
    total_pairs = sum(pair_cls.values())
    verdict = {
        "smoke": SMOKE,
        "population_rows": len(rows),
        "aggregated_rows": total_rows,
        "aggregated_pairs": total_pairs,
        "row_weighted_class": dict(row_cls),
        "pair_weighted_class": dict(pair_cls),
        "row_weighted_coverage_unique_program":
            round(row_cls["unique_program"] / max(1, total_rows),
                  4),
        "pair_weighted_coverage_unique_program":
            round(pair_cls["unique_program"]
                  / max(1, total_pairs), 4),
        "no_engine_edge_subtypes": dict(sub_cls),
        "by_source_file": {k: dict(v) for k, v in by_src.items()},
        "recovered_rule_hist": dict(by_rule),
        "i_unprod_exposure": dict(unprod_exposure),
        "n_workers": N_WORKERS,
        "wall_s": round(time.monotonic() - t0, 1),
        "start": START, "completion_commit": completion_commit()}
    VERD.write_text(json.dumps(verdict, indent=1))
    print(json.dumps({k: v for k, v in verdict.items()
                      if k not in ("start", "by_source_file")},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
