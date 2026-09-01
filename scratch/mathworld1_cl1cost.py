"""MATH-CYBER-1 CLOSED-LOOP-1 SEARCH-COST-PROXY-DESK-0 —
execute the frozen development desk (prereg
4dab4fc2d7cf00248d9fa59c824d1aa2526247d6, RESULTS L60564).

DEVELOPMENT ONLY on the known 96-root population: depth-0
reranking of the sealed seed-19001 CANONICAL candidate scores
under three frozen string proxies:
  P1 log((1+len(child))/(1+len(parent)))         [sstr chars]
  P2 log((1+terms(child))/(1+terms(parent)))     [depth-0
     additive separators; '-' binary iff a preceding
     non-space char exists and is not in '(',',','+','-',
     '*','/'; position-0 '-' unary]
  P3 log((1+ic(child))/(1+ic(parent)))           [count of
     "Integral(" substring]
Normalization q_j = 90th pct of |C_j| over the deduplicated
677-candidate universe (95 completely-scored roots); grid
lambda in {0,.25,.5,1,2,3,4,6,8,12,16}; rerank with the
standing tie law (score tie -> smallest factor code -> engine
order). Selection (frozen, lexicographic): smallest lambda
with FOCUS_DEMOTION 4/4 on {54,68,78,88} (strictly-lower-cost
replacement; equal-cost = non-demotion, named) AND root-23
top-1 preserved; then fewest collateral top-1 changes /91,
then smaller lambda, then P1<P2<P3. RAW FIRST: the complete
reranking artifact is hashed BEFORE selection. P4 exact child
branching + hce sweep runs AFTER selection is immutable
(fork-per-root, drain-before-join, per-root timeout;
diagnostic/descriptive only). No rollout, no model forward,
no solve-rate claim.

Outputs under logs/mathworld1/cl1/costdesk/ (refuse-if-
exists): rerank.jsonl, sweep.jsonl, cl1cost_receipt.json,
riders.json.

    .venv/bin/python scratch/mathworld1_cl1cost.py           (Mac)
"""
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from scratch.mathworld1_cl1run import (MANIFEST_SHA,  # noqa: E402
                                       POPDIR, fsha)
from scratch.mathworld1_svpbirth import gate  # noqa: E402

PREREG_COMMIT = "4dab4fc2d7cf00248d9fa59c824d1aa2526247d6"
RAW_CL1 = Path("logs/mathworld1/cl1/run/trajectories.jsonl")
RAW_CL1_SHA = ("22198ddf8d1e857a55bf98293d7aa101a5b0c29f48f474"
               "d30999ef4e872c1309")
OUTDIR = Path("logs/mathworld1/cl1/costdesk")
FOCUS = [54, 68, 78, 88]
CONTROL = 23
RIDER = 94
GRID = [0, 0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16]
PROXIES = ["P1", "P2", "P3"]
SWEEP_ROOT_TIMEOUT_S = 1800.0


def terms(s: str) -> int:
    """Frozen parser-free depth-0 additive-term count."""
    depth = 0
    seps = 0
    prev_nonspace = None
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0:
            if ch == "+":
                seps += 1
            elif ch == "-":
                binary = (prev_nonspace is not None
                          and prev_nonspace not in
                          ("(", ",", "+", "-", "*", "/"))
                if binary:
                    seps += 1
        if ch != " ":
            prev_nonspace = ch
    gate(depth == 0, f"UNBALANCED PARENS: {s[:60]}")
    return seps + 1


def ic(s: str) -> int:
    return s.count("Integral(")


def proxies(parent: str, child: str):
    c1 = math.log((1 + len(child)) / (1 + len(parent)))
    c2 = math.log((1 + terms(child)) / (1 + terms(parent)))
    c3 = math.log((1 + ic(child)) / (1 + ic(parent)))
    return c1, c2, c3


def rerank_top1(cands, key_score):
    """Standing tie law: max score, ties -> smallest factor
    code, then engine (list) order."""
    best = max(key_score(c) for c in cands)
    tied = [i for i, c in enumerate(cands)
            if key_score(c) == best]
    return min(tied, key=lambda i: (cands[i]["factor_code"], i))


def _sweep_worker(level, seed, root_cur, want_children, q):
    """Fork target: rebuild root, enumerate its stable legal
    set, then per child measure exact stable legal-set size
    and hce. Streams one row per child through the queue."""
    import sympy as sp
    from llmopt.mathgen.problems import make_integrate
    from llmopt.search.derivation import State, hce
    from scratch.mathworld1_svpeval import stable_legal_set
    X = sp.Symbol("x")
    p = make_integrate(level, seed)
    root = sp.Integral(p._expr, X)
    if sp.sstr(root) != root_cur:
        q.put({"error": "root_bytes"})
        return
    acts, stable = stable_legal_set(State(root))
    fresh = {}
    for n, c in acts:
        fresh[(n, sp.sstr(c.expr))] = c
    q.put({"meta": True, "fresh_n": len(acts),
           "stable": bool(stable)})
    for pos, (name, child_sstr) in enumerate(want_children):
        st = fresh.get((name, child_sstr))
        if st is None:
            q.put({"pos": pos, "missing": True})
            continue
        t0 = time.monotonic()
        cacts, cstable = stable_legal_set(st)
        h = float(hce(st))
        q.put({"pos": pos, "n_child_legal": len(cacts),
               "child_stable": bool(cstable), "hce": h,
               "wall_s": round(time.monotonic() - t0, 3)})
    q.put({"done": True})


def sweep_root(row, want_children):
    """Fork-isolated per-root sweep; drain queue with timeout
    while alive (streaming, no large-payload join deadlock)."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(target=_sweep_worker,
                       args=(row["level"],
                             row["generator_seed"],
                             row["root_cur"], want_children, q))
    proc.start()
    rows = []
    t0 = time.monotonic()
    done = False
    while time.monotonic() - t0 < SWEEP_ROOT_TIMEOUT_S:
        try:
            r = q.get(timeout=5.0)
        except Exception:
            if not proc.is_alive():
                break
            continue
        if r.get("done"):
            done = True
            break
        rows.append(r)
    if proc.is_alive():
        proc.kill()
    proc.join()
    return rows, done


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    gate(fsha(RAW_CL1) == RAW_CL1_SHA, "CL1 RAW PIN")
    gate(fsha(POPDIR / "manifest.jsonl") == MANIFEST_SHA,
         "MANIFEST PIN")
    START = start_provenance(
        ["scratch/mathworld1_cl1cost.py",
         "scratch/mathworld1_cl1run.py",
         "scratch/mathworld1_cl1pop.py",
         "scratch/mathworld1_svpeval.py",
         "scratch/mathworld1_svpbirth.py",
         "llmopt/search/derivation.py", "llmopt/search/rules.py",
         "llmopt/lab/provenance.py"])
    manifest = {r["row_index"]: r for r in
                (json.loads(l)
                 for l in open(POPDIR / "manifest.jsonl"))}

    # ---- load + POSITIONAL PREFIX join ----
    roots = {}
    for l in open(RAW_CL1):
        t = json.loads(l)
        if t["arm"] != "B":
            continue
        d0 = t["decisions"][0]
        legal = d0["legal"]
        cands = d0.get("candidates", [])
        gate(len(cands) <= len(legal), "CAND>LEGAL")
        for i, c in enumerate(cands):
            gate(c["name"] == legal[i]["name"],
                 f"PREFIX MISMATCH root {t['row_index']} {i}")
        roots[t["row_index"]] = {
            "parent": d0["parent"], "legal": legal,
            "cands": cands,
            "chosen_index": d0.get("chosen_index")}
    gate(len(roots) == 96, "ROOT COUNT")
    scored = {}
    for i, r in roots.items():
        cl = []
        complete = (len(r["cands"]) == len(r["legal"])
                    and all(c.get("score") is not None
                            for c in r["cands"]))
        for pos, lg in enumerate(r["legal"]):
            c = (r["cands"][pos] if pos < len(r["cands"])
                 else {})
            cl.append({"pos": pos, "name": lg["name"],
                       "child_sstr": lg["child_sstr"],
                       "factor_code": c.get("factor_code"),
                       "score": c.get("score")})
        r["joined"] = cl
        r["complete"] = complete
        if complete:
            scored[i] = r
    gate(len(scored) == 95 and 93 not in scored,
         "SCORED UNIVERSE")
    gate(len(roots[93]["legal"]) == 10
         and len(roots[93]["cands"]) == 9, "ROOT93 PREFIX")

    # ---- qualification: structure + proxies + finiteness ----
    for i, r in roots.items():
        gate(terms(r["parent"]) == 1, f"PARENT TERMS {i}")
        gate(ic(r["parent"]) == 1, f"PARENT IC {i}")
    n_univ = sum(len(r["legal"]) for r in scored.values())
    gate(n_univ == 677, "677 UNIVERSE")
    t0 = time.perf_counter_ns()
    per_cand_ns = []
    for i, r in roots.items():
        for e in r["joined"]:
            tc = time.perf_counter_ns()
            c1, c2, c3 = proxies(r["parent"], e["child_sstr"])
            per_cand_ns.append(time.perf_counter_ns() - tc)
            gate(all(math.isfinite(v) for v in (c1, c2, c3)),
                 "NON-FINITE PROXY")
            e["C1"], e["C2"], e["C3"] = c1, c2, c3
    timing = {
        "n_candidates": len(per_cand_ns),
        "total_ms": round((time.perf_counter_ns() - t0) / 1e6,
                          3),
        "mean_us_per_candidate": round(
            sum(per_cand_ns) / len(per_cand_ns) / 1e3, 2),
        "median_us_per_candidate": round(
            sorted(per_cand_ns)[len(per_cand_ns) // 2] / 1e3,
            2),
        "max_us_per_candidate": round(
            max(per_cand_ns) / 1e3, 2),
        "standing_cost_comparison": {
            "booked_depth0_enum_median_s": 4.07,
            "booked_model_scoring_wall_s_per_call":
                round(97.0 / 140, 3),
            "note": "reference figures from the booked "
                    "CLOSED-LOOP-1/splice receipts (L59529/"
                    "L60177 lineage): depth-0 stable-set "
                    "enumeration median and B model wall "
                    "97.0 s over 140 calls"}}

    # baseline==booked gate BEFORE namespace creation (a gate
    # failure must not brick the retry)
    baseline = {i: rerank_top1(scored[i]["joined"],
                               lambda c: c["score"])
                for i in scored}
    for i in scored:
        gate(baseline[i] == scored[i]["chosen_index"],
             f"BASELINE!=BOOKED root {i}")

    # ---- normalization over the deduplicated 677 universe ----
    univ = [e for i, r in scored.items() for e in r["joined"]]
    q_j = {}
    for j, key in (("P1", "C1"), ("P2", "C2"), ("P3", "C3")):
        q = float(np.percentile([abs(e[key]) for e in univ],
                                90))
        gate(q > 0, f"QJ ZERO {j}")
        q_j[j] = q

    # ---- full grid rerank; RAW artifact before selection ----
    OUTDIR.mkdir(parents=True)
    top1 = {}   # (proxy, lambda, root) -> position
    with open(OUTDIR / "rerank.jsonl", "w") as f:
        for i in sorted(scored):
            r = scored[i]
            base = rerank_top1(r["joined"],
                               lambda c: c["score"])
            row = {"row_index": i, "parent": r["parent"],
                   "baseline_top1_pos": base,
                   "booked_chosen_index": r["chosen_index"],
                   "candidates": [
                       {k: e[k] for k in
                        ("pos", "name", "factor_code",
                         "score", "C1", "C2", "C3")}
                       for e in r["joined"]],
                   "q_j": q_j, "adjusted_top1": {}}
            for j, key in (("P1", "C1"), ("P2", "C2"),
                           ("P3", "C3")):
                for lam in GRID:
                    tp = rerank_top1(
                        r["joined"],
                        lambda c, j=j, key=key, lam=lam:
                        c["score"]
                        - lam * (c[key] / q_j[j]))
                    top1[(j, lam, i)] = tp
                    row["adjusted_top1"][f"{j}:{lam}"] = tp
            f.write(json.dumps(row) + "\n")
        # root 93 proxy-only descriptive census
        f.write(json.dumps({
            "row_index": 93, "proxy_only": True,
            "candidates": [{k: e.get(k) for k in
                            ("pos", "name", "C1", "C2", "C3")}
                           for e in roots[93]["joined"]]})
                + "\n")
    raw_sha = fsha(OUTDIR / "rerank.jsonl")
    print(f"[cl1cost] RERANK SEALED sha256={raw_sha}",
          flush=True)

    # ---- SELECTION (frozen law; from in-memory = sealed rows) --
    for i in sorted(scored):
        gate(top1[("P1", 0, i)] == baseline[i],
             "LAMBDA0 IDENTITY")

    def cell(j, lam):
        key = {"P1": "C1", "P2": "C2", "P3": "C3"}[j]
        demote = 0
        equal_cost, higher_cost = [], []
        for i in FOCUS:
            b, t = baseline[i], top1[(j, lam, i)]
            if t != b:
                cb = scored[i]["joined"][b][key]
                ct = scored[i]["joined"][t][key]
                if ct < cb:
                    demote += 1
                elif ct == cb:
                    equal_cost.append(i)
                else:
                    higher_cost.append(i)
        c23 = top1[(j, lam, CONTROL)] == baseline[CONTROL]
        others = [i for i in scored if i not in FOCUS]
        changed = [i for i in others
                   if top1[(j, lam, i)] != baseline[i]]
        return demote, equal_cost, higher_cost, c23, changed

    pareto = {}
    survivors = {}
    for j in PROXIES:
        key = {"P1": "C1", "P2": "C2", "P3": "C3"}[j]
        for lam in GRID:
            d, eq, hc, c23, ch = cell(j, lam)
            eq_col, hi_col = [], []
            for i in ch:
                b, t = baseline[i], top1[(j, lam, i)]
                cb = scored[i]["joined"][b][key]
                ct = scored[i]["joined"][t][key]
                if ct == cb:
                    eq_col.append(i)
                elif ct > cb:
                    hi_col.append(i)
            pareto[f"{j}:{lam}"] = {
                "focus_demotion": d,
                "focus_equal_cost_displacements": eq,
                "focus_higher_cost_displacements": hc,
                "root23_preserved": c23,
                "collateral_changed": len(ch),
                "changed_roots": ch,
                "collateral_equal_cost": eq_col,
                "collateral_higher_cost": hi_col}
            if d == 4 and c23 and j not in survivors:
                survivors[j] = lam
    winner = None
    if survivors:
        order = sorted(
            survivors,
            key=lambda j: (
                pareto[f"{j}:{survivors[j]}"]
                ["collateral_changed"],
                survivors[j],
                PROXIES.index(j)))
        winner = order[0]
    result = ("NO-CLEAN-CHEAP-PROXY" if winner is None else
              {"proxy": winner, "q_j": q_j[winner],
               "lambda": survivors[winner],
               "collateral": pareto[
                   f"{winner}:{survivors[winner]}"]})
    print(json.dumps({"survivors": survivors,
                      "winner": result}, indent=1),
          flush=True)

    # ---- focus / control / rider tables ----
    def first_flip(j, i, genuine=False):
        """First grid lambda where top-1 changes; genuine=True
        additionally requires strictly lower proxy cost."""
        key = {"P1": "C1", "P2": "C2", "P3": "C3"}[j]
        for lam in GRID:
            t = top1[(j, lam, i)]
            if t != baseline[i]:
                if not genuine:
                    return lam
                cb = scored[i]["joined"][baseline[i]][key]
                if scored[i]["joined"][t][key] < cb:
                    return lam
        return None

    focus_table = {}
    for i in FOCUS:
        r = scored[i]
        b = baseline[i]
        ent = {"harmful_action": r["joined"][b]["name"],
               "baseline_scores_sorted_gap": None}
        sc = sorted((c["score"] for c in r["joined"]),
                    reverse=True)
        ent["top1_top2_gap"] = round(sc[0] - sc[1], 4)
        for j, key in (("P1", "C1"), ("P2", "C2"),
                       ("P3", "C3")):
            lam = first_flip(j, i, genuine=True)
            t = (top1[(j, lam, i)] if lam is not None
                 else None)
            ent[j] = {
                "harmful_cost": round(r["joined"][b][key], 4),
                "first_demote_lambda": lam,
                "replacement":
                    r["joined"][t]["name"] if t is not None
                    else None,
                "replacement_cost":
                    round(r["joined"][t][key], 4)
                    if t is not None else None}
        focus_table[str(i)] = ent
    r23 = scored[CONTROL]
    b23 = baseline[CONTROL]
    sc23 = sorted((c["score"] for c in r23["joined"]),
                  reverse=True)
    control_table = {
        "action": r23["joined"][b23]["name"],
        "C1": round(r23["joined"][b23]["C1"], 4),
        "C2": round(r23["joined"][b23]["C2"], 4),
        "C3": round(r23["joined"][b23]["C3"], 4),
        "score_margin": round(sc23[0] - sc23[1], 4),
        "first_flip_lambda": {j: first_flip(j, CONTROL)
                              for j in PROXIES}}
    rider_table = {"first_flip_lambda":
                   {j: first_flip(j, RIDER)
                    for j in PROXIES},
                   "replacement": {
                       j: (scored[RIDER]["joined"][
                           top1[(j, first_flip(j, RIDER),
                                 RIDER)]]["name"]
                           if first_flip(j, RIDER) is not None
                           else None)
                       for j in PROXIES}}

    receipt = {
        "prereg": "MATH-CYBER-1-CLOSED-LOOP-1-SEARCH-COST-"
                  "PROXY-DESK-PREREG-0",
        "prereg_commit": PREREG_COMMIT,
        "cl1_raw_pin": RAW_CL1_SHA,
        "manifest_sha": MANIFEST_SHA,
        "rerank_sha": raw_sha,
        "q_j": {j: repr(q_j[j]) for j in q_j},
        "timing_p123": timing,
        "survivors_min_lambda": survivors,
        "selected": result,
        "pareto": pareto,
        "focus_table": focus_table,
        "root23": control_table,
        "root94": rider_table,
        "start": START,
        "completion_commit": completion_commit()}
    (OUTDIR / "cl1cost_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print("[cl1cost] SELECTION SEALED", flush=True)

    # ---- P4 + HCE sweep (AFTER selection; diagnostic) ----
    t_sw = time.time()
    GLOBAL_SWEEP_DEADLINE_S = 6 * 3600.0
    skipped_roots = []
    with open(OUTDIR / "sweep.jsonl", "w") as f:
        for i in sorted(roots):
            if time.time() - t_sw > GLOBAL_SWEEP_DEADLINE_S:
                skipped_roots.append(i)
                f.write(json.dumps({"row_index": i,
                                    "skipped_deadline": True})
                        + "\n")
                continue
            want = [(lg["name"], lg["child_sstr"])
                    for lg in roots[i]["legal"]]
            rows, done = sweep_root(manifest[i], want)
            f.write(json.dumps({"row_index": i,
                                "complete": done,
                                "rows": rows}) + "\n")
            f.flush()
            n_ok = sum(1 for r in rows
                       if "n_child_legal" in r)
            print(f"[cl1cost] sweep root {i:2d}: "
                  f"{n_ok}/{len(want)} children "
                  f"(done={done})", flush=True)
    sweep_sha = fsha(OUTDIR / "sweep.jsonl")
    print(f"[cl1cost] SWEEP SEALED sha256={sweep_sha} "
          f"wall={round(time.time() - t_sw, 1)}s", flush=True)

    # ---- riders: P4 validity + HCE agreement + outcomes ----
    sweep = {}
    for l in open(OUTDIR / "sweep.jsonl"):
        t = json.loads(l)
        sweep[t["row_index"]] = t

    def spearman(xs, ys):
        def rank(v):
            order = sorted(range(len(v)), key=lambda k: v[k])
            rk = [0.0] * len(v)
            k = 0
            while k < len(order):
                m = k
                while (m + 1 < len(order)
                       and v[order[m + 1]] == v[order[k]]):
                    m += 1
                avg = (k + m) / 2 + 1
                for t2 in range(k, m + 1):
                    rk[order[t2]] = avg
                k = m + 1
            return rk
        rx, ry = rank(xs), rank(ys)
        n = len(xs)
        mx = sum(rx) / n
        my = sum(ry) / n
        num = sum((a - mx) * (b - my)
                  for a, b in zip(rx, ry))
        dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
        dy = math.sqrt(sum((b - my) ** 2 for b in ry))
        return num / (dx * dy) if dx and dy else None

    p4pairs = {"P1": [], "P2": [], "P3": []}
    hce_best = {}
    for i, r in roots.items():
        sw = sweep.get(i)
        if not sw:
            continue
        by_pos = {x["pos"]: x for x in sw["rows"]
                  if "n_child_legal" in x}
        kp = len(r["legal"])
        hvals = {}
        for pos, lg in enumerate(r["legal"]):
            x = by_pos.get(pos)
            if x is None:
                continue
            p4 = x["n_child_legal"] / kp
            e = (r["joined"][pos] if "joined" in r
                 and pos < len(r["joined"]) else None)
            if e is not None and "C1" in e:
                p4pairs["P1"].append(
                    (i, pos, e["C1"], p4))
                p4pairs["P2"].append(
                    (i, pos, e["C2"], p4))
                p4pairs["P3"].append(
                    (i, pos, e["C3"], p4))
            hvals[pos] = x["hce"]
        if hvals:
            # reconstructed hce argmin; tie law approximates
            # arm A's (hce, name, child key) with
            # (hce, name, child_sstr) — disclosed label
            hce_best[i] = min(
                hvals,
                key=lambda p: (hvals[p],
                               r["legal"][p]["name"],
                               r["legal"][p]["child_sstr"]))
    def corr(j):
        if not p4pairs[j]:
            return None
        v = spearman([c for _, _, c, _ in p4pairs[j]],
                     [p for _, _, _, p in p4pairs[j]])
        return round(v, 4) if v is not None else None
    p4_corr = {j: corr(j) for j in PROXIES}
    # P4 validity anatomy: FP = cheap-proxy top-decile cost
    # with bottom-half P4; FN = top-decile P4 with bottom-half
    # proxy cost (descriptive deciles over the measured pairs)
    p4_anatomy = {}
    for j in PROXIES:
        pr = p4pairs[j]
        if not pr:
            continue
        cs = sorted(c for _, _, c, _ in pr)
        ps = sorted(p for _, _, _, p in pr)
        c90 = cs[int(0.9 * (len(cs) - 1))]
        c50 = cs[len(cs) // 2]
        p90 = ps[int(0.9 * (len(ps) - 1))]
        p50 = ps[len(ps) // 2]
        fp = [(i, pos) for i, pos, c, p in pr
              if c >= c90 and p <= p50]
        fn = [(i, pos) for i, pos, c, p in pr
              if p >= p90 and c <= c50]
        p4_anatomy[j] = {
            "false_positives_cheapcost_lowP4": fp[:15],
            "false_negatives_highP4_lowcost": fn[:15],
            "n_fp": len(fp), "n_fn": len(fn)}
    # focus/23/94 placement under P4
    place = {}
    for i in FOCUS + [CONTROL, RIDER]:
        b = scored[i]["chosen_index"]
        pr = [(pos, p) for (ri, pos, _, p) in p4pairs["P1"]
              if ri == i]
        pv = dict(pr)
        place[str(i)] = {
            "booked_choice_P4": pv.get(b),
            "max_P4_in_set": max(pv.values()) if pv else None}
    # surrogate-failure check: sign of selected-proxy v P4
    # relation on the focus class (mechanical winner immutable)
    surrogate_flag = None
    if winner is not None:
        wpairs = [(c, p) for ri, pos, c, p in p4pairs[winner]
                  if ri in FOCUS]
        if wpairs:
            v = spearman([c for c, _ in wpairs],
                         [p for _, p in wpairs])
            surrogate_flag = {
                "focus_class_spearman": (round(v, 4)
                                         if v is not None
                                         else None),
                "STRING_COST_SURROGATE_FAILURE":
                    bool(v is not None and v < 0)}
    hce_agree = {"baseline": 0, "selected": 0, "n": 0,
                 "gained": [], "lost": [],
                 "tie_law_note": "reconstructed hce argmin "
                 "with (hce, name, child_sstr) ties — "
                 "approximates arm A's (hce, name, key)"}
    if winner is not None:
        lam = survivors[winner]
        for i in scored:
            if i not in hce_best:
                continue
            hce_agree["n"] += 1
            b_ok = baseline[i] == hce_best[i]
            s_ok = top1[(winner, lam, i)] == hce_best[i]
            hce_agree["baseline"] += b_ok
            hce_agree["selected"] += s_ok
            if s_ok and not b_ok:
                hce_agree["gained"].append(i)
            if b_ok and not s_ok:
                hce_agree["lost"].append(i)
    # known-outcome riders (post-selection, descriptive only):
    # join booked CL1 solved bits + categories to collateral
    outcome_join = {}
    if winner is not None:
        solved_a, solved_b = {}, {}
        for l in open(RAW_CL1):
            t = json.loads(l)
            if t["arm"] == "A":
                solved_a[t["row_index"]] = t["solved"]
            elif t["arm"] == "B":
                solved_b[t["row_index"]] = t["solved"]
        lam = survivors[winner]
        ch = pareto[f"{winner}:{lam}"]["changed_roots"]
        cat = {"concordant_solved": [], "concordant_failed": [],
               "A_only": [], "B_only": []}
        for i in ch:
            a, b = solved_a[i], solved_b[i]
            k = ("concordant_solved" if a and b else
                 "concordant_failed" if not a and not b else
                 "A_only" if a else "B_only")
            cat[k].append(i)
        outcome_join = {
            "selected_lambda_changed_roots_by_outcome": cat,
            "expand_family_changed":
                [i for i in ch
                 if i in (20, 23, 54, 68, 72, 78, 88)],
            "root72_changed": 72 in ch,
            "root94_changed": RIDER in ch}
    riders = {"sweep_sha": sweep_sha,
              "sweep_wall_s": round(time.time() - t_sw, 1),
              "sweep_skipped_roots": skipped_roots,
              "p4_spearman_v_proxies": p4_corr,
              "p4_pairs_n": {j: len(p4pairs[j])
                             for j in PROXIES},
              "p4_anatomy": p4_anatomy,
              "p4_placement_focus_23_94": place,
              "surrogate_failure_check": surrogate_flag,
              "hce_agreement_descriptive": hce_agree,
              "known_outcome_collateral_join": outcome_join}
    (OUTDIR / "riders.json").write_text(
        json.dumps(riders, indent=1))
    print(json.dumps({k: riders[k] for k in
                      ("p4_spearman_v_proxies",
                       "surrogate_failure_check",
                       "hce_agreement_descriptive",
                       "known_outcome_collateral_join")},
                     indent=1)[:2000], flush=True)
    print("[cl1cost] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
