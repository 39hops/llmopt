"""MATH-CYBER-1-RENDER-ATLAS-CAYLEY-LANDSCAPE-0 — zero-logit Cayley-graph
landscape analysis of the booked render atlases (definitions frozen at
RESULTS L65753; prereg L65895). Banks A (component geometry at the frozen
NEAR / MAJORITY / STRONG thresholds), B (Cayley distances and widest-path
barriers, set-to-set conventions), C (steepest-ascent basins per field and
under the frozen lexicographic maximin tuple), D (edge-conflict field,
local Pareto optimality, common-monotone paths) and K (2 x 2 similarity),
run on the four discovery atlases and then unchanged on the eight fresh
atlases (never pooled). No model, no logit; torch is never imported.

Usage:
    .venv/bin/python scratch/mathworld1_cayley.py
"""
import hashlib
import itertools
import json
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.provenance import completion_commit, start_provenance  # noqa: E402


def gate(cond, msg):
    """Local gate (the svpbirth gate would import torch; this analysis
    must stay torch-free by construction)."""
    if not cond:
        raise SystemExit(f"GATE FAILED: {msg}")

PINS = {
    "logs/mathworld1/prband2atlas/atlas_manifest.jsonl":
        "687b5e54e0da19bf057431eb4d44b755302c1963d18e13fb6d316fa99dd2f4b2",
    "logs/mathworld1/prband2atlas/atlas_policies.jsonl":
        "b4a1c08308ca429d4bd7eb01210dfe469cd11eb9c9c527daf42997cec6d86c71",
    "logs/mathworld1/prband2atlasscore/policy_table.jsonl":
        "73dba755a3089a42177ef00a91481e93c2114276938df92958509a8aeda2505e",
    "logs/mathworld1/prband2atlasscore/aggregate.json":
        "8df318f742952740ce44c98d57fd8fb2f9632fa283d75f5c4808c4e158e66105",
    "logs/mathworld1/prband2atlasfresh/policy_table.jsonl":
        "e42999947102112d03d0aa7529d79daf5896e0cc972ace8aa2acc41b6381d321",
    "logs/mathworld1/prband2atlasfresh/aggregate.json":
        "c6d16e8c7b2c68b9d73e9e7e28eb2d6e0b354b3ee64e459ca66460ccad1fa443"}
OUTDIR = Path("logs/mathworld1/cayley")
ROLES = ("HI_D", "HI_L", "LO_D", "LO_L", "K", "W")
THRESH = {"NEAR": lambda T, B, Tb, Bs: T >= Tb - 4 and B >= Bs - 2,
          "MAJORITY": lambda T, B, Tb, Bs: T >= 72 and B >= 24,
          "STRONG": lambda T, B, Tb, Bs: T >= 84 and B >= 36}
COHORTS = {
    "DISCOVERY": {"table": "logs/mathworld1/prband2atlasscore/policy_table.jsonl",
                  "cks": ["19001|CANONICAL", "19001|PARAM_FIRST", "20001|CANONICAL", "20001|PARAM_FIRST"],
                  "anchors": {"RAW": 12, "K_FIRST": 480, "LOW_PAIR_FIRST": 300, "R488": 488}},
    "FRESH": {"table": "logs/mathworld1/prband2atlasfresh/policy_table.jsonl",
              "cks": ["21001|CANONICAL", "21001|PARAM_FIRST", "22001|CANONICAL", "22001|PARAM_FIRST",
                      "23001|CANONICAL", "23001|PARAM_FIRST", "24001|CANONICAL", "24001|PARAM_FIRST"],
              "anchors": {"RAW": 12, "R488": 488, "R128": 128}}}


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_graph():
    man = [json.loads(l) for l in open("logs/mathworld1/prband2atlas/atlas_manifest.jsonl")]
    idx = {tuple(m["roles"]): m["atlas_index"] for m in man}
    gate(len(man) == 720 and len(idx) == 720 and [m["atlas_index"] for m in man] == list(range(720)), "MANIFEST")
    adj = {i: set() for i in range(720)}
    edges = set()
    for r, i in idx.items():
        for k in range(5):
            q = list(r)
            q[k], q[k + 1] = q[k + 1], q[k]
            j = idx[tuple(q)]
            adj[i].add(j)
            edges.add((min(i, j), max(i, j)))
    gate(all(len(v) == 5 for v in adj.values()) and len(edges) == 1800, "DEGREE/EDGES")
    adj = {i: sorted(v) for i, v in adj.items()}
    # diameter by BFS from every vertex
    diam = 0
    for s in range(720):
        d = bfs_dist(adj, [s])
        diam = max(diam, max(d.values()))
        gate(len(d) == 720, "CONNECTED")
    gate(diam == 15, f"DIAMETER {diam}")
    return man, adj, sorted(edges), {m["atlas_index"]: m["render_id"] for m in man}


def bfs_dist(adj, sources, allowed=None):
    d = {s: 0 for s in sources}
    q = deque(sources)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v in d or (allowed is not None and v not in allowed):
                continue
            d[v] = d[u] + 1
            q.append(v)
    return d


def components(adj, S):
    S = set(S)
    seen, comps = set(), []
    for v in sorted(S):
        if v in seen:
            continue
        c, st = {v}, [v]
        seen.add(v)
        while st:
            u = st.pop()
            for w in adj[u]:
                if w in S and w not in seen:
                    seen.add(w)
                    c.add(w)
                    st.append(w)
        comps.append(sorted(c))
    return sorted(comps, key=lambda c: (-len(c), c[0]))


def comp_diameter(adj, c):
    cs = set(c)
    return max(max(bfs_dist(adj, [v], cs).values()) for v in c) if len(c) > 1 else 0


def widest(adj, field, A, Z):
    """Max over endpoint pairs of the bottleneck value; ties -> ascending (a, z);
    canonical path = BFS inside {v: field[v] >= W} from a to z, neighbors ascending."""
    best = None
    for a in sorted(A):
        for z in sorted(Z):
            hi = min(field[a], field[z])
            vals = sorted({field[v] for v in range(720) if field[v] <= hi}, reverse=True)
            w = None
            for t in vals:
                allowed = {v for v in range(720) if field[v] >= t}
                if a in allowed and z in allowed and z in bfs_dist(adj, [a], allowed):
                    w = t
                    break
            if w is None:
                continue
            if best is None or w > best[0]:
                best = (w, a, z)
    w, a, z = best
    allowed = {v for v in range(720) if field[v] >= w}
    par = {a: None}
    q = deque([a])
    while q and z not in par:
        u = q.popleft()
        for v in adj[u]:
            if v in allowed and v not in par:
                par[v] = u
                q.append(v)
    path = [z]
    while par[path[-1]] is not None:
        path.append(par[path[-1]])
    path.reverse()
    return {"W": w, "a": a, "z": z, "D": min(field[a], field[z]) - w, "path": path,
            "path_len": len(path) - 1, "path_min_field": min(field[v] for v in path)}


def ascend(adj, key):
    """Steepest ascent: move to the strictly better neighbor with the max key,
    ties by ascending atlas index; returns (terminal, path length)."""
    out = {}
    for s in range(720):
        v, n = s, 0
        while True:
            best, bk_best, bk = None, None, key(v)
            for w in adj[v]:
                kw = key(w)
                if kw > bk and (best is None or kw > bk_best or (kw == bk_best and w < best)):
                    best, bk_best = w, kw
            if best is None:
                break
            v, n = best, n + 1
        out[s] = (v, n)
    return out


def analyze(cohort, spec, adj, edges, rid, pol):
    cks = spec["cks"]
    tab = {}
    for l in open(spec["table"]):
        r = json.loads(l)
        tab[r["atlas_index"]] = r
    gate(len(tab) == 720, "TABLE")
    T = {ck: {i: tab[i][ck]["T"] for i in range(720)} for ck in cks}
    B = {ck: {i: tab[i][ck]["B"] for i in range(720)} for ck in cks}
    opt = {}
    for ck in cks:
        Bs, Tb = max(B[ck].values()), max(T[ck].values())
        opt[ck] = {"B_star": Bs, "Tbest": Tb,
                   "set": sorted(i for i in range(720) if B[ck][i] == Bs and T[ck][i] == Tb),
                   "B_set": sorted(i for i in range(720) if B[ck][i] == Bs)}
        gate(len(opt[ck]["set"]) >= 1, "OPT SET")
    anchors = dict(spec["anchors"])
    out = {"cohort": cohort, "checkpoints": cks, "anchors": anchors, "optima": opt}
    # ---- Bank A ----
    A = {}
    for ck in cks:
        A[ck] = {}
        for name, fn in THRESH.items():
            S = [i for i in range(720) if fn(T[ck][i], B[ck][i], opt[ck]["Tbest"], opt[ck]["B_star"])]
            Sset = set(S)
            comps = components(adj, S)
            internal = sum(1 for a, b in edges if a in Sset and b in Sset)
            boundary = sum(1 for a, b in edges if (a in Sset) != (b in Sset))
            cid = {v: k for k, c in enumerate(comps) for v in c}
            optc = {cid[v] for v in opt[ck]["set"] if v in cid}
            memb = {an: (v in Sset) for an, v in anchors.items()}
            r488 = {"in_set": 488 in Sset, "same_component_as_an_optimum": None, "within_threshold_path_len": None}
            if 488 in Sset and optc:
                if cid[488] in optc:
                    d = bfs_dist(adj, [488], Sset)
                    r488["same_component_as_an_optimum"] = True
                    r488["within_threshold_path_len"] = min(d[v] for v in opt[ck]["set"] if v in Sset and cid[v] == cid[488])
                else:
                    r488["same_component_as_an_optimum"] = False
                    r488["status"] = "DISCONNECTED AT THIS THRESHOLD"
            A[ck][name] = {"vertices": len(S), "n_components": len(comps),
                           "component_sizes": [len(c) for c in comps],
                           "largest_fraction": (len(comps[0]) / len(S)) if S else None,
                           "internal_edges": internal, "boundary_edges": boundary,
                           "boundary_over_volume": (boundary / len(S)) if S else None,
                           "isolated": sum(len(c) == 1 for c in comps),
                           "component_diameters": [comp_diameter(adj, c) for c in comps if len(c) > 1],
                           "optimum_set_in_set": all(v in Sset for v in opt[ck]["set"]),
                           "optima_in_one_component": len(optc) == 1,
                           "optimum_components": sorted(optc),
                           "membership": memb, "R488": r488,
                           "largest_component": comps[0][:64] if comps else []}
    out["A"] = A
    # ---- Bank B ----
    Bk = {"distances": {}, "widest": {}}
    sets = {an: [v] for an, v in anchors.items()}
    for ck in cks:
        sets[f"OPT[{ck}]"] = opt[ck]["set"]
    names = list(sets)
    for a_, b_ in itertools.combinations(names, 2):
        d = bfs_dist(adj, sets[a_])
        Bk["distances"][f"{a_} <-> {b_}"] = min(d[v] for v in sets[b_])
    for ck in cks:
        for an in anchors:
            for fld, F in (("B", B[ck]), ("T", T[ck])):
                Bk["widest"][f"{ck}|{fld}|{an} <-> OPT"] = widest(adj, F, sets[an], opt[ck]["set"])
        for ck2 in cks:
            if ck2 <= ck:
                continue
            for fld, F in (("B", B[ck]), ("T", T[ck])):
                Bk["widest"][f"{ck}|{fld}|OPT[{ck}] <-> OPT[{ck2}]"] = widest(adj, F, opt[ck]["set"], opt[ck2]["set"])
    out["B"] = Bk
    # ---- Bank C ----
    C = {}
    for ck in cks:
        C[ck] = {}
        for fld, F in (("B", B[ck]), ("T", T[ck])):
            flow = ascend(adj, lambda v, F=F: F[v])
            maxima = Counter(t for t, _n in flow.values())
            glob = set(opt[ck]["B_set"] if fld == "B" else [i for i in range(720) if T[ck][i] == opt[ck]["Tbest"]])
            gmax = max(F.values())
            basins = sorted(maxima.items(), key=lambda kv: (-kv[1], kv[0]))
            C[ck][fld] = {"n_local_maxima": len(maxima),
                          "local_maxima": [{"v": v, "score": F[v], "basin": n, "global": v in glob} for v, n in basins],
                          "starts_reaching_global": sum(n for v, n in maxima.items() if F[v] == gmax),
                          "largest_basin": basins[0][1], "largest_basin_is_global": F[basins[0][0]] == gmax,
                          "global_basin_ranks": [k + 1 for k, (v, n) in enumerate(basins) if F[v] == gmax],
                          "anchor_flow": {an: {"terminal": flow[v][0], "terminal_score": F[flow[v][0]],
                                               "steps": flow[v][1], "terminal_is_global": F[flow[v][0]] == gmax}
                                          for an, v in anchors.items()},
                          "mean_path_len": sum(n for _t, n in flow.values()) / 720,
                          "max_path_len": max(n for _t, n in flow.values())}
    # joint lexicographic maximin objective
    def jkey(v):
        bs = [B[ck][v] for ck in cks]
        ts = [T[ck][v] for ck in cks]
        return (min(bs), min(ts), sum(bs), sum(ts), -int(rid[v][:16], 16))
    flow = ascend(adj, jkey)
    maxima = Counter(t for t, _n in flow.values())
    gbest = max(jkey(v) for v in range(720))
    C["JOINT"] = {"n_local_maxima": len(maxima),
                  "local_maxima": [{"v": v, "key": list(jkey(v))[:4], "basin": n, "global": jkey(v) == gbest}
                                   for v, n in sorted(maxima.items(), key=lambda kv: (-kv[1], kv[0]))],
                  "anchor_flow": {an: {"terminal": flow[v][0], "steps": flow[v][1],
                                       "is_local_optimum": flow[v][0] == v}
                                  for an, v in anchors.items()},
                  "starts_reaching_global": sum(n for v, n in maxima.items() if jkey(v) == gbest)}
    out["C"] = C
    # ---- Bank D ----
    dT = {ck: {} for ck in cks}
    dB = {ck: {} for ck in cks}
    for a, b in edges:
        for ck in cks:
            dT[ck][(a, b)] = T[ck][b] - T[ck][a]
            dB[ck][(a, b)] = B[ck][b] - B[ck][a]
    def sgn(x):
        return (x > 0) - (x < 0)
    pairs = {}
    for c1, c2 in itertools.combinations(cks, 2):
        st = {}
        for fld, dd in (("T", dT), ("B", dB)):
            x = [dd[c1][e] for e in edges]
            y = [dd[c2][e] for e in edges]
            nz = [(u, v) for u, v in zip(x, y) if u != 0 and v != 0]
            st[fld] = {"both_pos": sum(u > 0 and v > 0 for u, v in zip(x, y)),
                       "both_neg": sum(u < 0 and v < 0 for u, v in zip(x, y)),
                       "opposite": sum(sgn(u) * sgn(v) < 0 for u, v in zip(x, y)),
                       "either_zero": sum(u == 0 or v == 0 for u, v in zip(x, y)),
                       "sign_agreement_nonzero": (sum(sgn(u) == sgn(v) for u, v in nz) / len(nz)) if nz else None,
                       "pearson": pearson(x, y), "spearman": spearman(x, y),
                       "abs_pearson": pearson([abs(u) for u in x], [abs(v) for v in y]),
                       "largest_opposite": sorted(((abs(u) + abs(v)), e) for e, u, v in zip(edges, x, y)
                                                  if sgn(u) * sgn(v) < 0)[-3:][::-1]}
        pairs[f"{c1} v {c2}"] = st
    def improves(e, ck):
        return (dB[ck][e], dT[ck][e]) > (0, 0)
    def weak(e, ck):
        return (dB[ck][e], dT[ck][e]) >= (0, 0)
    hist = Counter(sum(dB[ck][e] > 0 for ck in cks) for e in edges)
    allimp = [e for e in edges if all(weak(e, ck) for ck in cks) and any(improves(e, ck) for ck in cks)]
    allimp_rev = [e for e in edges if all((-dB[ck][e], -dT[ck][e]) >= (0, 0) for ck in cks)
                  and any((-dB[ck][e], -dT[ck][e]) > (0, 0) for ck in cks)]
    D = {"pairs": pairs,
         "edges_B_improving_all": sum(all(dB[ck][e] > 0 for ck in cks) for e in edges),
         "edges_B_hurting_all": sum(all(dB[ck][e] < 0 for ck in cks) for e in edges),
         "edges_B_unchanged_all": sum(all(dB[ck][e] == 0 for ck in cks) for e in edges),
         "edges_B_mixed": sum(any(dB[ck][e] > 0 for ck in cks) and any(dB[ck][e] < 0 for ck in cks) for e in edges),
         "hist_B_improved_count": {str(k): v for k, v in sorted(hist.items())},
         "edges_lex_weakly_improving_all_forward": len(allimp),
         "edges_lex_weakly_improving_all_reverse": len(allimp_rev)}
    def vertex_profile(v):
        prof = Counter()
        for w in adj[v]:
            e = (min(v, w), max(v, w))
            s = 1 if v < w else -1
            n = sum(s * dB[ck][e] > 0 or (s * dB[ck][e] == 0 and s * dT[ck][e] > 0) for ck in cks)
            prof[n] += 1
        return {str(k): prof[k] for k in range(len(cks) + 1)}
    def local_pareto(v):
        for w in adj[v]:
            e = (min(v, w), max(v, w))
            s = 1 if v < w else -1
            wk = all((s * dB[ck][e], s * dT[ck][e]) >= (0, 0) for ck in cks)
            stx = any((s * dB[ck][e], s * dT[ck][e]) > (0, 0) for ck in cks)
            if wk and stx:
                return False, w
        return True, None
    D["vertex_profiles"] = {an: vertex_profile(v) for an, v in anchors.items()}
    D["optimum_profiles"] = {ck: {str(v): vertex_profile(v) for v in opt[ck]["set"][:8]} for ck in cks}
    D["local_pareto"] = {an: dict(zip(("is_local_pareto_optimum", "improving_neighbor"), local_pareto(v)))
                         for an, v in anchors.items()}
    # monotone paths RAW -> R488 (and RAW -> R128 in fresh)
    def mono_path(src, dst, rule):
        par = {src: None}
        q = deque([src])
        while q and dst not in par:
            u = q.popleft()
            for w in adj[u]:
                e = (min(u, w), max(u, w))
                s = 1 if u < w else -1
                ok = all(rule(s * dB[ck][e], s * dT[ck][e]) for ck in cks)
                if ok and w not in par:
                    par[w] = u
                    q.append(w)
        if dst not in par:
            return {"exists": False, "reachable_set_size": len(par)}
        p = [dst]
        while par[p[-1]] is not None:
            p.append(par[p[-1]])
        return {"exists": True, "path": p[::-1], "len": len(p) - 1, "reachable_set_size": len(par)}
    D["monotone_paths"] = {}
    for an, v in anchors.items():
        if an == "RAW":
            continue
        D["monotone_paths"][f"RAW -> {an} (B non-decreasing)"] = mono_path(12, v, lambda db, dt: db >= 0)
        D["monotone_paths"][f"RAW -> {an} (lex (B,T) non-decreasing)"] = mono_path(12, v, lambda db, dt: (db, dt) >= (0, 0))
    out["D"] = D
    # ---- Bank K ----
    K = {}
    maj = {ck: {i for i in range(720) if T[ck][i] >= 72 and B[ck][i] >= 24} for ck in cks}
    strg = {ck: {i for i in range(720) if T[ck][i] >= 84 and B[ck][i] >= 36} for ck in cks}
    near = {ck: {i for i in range(720) if T[ck][i] >= opt[ck]["Tbest"] - 4 and B[ck][i] >= opt[ck]["B_star"] - 2} for ck in cks}
    def jac(a, b):
        return (len(a & b) / len(a | b)) if (a | b) else None
    for c1, c2 in itertools.combinations(cks, 2):
        s1, r1 = c1.split("|")
        s2, r2 = c2.split("|")
        d1 = bfs_dist(adj, opt[c1]["set"])
        dn = bfs_dist(adj, sorted(near[c1]))
        K[f"{c1} v {c2}"] = {"same_seed": s1 == s2, "same_rep": r1 == r2,
                             "spearman_T_nodes": spearman([T[c1][i] for i in range(720)], [T[c2][i] for i in range(720)]),
                             "spearman_B_nodes": spearman([B[c1][i] for i in range(720)], [B[c2][i] for i in range(720)]),
                             "pearson_dT_edges": pairs[f"{c1} v {c2}"]["T"]["pearson"],
                             "pearson_dB_edges": pairs[f"{c1} v {c2}"]["B"]["pearson"],
                             "jaccard_MAJORITY": jac(maj[c1], maj[c2]), "jaccard_STRONG": jac(strg[c1], strg[c2]),
                             "dist_optimum_sets": min(d1[v] for v in opt[c2]["set"]),
                             "dist_near_sets": min(dn[v] for v in near[c2])}
    def summ(keyfn, label):
        groups = defaultdict(list)
        for k, v in K.items():
            groups["same_seed" if v["same_seed"] else "same_rep" if v["same_rep"] else "neither"].append(v[label])
        return {g: {"n": len(x), "median": sorted(x)[len(x) // 2], "mean": sum(x) / len(x)} for g, x in groups.items()}
    K["_summary"] = {lab: summ(None, lab) for lab in ("spearman_B_nodes", "pearson_dB_edges", "pearson_dT_edges",
                                                       "jaccard_MAJORITY", "dist_optimum_sets", "dist_near_sets")}
    out["K"] = K
    return out


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else None


def spearman(x, y):
    def rk(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            for k in range(i, j + 1):
                out[idx[k]] = (i + j) / 2 + 1
            i = j + 1
        return out
    return pearson(rk(x), rk(y))


def main():
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN DRIFT {p}")
    gate("torch" not in sys.modules, "TORCH IMPORTED")
    START = start_provenance(["scratch/mathworld1_cayley.py", "llmopt/lab/provenance.py"])
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for f in ("cayley_receipt.json", "DISCOVERY.json", "FRESH.json"):
        gate(not (OUTDIR / f).exists(), f"REFUSE OVERWRITE {f}")
    man, adj, edges, rid = build_graph()
    pol = {p["atlas_index"]: p for p in map(json.loads, open("logs/mathworld1/prband2atlas/atlas_policies.jsonl"))}
    results = {}
    for cohort, spec in COHORTS.items():
        res = analyze(cohort, spec, adj, edges, rid, pol)
        (OUTDIR / f"{cohort}.json").write_text(json.dumps(res, indent=1))
        results[cohort] = res
        print(f"[{cohort}] done {time.time() - t0:.1f}s", flush=True)
    receipt = {"prereg": "MATH-CYBER-1-RENDER-ATLAS-CAYLEY-LANDSCAPE-0", "pins": {p: fsha(p) for p in PINS},
               "graph": {"V": 720, "E": len(edges), "degree": 5, "diameter": 15},
               "cohorts": {c: {"sha256": fsha(str(OUTDIR / f"{c}.json")), "checkpoints": COHORTS[c]["cks"],
                               "anchors": COHORTS[c]["anchors"]} for c in COHORTS},
               "semantic_beyond_all_surface_identifiable": False,
               "wall_s": round(time.time() - t0, 1), "start": START, "completion_commit": completion_commit()}
    (OUTDIR / "cayley_receipt.json").write_text(json.dumps(receipt, indent=1))
    for c, res in results.items():
        print("==", c)
        for ck in res["checkpoints"]:
            a = res["A"][ck]
            print(ck, {n: (a[n]["vertices"], a[n]["n_components"], a[n]["component_sizes"][:5], a[n]["R488"].get("same_component_as_an_optimum")) for n in THRESH})
        print("mono", {k: v["exists"] for k, v in res["D"]["monotone_paths"].items()})
        print("localpareto", res["D"]["local_pareto"])
        print("D edges", {k: v for k, v in res["D"].items() if k.startswith("edges") or k.startswith("hist")})
        for ck in res["checkpoints"]:
            c_ = res["C"][ck]["B"]
            print(ck, "B maxima", c_["n_local_maxima"], "reach global", c_["starts_reaching_global"], "largest", c_["largest_basin"], c_["largest_basin_is_global"], "RAW->", c_["anchor_flow"]["RAW"])
        print("JOINT", res["C"]["JOINT"]["anchor_flow"], res["C"]["JOINT"]["n_local_maxima"], res["C"]["JOINT"]["starts_reaching_global"])
        print("K summary", json.dumps(res["K"]["_summary"]))


if __name__ == "__main__":
    main()
