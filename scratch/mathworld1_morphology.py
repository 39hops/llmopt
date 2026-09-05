"""MATH-CYBER-1-RENDER-ATLAS-MORPHOLOGY-0 — zero-logit plateau-quotient and
morphology analysis of the twelve booked render-atlas fields (definitions
frozen at RESULTS L65937 / L65753; prereg L65989). Step P0 (plateau
partition, quotient graph, quotient local maxima, quotient steepest ascent)
runs first; then banks E (robustness radius), F (distance-to-optimum
profile), H (total variation, Dirichlet energy, Laplacian spectrum, 200
string-seeded relabelings as a reference), I (Pareto-front geometry). The
four discovery atlases and the eight fresh atlases are analysed separately
and never pooled. Graph builder, gates, BFS and component helpers are
imported from the frozen scratch/mathworld1_cayley.py (not edited). No
model, no logit; torch is never imported.

Usage:
    .venv/bin/python scratch/mathworld1_morphology.py
"""
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mathworld1_cayley as C  # noqa: E402  (frozen instrument, imported not copied)
from llmopt.lab.provenance import completion_commit, start_provenance  # noqa: E402

gate, fsha, bfs_dist, components = C.gate, C.fsha, C.bfs_dist, C.components
SMOKE = os.environ.get("MORPH_SMOKE") == "1"   # path-isolated smoke: own directory, fewer relabelings
OUTDIR = Path("logs/mathworld1/morphology_smoke" if SMOKE else "logs/mathworld1/morphology")
CAYLEY = {"logs/mathworld1/cayley/DISCOVERY.json": "fc25acd443343fedad44fbed411cb059907a7dcbd39aaacdb2683c1cc0cc64e2",
          "logs/mathworld1/cayley/FRESH.json": "101475dbd35153974ae53f1d8170b1aea53d223d01a0e322f7a52bf254504307",
          "logs/mathworld1/cayley/cayley_receipt.json": "07f53a2bfe1ce407d323eac12eccbb3f5800baf6825ff5e0a37ea8653a70bdf9"}
PINS = dict(C.PINS, **CAYLEY)
N_NULL = 5 if SMOKE else 200
CUTS = {"10%": 72, "25%": 180, "50%": 360}      # round(fraction x 719) non-constant modes
CLASS = lambda m: "FLAT-WITH-FEW-PEAKS" if m <= 10 else ("RUGGED" if m >= 30 else "INTERMEDIATE")
FRONT_SIZES = {"DISCOVERY": {"B": 20, "T": 21}, "FRESH": {"B": 78, "T": 114}}


def union_find_plateaus(adj, x):
    """Plateaus = connected components of equal-value induced subgraphs."""
    parent = list(range(720))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for u in range(720):
        for v in adj[u]:
            if v > u and x[u] == x[v]:
                ru, rv = find(u), find(v)
                if ru != rv:
                    parent[max(ru, rv)] = min(ru, rv)
    pid = {}
    members = defaultdict(list)
    for v in range(720):
        r = find(v)
        members[r].append(v)
    plats = sorted(members.values(), key=lambda m: m[0])
    for k, m in enumerate(plats):
        for v in m:
            pid[v] = k
    return plats, pid


def p0(adj, x, vertex_level_list):
    plats, pid = union_find_plateaus(adj, x)
    score = [x[m[0]] for m in plats]
    lo, hi = min(x), max(x)
    qadj = [set() for _ in plats]
    for u in range(720):
        for v in adj[u]:
            if pid[u] != pid[v]:
                qadj[pid[u]].add(pid[v])
    for k in range(len(plats)):
        gate(all(score[j] != score[k] for j in qadj[k]), "QUOTIENT ADJACENT EQUAL SCORE")
    qmax = [k for k in range(len(plats)) if all(score[j] < score[k] for j in qadj[k])]
    floor_plats = [k for k in range(len(plats)) if score[k] == lo]
    ceil_plats = [k for k in range(len(plats)) if score[k] == hi]
    interior = [k for k in qmax if lo < score[k] < hi]
    # quotient steepest ascent
    term, steps = {}, {}
    for k in range(len(plats)):
        p, n = k, 0
        while True:
            cand = [j for j in qadj[p] if score[j] > score[p]]
            if not cand:
                break
            best = max(score[j] for j in cand)
            p = min((j for j in cand if score[j] == best), key=lambda j: plats[j][0])
            n += 1
        term[k], steps[k] = p, n
    basin = Counter()
    for k in range(len(plats)):
        basin[term[k]] += len(plats[k])
    gate(set(basin) <= set(qmax) and all(term[k] == k for k in qmax), "ASCENT TERMINALS")
    reach_global = sum(len(plats[k]) for k in range(len(plats)) if score[term[k]] == hi)
    largest = max(basin.items(), key=lambda kv: (kv[1], -plats[kv[0]][0]))
    # vertex-level (L65907 bank C) decomposition, joined to the field
    vl = vertex_level_list
    vl_floor = sum(1 for m in vl if m["score"] == lo)
    vl_ceil = sum(1 for m in vl if m["score"] == hi)
    gate(all(x[m["v"]] == m["score"] for m in vl), "VERTEX-LEVEL JOIN")
    gate(len(qmax) <= len(vl), "QUOTIENT > VERTEX COUNT")
    gate(all(all(x[w] <= x[v] for w in adj[v]) for k in qmax for v in plats[k]), "QMAX NOT FIXED POINTS")
    gate(not any(k in qmax for k in floor_plats) or lo == hi, "FLOOR PLATEAU IS MAXIMUM")
    sizes = [len(m) for m in plats]
    return {"n_plateaus": len(plats), "plateau_size_max": max(sizes), "plateau_size_mean": round(sum(sizes) / len(sizes), 3),
            "n_singletons": sum(s == 1 for s in sizes), "n_plateaus_ge5": sum(s >= 5 for s in sizes),
            "floor": lo, "ceiling": hi, "n_floor_plateaus": len(floor_plats),
            "floor_total_size": sum(len(plats[k]) for k in floor_plats),
            "floor_plateau_is_maximum": any(k in qmax for k in floor_plats),
            "n_ceiling_plateaus": len(ceil_plats), "ceiling_plateau_sizes": [len(plats[k]) for k in ceil_plats],
            "n_quotient_maxima": len(qmax), "n_interior_quotient_maxima": len(interior),
            "class": CLASS(len(interior)),
            "quotient_maxima": [{"min_index": plats[k][0], "score": score[k], "size": len(plats[k]),
                                 "basin_vertices": basin.get(k, 0), "global": score[k] == hi} for k in
                                sorted(qmax, key=lambda k: (-score[k], plats[k][0]))],
            "vertex_level": {"n_fixed_points": len(vl), "on_floor": vl_floor, "above_floor": len(vl) - vl_floor,
                             "interior": len(vl) - vl_floor - vl_ceil, "at_ceiling": vl_ceil},
            "starts_reaching_global": reach_global, "largest_basin": largest[1],
            "largest_basin_is_global": score[largest[0]] == hi,
            "largest_global_basin": max([basin.get(k, 0) for k in ceil_plats]),
            "mean_path_len": round(sum(steps[pid[v]] for v in range(720)) / 720, 4),
            "max_path_len": max(steps.values()),
            "_pid": pid, "_plats": plats, "_term": term, "_score": score}


def anchor_flow(res, anchors):
    out = {}
    for name, a in anchors.items():
        k = res["_pid"][a]
        t = res["_term"][k]
        out[name] = {"start_plateau_min_index": res["_plats"][k][0], "start_plateau_size": len(res["_plats"][k]),
                     "terminal_min_index": res["_plats"][t][0], "terminal_score": res["_score"][t],
                     "terminal_global": res["_score"][t] == res["ceiling"]}
    return out


def radius(adj, S, v):
    if v not in S:
        d = bfs_dist(adj, [v])
        return {"status": "NOT-IN-SET", "distance_to_set": min(d[u] for u in S)}
    d = bfs_dist(adj, [v])
    return {"status": "IN-SET", "radius": min(d[u] for u in range(720) if u not in S)}


def spearman(x, y):
    return C.spearman(x, y)


def shell_stats(vals):
    a = np.array(vals, dtype=float)
    q25, q75 = np.percentile(a, [25, 75])
    return {"n": len(vals), "mean": round(float(a.mean()), 4), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max()), "iqr": round(float(q75 - q25), 4)}


def bank_f(adj, opt, B, T):
    d = bfs_dist(adj, sorted(opt))
    gate(len(d) == 720, "F BFS")
    shells = {}
    for s in sorted(set(d.values())):
        vs = [v for v in range(720) if d[v] == s]
        shells[str(s)] = {"B": shell_stats([B[v] for v in vs]), "T": shell_stats([T[v] for v in vs])}
    dv = [d[v] for v in range(720)]
    out = {"max_distance": max(dv), "shells": shells,
           "spearman_B_vs_d": round(spearman([B[v] for v in range(720)], dv), 6),
           "spearman_T_vs_d": round(spearman([T[v] for v in range(720)], dv), 6)}
    nz = [v for v in range(720) if B[v] > 0]
    sh2 = {}
    for s in sorted({d[v] for v in nz}):
        vs = [v for v in nz if d[v] == s]
        sh2[str(s)] = {"B": shell_stats([B[v] for v in vs]), "T": shell_stats([T[v] for v in vs])}
    out["floor_complement"] = {"n": len(nz), "shells": sh2,
                               "spearman_B_vs_d": round(spearman([B[v] for v in nz], [d[v] for v in nz]), 6) if len(nz) > 1 else None}
    return out


def tv_q(edges, x):
    x = np.asarray(x, dtype=float)
    d = np.abs(x[[e[0] for e in edges]] - x[[e[1] for e in edges]])
    tv, q = float(d.sum()), float((d ** 2).sum())
    rng, var = float(x.max() - x.min()), float(x.var())
    return {"TV": tv, "Q": q, "TV_norm": tv / (len(edges) * rng) if rng else None,
            "Q_norm": q / (len(edges) * var) if var else None, "d": d}


def spectral(U, lam, x, k_cuts):
    x = np.asarray(x, dtype=float)
    c = U.T @ (x - x.mean())
    e = c[1:] ** 2                      # non-constant modes, ascending eigenvalue
    tot = float(e.sum())
    cum = np.cumsum(e)
    out = {"total_energy": tot, "cuts": {}}
    for name, k in k_cuts.items():
        lam_k, lam_k1 = float(lam[k]), float(lam[k + 1])   # lam[k] is the k-th non-constant mode (index k of 0..719)
        split = abs(lam_k - lam_k1) < 1e-9
        cut = {"k": k, "fraction": float(cum[k - 1] / tot) if tot else None, "eigenvalue_at_cut": round(lam_k, 9),
               "splits_degenerate_eigenspace": split}
        if split:
            lo = k
            while lo > 1 and abs(float(lam[lo]) - lam_k) < 1e-9:
                lo -= 1
            hi = k
            while hi < 719 and abs(float(lam[hi + 1]) - lam_k) < 1e-9:
                hi += 1
            cut["boundary_below"] = {"k": lo, "fraction": float(cum[lo - 1] / tot) if tot else None}
            cut["boundary_above"] = {"k": hi, "fraction": float(cum[hi - 1] / tot) if tot else None}
        out["cuts"][name] = cut
    return out


def pareto(vecs):
    front = []
    for i, v in vecs.items():
        n = len(v)
        if not any(all(w[k] >= v[k] for k in range(n)) and any(w[k] > v[k] for k in range(n))
                   for j, w in vecs.items() if j != i):
            front.append(i)
    return sorted(front)


def analyze(cohort, spec, adj, edges, U, lam, cay):
    rows = {r["atlas_index"]: r for r in map(json.loads, open(spec["table"]))}
    gate(len(rows) == 720, "TABLE")
    cks = spec["cks"]
    F = {ck: {"B": [rows[i][ck]["B"] for i in range(720)], "T": [rows[i][ck]["T"] for i in range(720)]} for ck in cks}
    opt = {ck: [i for i in range(720) if F[ck]["B"][i] == 48 and F[ck]["T"][i] == 96] for ck in cks}
    for ck in cks:
        gate(max(F[ck]["B"]) == 48 and max(F[ck]["T"]) == 96 and opt[ck], f"CEILING {ck}")
        gate(set(opt[ck]) == {i for i in range(720) if F[ck]["B"][i] == 48} == {i for i in range(720) if F[ck]["T"][i] == 96}, f"OPT SET {ck}")
    res = {"cohort": cohort, "checkpoints": cks, "anchors": spec["anchors"], "optima": {ck: opt[ck] for ck in cks},
           "P0": {}, "E": {}, "F": {}, "H": {}, "I": {}}
    keep = {}
    for ck in cks:
        res["P0"][ck] = {}
        for f in ("B", "T"):
            r = p0(adj, F[ck][f], cay["C"][ck][f]["local_maxima"])
            gate(r["vertex_level"]["n_fixed_points"] == cay["C"][ck][f]["n_local_maxima"], "VL COUNT")
            r["anchor_flow"] = anchor_flow(r, spec["anchors"])
            keep[(ck, f)] = r
            res["P0"][ck][f] = {k: v for k, v in r.items() if not k.startswith("_")}
        # E
        Tb, Bs = 96, 48
        E = {}
        bpid = keep[(ck, "B")]["_pid"]
        bpl = keep[(ck, "B")]["_plats"]
        for name, fn in C.THRESH.items():
            S = {i for i in range(720) if fn(F[ck]["T"][i], F[ck]["B"][i], Tb, Bs)}
            e = {"set_size": len(S), "anchors": {a: radius(adj, S, v) for a, v in spec["anchors"].items()}}
            rads = [radius(adj, S, v)["radius"] for v in opt[ck]]
            rs = sorted(rads)
            n = len(rs)
            e["optima"] = {"n": n, "min": rs[0], "median": (rs[n // 2] if n % 2 else (rs[n // 2 - 1] + rs[n // 2]) / 2),
                           "max": rs[-1],
                           "per_optimum": [{"v": v, "radius": rd, "ceiling_plateau_size": len(bpl[bpid[v]])} for v, rd in zip(opt[ck], rads)]}
            E[name] = e
        res["E"][ck] = E
        res["F"][ck] = bank_f(adj, opt[ck], F[ck]["B"], F[ck]["T"])
        # H
        H = {}
        for f in ("B", "T"):
            x = F[ck][f]
            t = tv_q(edges, x)
            d = t.pop("d")
            sp = spectral(U, lam, x, CUTS)
            null = {"TV_norm": [], "Q_norm": [], "cut_fraction": {c: [] for c in CUTS}}
            for k in range(N_NULL):
                y = list(x)
                random.Random(f"morph-null-{ck}-{f}-{k}").shuffle(y)
                tn = tv_q(edges, y)
                null["TV_norm"].append(tn["TV_norm"])
                null["Q_norm"].append(tn["Q_norm"])
                sn = spectral(U, lam, y, CUTS)
                for c in CUTS:
                    null["cut_fraction"][c].append(sn["cuts"][c]["fraction"])
            H[f] = dict(t, **{
                "edge_abs_delta": {"mean": round(float(d.mean()), 4), "median": float(np.median(d)),
                                   "p90": float(np.percentile(d, 90)), "max": float(d.max()),
                                   "zero_fraction": round(float((d == 0).mean()), 6)},
                "spectral": sp,
                "null": {"n": N_NULL,
                         "TV_norm": {"min": min(null["TV_norm"]), "max": max(null["TV_norm"]),
                                     "n_null_below_field": sum(v < t["TV_norm"] for v in null["TV_norm"]),
                                     "field_below_all": all(t["TV_norm"] < v for v in null["TV_norm"])},
                         "Q_norm": {"min": min(null["Q_norm"]), "max": max(null["Q_norm"]),
                                    "field_below_all": all(t["Q_norm"] < v for v in null["Q_norm"])},
                         "cut_fraction": {c: {"min": min(null["cut_fraction"][c]), "max": max(null["cut_fraction"][c]),
                                              "field_above_all": all(sp["cuts"][c]["fraction"] > v for v in null["cut_fraction"][c])}
                                          for c in CUTS}}})
        res["H"][ck] = H
    # I
    for f in ("B", "T"):
        fr = pareto({i: tuple(F[ck][f][i] for ck in cks) for i in range(720)})
        gate(len(fr) == FRONT_SIZES[cohort][f], f"FRONT SIZE {cohort} {f} {len(fr)}")
        comps = components(adj, fr)
        vec = {i: tuple(F[ck][f][i] for ck in cks) for i in fr}
        # plateau-on-all-checkpoints inside the front: components under identical-vector edges
        ident_adj = {i: [j for j in adj[i] if j in vec and vec[j] == vec[i]] for i in fr}
        ic = components(ident_adj, fr)
        res["I"][f] = {"size": len(fr), "front": fr, "n_components": len(comps), "component_sizes": [len(c) for c in comps],
                       "connected": len(comps) == 1, "R488_on_front": 488 in fr, "R128_on_front": 128 in fr,
                       "internal_edges": sum(1 for (u, v) in edges if u in vec and v in vec),
                       "vertices_in_shared_plateau_on_all": sum(len(c) for c in ic if len(c) >= 2),
                       "shared_plateau_groups": [c for c in ic if len(c) >= 2]}
    # bars (discovery adjudicates; fresh = recurrence quantities)
    bars = {}
    Bcls = [res["P0"][ck]["B"]["class"] for ck in cks]
    Tcls = [res["P0"][ck]["T"]["class"] for ck in cks]
    bars["B1"] = {"pass": all(not res["P0"][ck][f]["floor_plateau_is_maximum"] and
                              res["P0"][ck][f]["n_quotient_maxima"] <= res["P0"][ck][f]["vertex_level"]["n_fixed_points"]
                              for ck in cks for f in ("B", "T"))}
    nflat = Bcls.count("FLAT-WITH-FEW-PEAKS")
    bars["B2"] = {"n_flat": nflat, "classes": Bcls,
                  "fires": "B-FLAT" if nflat == len(cks) else ("B-MIXED" if nflat > 0 else
                           ("B-RUGGED-PERSISTS" if "RUGGED" in Bcls else "B-INTERMEDIATE-ONLY"))}
    bars["B3"] = {"n_rugged": Tcls.count("RUGGED"), "classes": Tcls, "fires": Tcls.count("RUGGED") == len(cks)}
    lgb = {ck: res["P0"][ck]["B"]["largest_global_basin"] / 720 for ck in cks}
    bars["B4"] = {"largest_global_basin_fraction": lgb, "fires": all(v < 0.25 for v in lgb.values())}
    r488 = {ck: res["E"][ck]["MAJORITY"]["anchors"].get("R488") for ck in cks}
    bars["B5"] = {"R488_MAJORITY": r488, "fires": all(v["status"] == "IN-SET" and v["radius"] == 1 for v in r488.values())}
    sp = {ck: res["F"][ck]["spearman_B_vs_d"] for ck in cks}
    bars["B6"] = {"spearman_B_vs_d": sp, "fires": all(v <= -0.30 for v in sp.values())}
    tvb = {f"{ck}|{f}": res["H"][ck][f]["null"]["TV_norm"]["field_below_all"] for ck in cks for f in ("B", "T")}
    bars["B7"] = {"field_below_all_200": tvb, "fires": all(tvb.values())}
    bars["B8"] = {"B_front_connected": res["I"]["B"]["connected"], "n_components": res["I"]["B"]["n_components"],
                  "fires": "FRONT-CONNECTED" if res["I"]["B"]["connected"] else "FRONT-DISCONNECTED"}
    cut10 = {ck: res["H"][ck]["B"]["null"]["cut_fraction"]["10%"]["field_above_all"] for ck in cks}
    bars["prior7_low_mode_B"] = {"field_above_all_200": cut10, "all": all(cut10.values())}
    res["bars"] = bars
    return res


def main():
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN DRIFT {p}")
    gate("torch" not in sys.modules, "TORCH IMPORTED")
    START = start_provenance(["scratch/mathworld1_morphology.py", "scratch/mathworld1_cayley.py", "llmopt/lab/provenance.py"])
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for f in ("morphology_receipt.json", "DISCOVERY.json", "FRESH.json"):
        gate(not (OUTDIR / f).exists(), f"REFUSE OVERWRITE {f}")
    _, adj, edges, _ = C.build_graph()
    L = np.zeros((720, 720))
    for u, v in edges:
        L[u, v] -= 1
        L[v, u] -= 1
        L[u, u] += 1
        L[v, v] += 1
    lam, U = np.linalg.eigh(L)
    gate(abs(lam[0]) < 1e-9 and lam[1] > 1e-6 and np.allclose(U @ np.diag(lam) @ U.T, L, atol=1e-8), "EIGH")
    results = {}
    for cohort, spec in C.COHORTS.items():
        cay = json.load(open(f"logs/mathworld1/cayley/{cohort}.json"))
        res = analyze(cohort, spec, adj, edges, U, lam, cay)
        (OUTDIR / f"{cohort}.json").write_text(json.dumps(res, indent=1))
        results[cohort] = res
        print(f"[{cohort}] done {time.time() - t0:.1f}s", flush=True)
    receipt = {"prereg": "MATH-CYBER-1-RENDER-ATLAS-MORPHOLOGY-0" + ("-SMOKE" if SMOKE else ""), "pins": {p: fsha(p) for p in PINS},
               "graph": {"V": 720, "E": len(edges), "degree": 5, "diameter": 15,
                         "laplacian_eigenvalues_distinct": int(len(np.unique(np.round(lam, 9))))},
               "cuts": CUTS, "n_null": N_NULL, "smoke": SMOKE,
               "cohorts": {c: {"sha256": fsha(str(OUTDIR / f"{c}.json")), "checkpoints": C.COHORTS[c]["cks"],
                               "anchors": C.COHORTS[c]["anchors"], "bars": results[c]["bars"]} for c in results},
               "semantic_beyond_all_surface_identifiable": False,
               "wall_s": round(time.time() - t0, 1), "start": START, "completion_commit": completion_commit()}
    (OUTDIR / "morphology_receipt.json").write_text(json.dumps(receipt, indent=1))
    for c, res in results.items():
        print("==", c)
        for ck in res["checkpoints"]:
            for f in ("B", "T"):
                p = res["P0"][ck][f]
                print(ck, f, "plateaus", p["n_plateaus"], "qmax", p["n_quotient_maxima"], "interior", p["n_interior_quotient_maxima"],
                      p["class"], "vl", p["vertex_level"], "reach", p["starts_reaching_global"], "largest_global", p["largest_global_basin"],
                      "ceil_plats", p["n_ceiling_plateaus"])
        print("bars", json.dumps(res["bars"], default=str)[:3000])


if __name__ == "__main__":
    main()
