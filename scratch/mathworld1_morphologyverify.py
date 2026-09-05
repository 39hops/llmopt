"""Independent verifier for MATH-CYBER-1-RENDER-ATLAS-MORPHOLOGY-0 (prereg
RESULTS L65989). Shares no code with scratch/mathworld1_morphology.py:
rebuilds the Cayley graph from the manifest by inversion distance, finds
plateaus with its own union-find, recounts quotient maxima, interior
maxima and classes, re-runs quotient ascent, recomputes radii, the
distance profile Spearman, total variation and zero-change fractions,
the Laplacian spectrum through scipy.linalg.eigh (with orthonormality,
reconstruction and Parseval checks), the 200 string-seeded relabelings,
and the Pareto fronts, then compares every quantity with the instrument's
JSON. Writes logs/mathworld1/morphology/verify_receipt.json.

Usage:
    .venv/bin/python scratch/mathworld1_morphologyverify.py
"""
import hashlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.linalg

OUTDIR = Path("logs/mathworld1/morphology")
D = []
INV = {}   # prior-7 readouts: instrument basis, verifier basis, invariant eigenspace boundary


def chk(c, m):
    if not c:
        D.append(m)


def inv_dist(p, q):
    pos = {r: i for i, r in enumerate(q)}
    s = [pos[r] for r in p]
    return sum(1 for i in range(6) for j in range(i + 1, 6) if s[i] > s[j])


def avg_rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = np.array(avg_rank(x)), np.array(avg_rank(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    man = [json.loads(l) for l in open("logs/mathworld1/prband2atlas/atlas_manifest.jsonl")]
    roles = {m["atlas_index"]: tuple(m["roles"]) for m in man}
    adj = {i: [] for i in range(720)}
    for i in range(720):
        for j in range(i + 1, 720):
            if inv_dist(roles[i], roles[j]) == 1:
                adj[i].append(j)
                adj[j].append(i)
    edges = [(i, j) for i in range(720) for j in adj[i] if i < j]
    chk(len(edges) == 1800 and all(len(adj[i]) == 5 for i in adj), "graph")

    def bfs(src):
        d = {s: 0 for s in src}
        q = list(src)
        while q:
            u = q.pop(0)
            for v in adj[u]:
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        return d

    def plateaus(x):
        lab = {}
        out = []
        for s in range(720):
            if s in lab:
                continue
            comp, st = [s], [s]
            lab[s] = len(out)
            while st:
                u = st.pop()
                for v in adj[u]:
                    if v not in lab and x[v] == x[u]:
                        lab[v] = len(out)
                        comp.append(v)
                        st.append(v)
            out.append(sorted(comp))
        return out, lab

    L = np.zeros((720, 720))
    for u, v in edges:
        L[u, v] = L[v, u] = -1
        L[u, u] += 1
        L[v, v] += 1
    lam, U = scipy.linalg.eigh(L)
    chk(np.allclose(U.T @ U, np.eye(720), atol=1e-8) and np.allclose(U @ np.diag(lam) @ U.T, L, atol=1e-7), "eigh")
    rec_inputs = {}
    for cohort in ("DISCOVERY", "FRESH"):
        res = json.load(open(OUTDIR / f"{cohort}.json"))
        rec_inputs[cohort] = hashlib.sha256((OUTDIR / f"{cohort}.json").read_bytes()).hexdigest()
        table = {"DISCOVERY": "logs/mathworld1/prband2atlasscore/policy_table.jsonl",
                 "FRESH": "logs/mathworld1/prband2atlasfresh/policy_table.jsonl"}[cohort]
        rows = {r["atlas_index"]: r for r in map(json.loads, open(table))}
        cks = res["checkpoints"]
        anchors = res["anchors"]
        Fv = {ck: {f: [rows[i][ck][f] for i in range(720)] for f in ("B", "T")} for ck in cks}
        opt = {ck: [i for i in range(720) if Fv[ck]["B"][i] == 48] for ck in cks}
        for ck in cks:
            chk(res["optima"][ck] == opt[ck], f"{cohort} {ck} optima")
            for f in ("B", "T"):
                x = Fv[ck][f]
                P = res["P0"][ck][f]
                pl, lab = plateaus(x)
                sc = [x[p[0]] for p in pl]
                lo, hi = min(x), max(x)
                nb = [set() for _ in pl]
                for u in range(720):
                    for v in adj[u]:
                        if lab[u] != lab[v]:
                            nb[lab[u]].add(lab[v])
                qm = [k for k in range(len(pl)) if all(sc[j] < sc[k] for j in nb[k])]
                inter = [k for k in qm if lo < sc[k] < hi]
                cls = "FLAT-WITH-FEW-PEAKS" if len(inter) <= 10 else ("RUGGED" if len(inter) >= 30 else "INTERMEDIATE")
                chk(P["n_plateaus"] == len(pl), f"{cohort} {ck} {f} n_plateaus")
                chk(P["n_quotient_maxima"] == len(qm), f"{cohort} {ck} {f} n_qmax {P['n_quotient_maxima']} v {len(qm)}")
                chk(P["n_interior_quotient_maxima"] == len(inter), f"{cohort} {ck} {f} interior {P['n_interior_quotient_maxima']} v {len(inter)}")
                chk(P["class"] == cls, f"{cohort} {ck} {f} class")
                chk(P["floor_plateau_is_maximum"] is False and not any(sc[k] == lo for k in qm), f"{cohort} {ck} {f} floor max")
                chk(P["n_ceiling_plateaus"] == sum(sc[k] == hi for k in range(len(pl))), f"{cohort} {ck} {f} ceiling plateaus")
                chk(sorted(P["ceiling_plateau_sizes"]) == sorted(len(pl[k]) for k in range(len(pl)) if sc[k] == hi), f"{cohort} {ck} {f} ceiling sizes")
                chk(P["floor_total_size"] == sum(len(pl[k]) for k in range(len(pl)) if sc[k] == lo), f"{cohort} {ck} {f} floor size")
                chk(P["plateau_size_max"] == max(len(p) for p in pl) and P["n_singletons"] == sum(len(p) == 1 for p in pl), f"{cohort} {ck} {f} sizes")
                # quotient ascent
                term = {}
                for k in range(len(pl)):
                    p, n = k, 0
                    while True:
                        c = [j for j in nb[p] if sc[j] > sc[p]]
                        if not c:
                            break
                        b = max(sc[j] for j in c)
                        p = min((j for j in c if sc[j] == b), key=lambda j: pl[j][0])
                        n += 1
                    term[k] = (p, n)
                basin = Counter()
                for k in range(len(pl)):
                    basin[term[k][0]] += len(pl[k])
                reach = sum(len(pl[k]) for k in range(len(pl)) if sc[term[k][0]] == hi)
                chk(P["starts_reaching_global"] == reach, f"{cohort} {ck} {f} reach {P['starts_reaching_global']} v {reach}")
                chk(P["largest_basin"] == max(basin.values()), f"{cohort} {ck} {f} largest basin")
                chk(P["largest_global_basin"] == max(basin.get(k, 0) for k in range(len(pl)) if sc[k] == hi), f"{cohort} {ck} {f} largest global basin")
                chk(P["max_path_len"] == max(n for _, n in term.values()), f"{cohort} {ck} {f} max path")
                qmset = {(pl[k][0], sc[k], len(pl[k]), basin.get(k, 0)) for k in qm}
                chk({(q["min_index"], q["score"], q["size"], q["basin_vertices"]) for q in P["quotient_maxima"]} == qmset, f"{cohort} {ck} {f} qmax list")
                for a, v in anchors.items():
                    t = term[lab[v]][0]
                    af = P["anchor_flow"][a]
                    chk(af["terminal_min_index"] == pl[t][0] and af["terminal_score"] == sc[t], f"{cohort} {ck} {f} anchor {a}")
                # vertex-level decomposition v the cayley receipts (independent read)
                cay = json.load(open(f"logs/mathworld1/cayley/{cohort}.json"))["C"][ck][f]["local_maxima"]
                chk(P["vertex_level"] == {"n_fixed_points": len(cay), "on_floor": sum(m["score"] == lo for m in cay),
                                          "above_floor": sum(m["score"] != lo for m in cay),
                                          "interior": sum(lo < m["score"] < hi for m in cay),
                                          "at_ceiling": sum(m["score"] == hi for m in cay)}, f"{cohort} {ck} {f} vertex level")
                chk(len(qm) <= len(cay), f"{cohort} {ck} {f} bound")
                # H
                dd = np.array([abs(x[u] - x[v]) for u, v in edges], dtype=float)
                H = res["H"][ck][f]
                tvn = dd.sum() / (1800 * (hi - lo))
                chk(abs(H["TV_norm"] - tvn) < 1e-12, f"{cohort} {ck} {f} TV_norm")
                chk(abs(H["Q_norm"] - (dd ** 2).sum() / (1800 * np.var(x))) < 1e-9, f"{cohort} {ck} {f} Q_norm")
                chk(abs(H["edge_abs_delta"]["zero_fraction"] - (dd == 0).mean()) < 1e-6, f"{cohort} {ck} {f} zero fraction")
                chk(H["edge_abs_delta"]["max"] == dd.max(), f"{cohort} {ck} {f} max delta")
                xc = np.array(x, dtype=float) - np.mean(x)
                c = U.T @ xc
                chk(abs((c ** 2).sum() - (xc ** 2).sum()) < 1e-6 and abs(c[0]) < 1e-9, f"{cohort} {ck} {f} parseval")
                e = c[1:] ** 2
                cum = np.cumsum(e) / e.sum()
                # Energy inside a degenerate eigenspace is basis-invariant only as
                # a whole: a cut that splits an eigenspace gives a basis-dependent
                # fraction (numpy and scipy eigh pick different bases), so the
                # fraction is checked exactly only at non-splitting cuts and is
                # otherwise checked to lie between the invariant boundary values.
                for name, k in (("10%", 72), ("25%", 180), ("50%", 360)):
                    cut = H["spectral"]["cuts"][name]
                    split = abs(lam[k] - lam[k + 1]) < 1e-9
                    chk(cut["k"] == k and abs(cut["eigenvalue_at_cut"] - lam[k]) < 1e-6, f"{cohort} {ck} {f} eig {name}")
                    chk(cut["splits_degenerate_eigenspace"] == split, f"{cohort} {ck} {f} split {name}")
                    if not split:
                        chk(abs(cut["fraction"] - cum[k - 1]) < 1e-9, f"{cohort} {ck} {f} cut {name}")
                    else:
                        lo_k = k
                        while lo_k > 1 and abs(lam[lo_k] - lam[k]) < 1e-9:
                            lo_k -= 1
                        hi_k = k
                        while hi_k < 719 and abs(lam[hi_k + 1] - lam[k]) < 1e-9:
                            hi_k += 1
                        chk(cut["boundary_below"]["k"] == lo_k and abs(cut["boundary_below"]["fraction"] - cum[lo_k - 1]) < 1e-9, f"{cohort} {ck} {f} boundary below {name}")
                        chk(cut["boundary_above"]["k"] == hi_k and abs(cut["boundary_above"]["fraction"] - cum[hi_k - 1]) < 1e-9, f"{cohort} {ck} {f} boundary above {name}")
                        chk(cum[lo_k - 1] - 1e-9 <= cut["fraction"] <= cum[hi_k - 1] + 1e-9, f"{cohort} {ck} {f} cut {name} outside boundaries")
                # null (same string-seed law)
                below, above10, above_inv = 0, 0, 0
                k_inv = 72
                while k_inv > 1 and abs(lam[k_inv] - lam[72]) < 1e-9:
                    k_inv -= 1          # invariant boundary below the 10 % cut
                tvs = []
                for kk in range(200):
                    y = list(x)
                    random.Random(f"morph-null-{ck}-{f}-{kk}").shuffle(y)
                    dy = np.array([abs(y[u] - y[v]) for u, v in edges], dtype=float)
                    t = dy.sum() / (1800 * (hi - lo))
                    tvs.append(t)
                    below += t < tvn
                    cy = U.T @ (np.array(y, dtype=float) - np.mean(y))
                    ey = cy[1:] ** 2
                    above10 += (ey[:72].sum() / ey.sum()) < cum[71]
                    above_inv += (ey[:k_inv].sum() / ey.sum()) < cum[k_inv - 1]
                chk(H["null"]["TV_norm"]["n_null_below_field"] == below, f"{cohort} {ck} {f} null below {H['null']['TV_norm']['n_null_below_field']} v {below}")
                chk(H["null"]["TV_norm"]["field_below_all"] == (below == 0), f"{cohort} {ck} {f} null all")
                chk(abs(H["null"]["TV_norm"]["min"] - min(tvs)) < 1e-12, f"{cohort} {ck} {f} null min")
                # basis-dependent at a splitting cut: recorded, not asserted
                INV.setdefault(cohort, {})[f"{ck}|{f}"] = {"instrument_cut10_above_all": H["null"]["cut_fraction"]["10%"]["field_above_all"],
                                                        "verifier_basis_cut10_above_all": bool(above10 == 200),
                                                        "invariant_boundary_k": int(k_inv), "invariant_boundary_above_all": bool(above_inv == 200)}
            # E
            for name, fn in (("NEAR", lambda T, B: T >= 92 and B >= 46), ("MAJORITY", lambda T, B: T >= 72 and B >= 24),
                             ("STRONG", lambda T, B: T >= 84 and B >= 36)):
                S = {i for i in range(720) if fn(Fv[ck]["T"][i], Fv[ck]["B"][i])}
                E = res["E"][ck][name]
                chk(E["set_size"] == len(S), f"{cohort} {ck} {name} set size")
                for a, v in anchors.items():
                    d = bfs([v])
                    got = E["anchors"][a]
                    if v in S:
                        chk(got["status"] == "IN-SET" and got["radius"] == min(d[u] for u in range(720) if u not in S), f"{cohort} {ck} {name} radius {a}")
                    else:
                        chk(got["status"] == "NOT-IN-SET" and got["distance_to_set"] == min(d[u] for u in S), f"{cohort} {ck} {name} dist {a}")
                rads = sorted(min(bfs([v])[u] for u in range(720) if u not in S) for v in opt[ck])
                n = len(rads)
                med = rads[n // 2] if n % 2 else (rads[n // 2 - 1] + rads[n // 2]) / 2
                chk(E["optima"]["min"] == rads[0] and E["optima"]["max"] == rads[-1] and E["optima"]["median"] == med and E["optima"]["n"] == n, f"{cohort} {ck} {name} optimum radii")
            # F
            d = bfs(opt[ck])
            dv = [d[i] for i in range(720)]
            Fr = res["F"][ck]
            chk(abs(Fr["spearman_B_vs_d"] - spearman(Fv[ck]["B"], dv)) < 1e-5, f"{cohort} {ck} spearman B")
            chk(abs(Fr["spearman_T_vs_d"] - spearman(Fv[ck]["T"], dv)) < 1e-5, f"{cohort} {ck} spearman T")
            chk(Fr["max_distance"] == max(dv), f"{cohort} {ck} max distance")
            for s, sh in Fr["shells"].items():
                vs = [i for i in range(720) if d[i] == int(s)]
                chk(sh["B"]["n"] == len(vs) and abs(sh["B"]["mean"] - np.mean([Fv[ck]["B"][i] for i in vs])) < 1e-3, f"{cohort} {ck} shell {s}")
        # I
        for f in ("B", "T"):
            vec = {i: tuple(Fv[ck][f][i] for ck in cks) for i in range(720)}
            fr = sorted(i for i, v in vec.items() if not any(all(w[k] >= v[k] for k in range(len(v))) and w != v for j, w in vec.items() if j != i))
            I = res["I"][f]
            chk(I["front"] == fr and I["size"] == len(fr), f"{cohort} front {f}")
            frs = set(fr)
            seen, comps = set(), []
            for s in fr:
                if s in seen:
                    continue
                c, st = [s], [s]
                seen.add(s)
                while st:
                    u = st.pop()
                    for v in adj[u]:
                        if v in frs and v not in seen:
                            seen.add(v)
                            c.append(v)
                            st.append(v)
                comps.append(len(c))
            chk(sorted(comps, reverse=True) == I["component_sizes"] and I["n_components"] == len(comps), f"{cohort} front comps {f}")
            chk(I["R488_on_front"] == (488 in frs) and I["R128_on_front"] == (128 in frs), f"{cohort} front anchors {f}")
        # bars recomputed from the verified fields
        bars = res["bars"]
        Bcls = [res["P0"][ck]["B"]["class"] for ck in cks]
        chk(bars["B2"]["n_flat"] == Bcls.count("FLAT-WITH-FEW-PEAKS"), f"{cohort} B2")
        chk(bars["B4"]["fires"] == all(res["P0"][ck]["B"]["largest_global_basin"] / 720 < 0.25 for ck in cks), f"{cohort} B4")
        chk(bars["B6"]["fires"] == all(res["F"][ck]["spearman_B_vs_d"] <= -0.30 for ck in cks), f"{cohort} B6")
        chk(bars["B7"]["fires"] == all(res["H"][ck][f]["null"]["TV_norm"]["field_below_all"] for ck in cks for f in ("B", "T")), f"{cohort} B7")
        print(cohort, "checked; discrepancies so far", len(D), flush=True)
    pins = {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in (
        "logs/mathworld1/prband2atlas/atlas_manifest.jsonl", "logs/mathworld1/prband2atlasscore/policy_table.jsonl",
        "logs/mathworld1/prband2atlasfresh/policy_table.jsonl", "logs/mathworld1/cayley/DISCOVERY.json",
        "logs/mathworld1/cayley/FRESH.json")}
    rec = {"verdict": "VERIFIED" if not D else "DISCREPANCIES", "discrepancies": D[:40], "n_discrepancies": len(D),
           "input_pins": pins, "inputs": rec_inputs, "prior7_spectral": INV,
           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "verifier_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
    (OUTDIR / "verify_receipt.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1)[:2000])


if __name__ == "__main__":
    main()
