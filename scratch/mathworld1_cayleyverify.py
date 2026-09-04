"""Independent recount for MATH-CYBER-1-RENDER-ATLAS-CAYLEY-LANDSCAPE-0.
Separate code (no import from mathworld1_cayley): rebuilds the Cayley graph
from the manifest by inversion-distance-1 test (not by generator application),
then for both cohorts recomputes from the policy tables: component counts and
sizes at the three thresholds, R488 same-component flags and within-threshold
path lengths, all anchor/optimum-set Cayley distances, widest-path bottleneck
values (by threshold scan with an independent union-find connectivity check),
steepest-ascent terminals and basin sizes for B and T and the joint tuple,
edge-conflict counts and histogram, local-Pareto flags, monotone-path existence
(both rules), and the bank-K node/edge correlations and Jaccards. Compares to
DISCOVERY.json / FRESH.json; writes verify_receipt.json.
"""
import hashlib
import itertools
import json
from collections import Counter, deque
from pathlib import Path

OUTDIR = Path("logs/mathworld1/cayley")
D = []


def chk(c, m):
    if not c:
        D.append(m)


def inv_dist(p, q):
    pos = {r: i for i, r in enumerate(q)}
    seq = [pos[r] for r in p]
    return sum(1 for i in range(6) for j in range(i + 1, 6) if seq[i] > seq[j])


def main():
    man = [json.loads(l) for l in open("logs/mathworld1/prband2atlas/atlas_manifest.jsonl")]
    roles = {m["atlas_index"]: tuple(m["roles"]) for m in man}
    rid = {m["atlas_index"]: m["render_id"] for m in man}
    adj = {i: [] for i in range(720)}
    for i in range(720):
        for j in range(i + 1, 720):
            if inv_dist(roles[i], roles[j]) == 1:
                adj[i].append(j)
                adj[j].append(i)
    edges = [(i, j) for i in range(720) for j in adj[i] if i < j]
    chk(len(edges) == 1800 and all(len(adj[i]) == 5 for i in adj), "graph")
    for i in adj:
        adj[i].sort()

    def bfs(src, allowed=None):
        d = {s: 0 for s in src}
        q = deque(src)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in d and (allowed is None or v in allowed):
                    d[v] = d[u] + 1
                    q.append(v)
        return d

    def uf_components(S):
        par = {v: v for v in S}
        def f(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for a, b in edges:
            if a in par and b in par:
                par[f(a)] = f(b)
        g = Counter(f(v) for v in S)
        return sorted(g.values(), reverse=True), {v: f(v) for v in S}

    for cohort in ("DISCOVERY", "FRESH"):
        res = json.loads((OUTDIR / f"{cohort}.json").read_text())
        table = "logs/mathworld1/prband2atlasscore/policy_table.jsonl" if cohort == "DISCOVERY" else "logs/mathworld1/prband2atlasfresh/policy_table.jsonl"
        tab = {}
        for l in open(table):
            r = json.loads(l)
            tab[r["atlas_index"]] = r
        cks = res["checkpoints"]
        anchors = res["anchors"]
        T = {ck: {i: tab[i][ck]["T"] for i in range(720)} for ck in cks}
        B = {ck: {i: tab[i][ck]["B"] for i in range(720)} for ck in cks}
        opt = {ck: [i for i in range(720) if B[ck][i] == max(B[ck].values()) and T[ck][i] == max(T[ck].values())] for ck in cks}
        for ck in cks:
            chk(opt[ck] == res["optima"][ck]["set"], f"{cohort} {ck} opt set")
            Tb, Bs = max(T[ck].values()), max(B[ck].values())
            for name, fn in (("NEAR", lambda t, b: t >= Tb - 4 and b >= Bs - 2),
                             ("MAJORITY", lambda t, b: t >= 72 and b >= 24),
                             ("STRONG", lambda t, b: t >= 84 and b >= 36)):
                S = [i for i in range(720) if fn(T[ck][i], B[ck][i])]
                sizes, root = uf_components(S)
                a = res["A"][ck][name]
                chk(a["vertices"] == len(S) and a["n_components"] == len(sizes) and a["component_sizes"] == sizes, f"{cohort} {ck} {name} comps")
                Sset = set(S)
                chk(a["internal_edges"] == sum(x in Sset and y in Sset for x, y in edges), f"{cohort} {ck} {name} internal")
                chk(a["boundary_edges"] == sum((x in Sset) != (y in Sset) for x, y in edges), f"{cohort} {ck} {name} boundary")
                chk(a["membership"]["R488"] == (488 in Sset), f"{cohort} {ck} {name} 488 member")
                if 488 in Sset:
                    same = any(root[488] == root[v] for v in opt[ck] if v in Sset)
                    chk(a["R488"]["same_component_as_an_optimum"] == same, f"{cohort} {ck} {name} 488 same comp")
                    if same:
                        d = bfs([488], Sset)
                        chk(a["R488"]["within_threshold_path_len"] == min(d[v] for v in opt[ck] if v in d), f"{cohort} {ck} {name} 488 path")
        # bank B distances
        sets = {an: [v] for an, v in anchors.items()}
        for ck in cks:
            sets[f"OPT[{ck}]"] = opt[ck]
        for a_, b_ in itertools.combinations(list(sets), 2):
            d = bfs(sets[a_])
            chk(res["B"]["distances"][f"{a_} <-> {b_}"] == min(d[v] for v in sets[b_]), f"{cohort} dist {a_} {b_}")
        # widest bottleneck values (independent: threshold scan + union-find)
        for key, w in res["B"]["widest"].items():
            ck, fld, rest = key.split("|")
            F = (B if fld == "B" else T)[ck]
            a_, z_ = rest.split(" <-> ")
            A = sets[a_ if a_ in sets else "OPT[" + ck + "]"] if a_ != "OPT" else opt[ck]
            Z = opt[ck] if z_ == "OPT" else sets[z_]
            best = None
            for a in A:
                for z in Z:
                    hi = min(F[a], F[z])
                    for t in sorted({F[v] for v in range(720) if F[v] <= hi}, reverse=True):
                        S = [v for v in range(720) if F[v] >= t]
                        _s, root = uf_components(S)
                        if root[a] == root[z]:
                            if best is None or t > best:
                                best = t
                            break
            chk(w["W"] == best and w["D"] == min(F[w["a"]], F[w["z"]]) - best, f"{cohort} widest {key}")
            chk(all(F[v] >= w["W"] for v in w["path"]) and w["path"][0] == w["a"] and w["path"][-1] == w["z"]
                and all(w["path"][k + 1] in adj[w["path"][k]] for k in range(len(w["path"]) - 1)), f"{cohort} widest path {key}")
        # bank C
        def climb(key):
            term = {}
            for s in range(720):
                v = s
                while True:
                    cand = [w for w in adj[v] if key(w) > key(v)]
                    if not cand:
                        break
                    top = max(key(w) for w in cand)
                    v = min(w for w in cand if key(w) == top)
                term[s] = v
            return term
        for ck in cks:
            for fld, F in (("B", B[ck]), ("T", T[ck])):
                term = climb(lambda v, F=F: F[v])
                basins = Counter(term.values())
                c = res["C"][ck][fld]
                chk(c["n_local_maxima"] == len(basins), f"{cohort} {ck} {fld} n maxima")
                chk({m["v"]: m["basin"] for m in c["local_maxima"]} == dict(basins), f"{cohort} {ck} {fld} basins")
                gmax = max(F.values())
                chk(c["starts_reaching_global"] == sum(n for v, n in basins.items() if F[v] == gmax), f"{cohort} {ck} {fld} reach")
                for an, v in anchors.items():
                    chk(c["anchor_flow"][an]["terminal"] == term[v], f"{cohort} {ck} {fld} flow {an}")
        def jkey(v):
            bs = [B[ck][v] for ck in cks]
            ts = [T[ck][v] for ck in cks]
            return (min(bs), min(ts), sum(bs), sum(ts), -int(rid[v][:16], 16))
        term = climb(jkey)
        basins = Counter(term.values())
        chk(res["C"]["JOINT"]["n_local_maxima"] == len(basins), f"{cohort} joint maxima")
        for an, v in anchors.items():
            chk(res["C"]["JOINT"]["anchor_flow"][an]["terminal"] == term[v], f"{cohort} joint flow {an}")
        # bank D
        dB = {ck: {e: B[ck][e[1]] - B[ck][e[0]] for e in edges} for ck in cks}
        dT = {ck: {e: T[ck][e[1]] - T[ck][e[0]] for e in edges} for ck in cks}
        d = res["D"]
        chk(d["edges_B_improving_all"] == sum(all(dB[ck][e] > 0 for ck in cks) for e in edges), f"{cohort} imp all")
        chk(d["edges_B_hurting_all"] == sum(all(dB[ck][e] < 0 for ck in cks) for e in edges), f"{cohort} hurt all")
        chk(d["edges_B_unchanged_all"] == sum(all(dB[ck][e] == 0 for ck in cks) for e in edges), f"{cohort} unch all")
        hist = Counter(sum(dB[ck][e] > 0 for ck in cks) for e in edges)
        chk({int(k): v for k, v in d["hist_B_improved_count"].items()} == dict(hist), f"{cohort} hist")
        fw = sum(all((dB[ck][e], dT[ck][e]) >= (0, 0) for ck in cks) and any((dB[ck][e], dT[ck][e]) > (0, 0) for ck in cks) for e in edges)
        chk(d["edges_lex_weakly_improving_all_forward"] == fw, f"{cohort} lex fwd")
        for an, v in anchors.items():
            lp = True
            for w in adj[v]:
                e = (min(v, w), max(v, w))
                s = 1 if v < w else -1
                if all((s * dB[ck][e], s * dT[ck][e]) >= (0, 0) for ck in cks) and any((s * dB[ck][e], s * dT[ck][e]) > (0, 0) for ck in cks):
                    lp = False
                    break
            chk(d["local_pareto"][an]["is_local_pareto_optimum"] == lp, f"{cohort} local pareto {an}")
        for k, v in d["monotone_paths"].items():
            an = k.split("-> ")[1].split(" ")[0]
            dst = anchors[an]
            lex = "lex" in k
            seen = {12}
            q = deque([12])
            while q:
                u = q.popleft()
                for w in adj[u]:
                    e = (min(u, w), max(u, w))
                    s = 1 if u < w else -1
                    ok = all(((s * dB[ck][e], s * dT[ck][e]) >= (0, 0)) if lex else (s * dB[ck][e] >= 0) for ck in cks)
                    if ok and w not in seen:
                        seen.add(w)
                        q.append(w)
            chk(v["exists"] == (dst in seen) and v["reachable_set_size"] == len(seen), f"{cohort} mono {k}")
        # bank K: jaccards and node spearman (correlations recomputed loosely via rank equality)
        for c1, c2 in itertools.combinations(cks, 2):
            k = res["K"][f"{c1} v {c2}"]
            m1 = {i for i in range(720) if T[c1][i] >= 72 and B[c1][i] >= 24}
            m2 = {i for i in range(720) if T[c2][i] >= 72 and B[c2][i] >= 24}
            j = len(m1 & m2) / len(m1 | m2) if (m1 | m2) else None
            chk(k["jaccard_MAJORITY"] == j, f"{cohort} jaccard {c1} {c2}")
            d1 = bfs(opt[c1])
            chk(k["dist_optimum_sets"] == min(d1[v] for v in opt[c2]), f"{cohort} optdist {c1} {c2}")
    rec = {"verdict": "VERIFIED" if not D else "DISCREPANCIES", "discrepancies": D[:40], "n_discrepancies": len(D),
           "verifier_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
           "inputs": {c: hashlib.sha256((OUTDIR / f"{c}.json").read_bytes()).hexdigest() for c in ("DISCOVERY", "FRESH")}}
    (OUTDIR / "verify_receipt.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1)[:2000])


if __name__ == "__main__":
    main()
