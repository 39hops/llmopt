"""Independent verifier for MATH-CYBER-1-RENDER-ATLAS-DECISION-ATLAS-0
(prereg RESULTS L66047). Shares no code with
scratch/mathworld1_decisionatlas.py: re-hashes the twelve streams, parses
them with its own loop and its own strict top-1 (exact float ties = TIE),
rebuilds the Cayley graph by inversion distance, recounts per-state
boundary edges, invariant states, correct-set components (own DFS),
wrong-action census, per-edge flip counts and histogram, zero-flip
fraction, theta medians, pair-partner and non-partner Jaccard medians,
anchor action strings, and the policy-table T / B / top_census gates, then
compares every quantity with the instrument's JSON. Refuses to overwrite
its receipt. Writes logs/mathworld1/decisionatlas/verify_receipt.json.

Usage:
    .venv/bin/python scratch/mathworld1_decisionatlasverify.py
"""
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

OUTDIR = Path("logs/mathworld1/decisionatlas")
STREAMS = {"DISCOVERY": "logs/mathworld1/prband2atlasscore", "FRESH": "logs/mathworld1/prband2atlasfresh"}
CODE = {"i_sum": "S", "A0": "A", "B0": "B", "I0/t5": "I"}
D = []


def chk(c, m):
    if not c:
        D.append(m)


def inv_dist(p, q):
    pos = {r: i for i, r in enumerate(q)}
    s = [pos[r] for r in p]
    return sum(1 for i in range(6) for j in range(i + 1, 6) if s[i] > s[j])


def med(v):
    v = sorted(v)
    n = len(v)
    return None if n == 0 else (v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2)


def main():
    chk(not (OUTDIR / "verify_receipt.json").exists(), "REFUSE OVERWRITE verify_receipt.json")
    if D:
        raise SystemExit(D[-1])
    inst = json.load(open(OUTDIR / "decisionatlas_receipt.json"))
    chk(inst.get("smoke") is False and inst["prereg"] == "MATH-CYBER-1-RENDER-ATLAS-DECISION-ATLAS-0", "instrument receipt identity")
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
    inputs = {}
    for cohort in ("DISCOVERY", "FRESH"):
        res = json.load(open(OUTDIR / f"{cohort}.json"))
        inputs[cohort] = hashlib.sha256((OUTDIR / f"{cohort}.json").read_bytes()).hexdigest()
        chk(inst["cohorts"][cohort]["sha256"] == inputs[cohort], f"{cohort} cohort sha v instrument receipt")
        agg = json.load(open(f"{STREAMS[cohort]}/aggregate.json"))["chunk_shas"]
        table = f"{STREAMS[cohort]}/policy_table.jsonl"
        pol = {r["atlas_index"]: r for r in map(json.loads, open(table))}
        for ck in res["checkpoints"]:
            R = res["checkpoint"][ck]
            seed, rep = ck.split("|")
            path = f"{STREAMS[cohort]}/chunk_{seed}_{rep}/scores.jsonl"
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for blk in iter(lambda: f.read(1 << 24), b""):
                    h.update(blk)
            chk(h.hexdigest() == agg[ck] == inst["streams"][ck]["sha256"], f"{ck} stream sha")
            sums = defaultdict(dict)
            meta = {}
            n_full = n_mask = dup = 0
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if r["arm"] == "MASK0":
                        n_mask += 1
                        continue
                    n_full += 1
                    s = r["state"]
                    meta.setdefault(s, (r["pair_id"], r["theta"], tuple(r["gold"])))
                    d = sums[(r["atlas_index"], s)]
                    if r["name"] in d:
                        dup += 1
                        chk(d[r["name"]] == r["sum"], f"{ck} duplicate differs")
                    d[r["name"]] = r["sum"]
            chk(R["row_census"] == {"FULL": n_full, "MASK0": n_mask, "replay_duplicates": dup}, f"{ck} row census")
            chk(n_full == 276864 and n_mask == 6144 and dup == 384, f"{ck} row counts")
            S = sorted(meta)
            chk(S == list(range(96)), f"{ck} states")
            gold = ""
            for s in S:
                gold += "A" if meta[s][1] == "SIN_LOW" else "B"
            chk(R["gold"] == gold, f"{ck} gold string")
            act = {}
            for a in range(720):
                row = []
                for s in S:
                    d = sums[(a, s)]
                    chk(len(d) == 4, f"{ck} four candidates {a} {s}")
                    best = max(d.values())
                    win = [k for k, v in d.items() if v == best]
                    row.append(CODE[win[0]] if len(win) == 1 else "T")
                act[a] = "".join(row)
            chk(R["decision_matrix"] == {str(a): act[a] for a in range(720)}, f"{ck} decision matrix")
            # policy-table gates
            for a in range(720):
                corr = [act[a][i] == gold[i] for i in range(96)]
                chk(sum(corr) == pol[a][ck]["T"], f"{ck} T {a}")
                pr = defaultdict(list)
                for i, s in enumerate(S):
                    pr[meta[s][0]].append(corr[i])
                chk(sum(all(v) for v in pr.values()) == pol[a][ck]["B"], f"{ck} B {a}")
                cen = Counter(act[a])
                exp = {CODE[k]: v for k, v in pol[a][ck]["top_census"].items()}
                chk({k: v for k, v in cen.items() if k != "T"} == exp and cen.get("T", 0) == pol[a][ck]["ties"], f"{ck} census {a}")
            # per state
            flip_sets = defaultdict(set)
            n_inv = frag = 0
            bfr = []
            theta_b = {"SIN_LOW": [], "COS_LOW": []}
            theta_w = {"SIN_LOW": Counter(), "COS_LOW": Counter()}
            for i, s in enumerate(S):
                P = R["per_state"][i]
                bnd = [(u, v) for u, v in edges if act[u][i] != act[v][i]]
                for e in bnd:
                    flip_sets[i].add(e)
                correct = {a for a in range(720) if act[a][i] == gold[i]}
                seen, comps = set(), 0
                for a in sorted(correct):
                    if a in seen:
                        continue
                    comps += 1
                    st = [a]
                    seen.add(a)
                    while st:
                        u = st.pop()
                        for v in adj[u]:
                            if v in correct and v not in seen:
                                seen.add(v)
                                st.append(v)
                wrong = Counter(act[a][i] for a in range(720) if act[a][i] != gold[i])
                chk(P["state"] == s and P["boundary_edges"] == len(bnd) and P["n_correct"] == len(correct) and P["correct_components"] == comps, f"{ck} state {s} counts")
                chk(P["wrong_census"] == dict(wrong), f"{ck} state {s} wrong census")
                if wrong:
                    top = max(wrong.values())
                    chk(wrong[P["top_wrong"]] == top, f"{ck} state {s} top wrong")
                chk(P["invariant"] == (len(bnd) == 0), f"{ck} state {s} invariant")
                n_inv += len(bnd) == 0
                frag += comps > 1
                bfr.append(len(bnd) / 1800)
                theta_b[meta[s][1]].append(len(bnd) / 1800)
                theta_w[meta[s][1]].update(wrong)
            chk(R["n_invariant"] == n_inv and R["states_fragmented"] == frag, f"{ck} invariant/fragmented")
            chk(abs(R["boundary_fraction"]["median"] - med(bfr)) < 1e-12, f"{ck} boundary median")
            for th in theta_b:
                chk(abs(R["theta"][th]["boundary_fraction"]["median"] - med(theta_b[th])) < 1e-12, f"{ck} {th} boundary median")
                chk(R["theta"][th]["wrong_census_total"] == dict(theta_w[th]), f"{ck} {th} wrong total")
                if theta_w[th]:
                    chk(theta_w[th][R["theta"][th]["top_wrong"]] == max(theta_w[th].values()), f"{ck} {th} top wrong")
            # edges
            flips = {e: sum(act[e[0]][i] != act[e[1]][i] for i in range(96)) for e in edges}
            hist = Counter(flips.values())
            chk(R["flip_hist"] == {str(k): v for k, v in sorted(hist.items())}, f"{ck} flip hist")
            chk(abs(R["mean_flip_fraction"] - sum(flips.values()) / 1800 / 96) < 1e-12, f"{ck} mean flip")
            chk(abs(R["zero_flip_fraction"] - hist.get(0, 0) / 1800) < 1e-12, f"{ck} zero flip")
            chk(R["max_flip"] == max(flips.values()), f"{ck} max flip")
            # coflip
            by_pair = defaultdict(list)
            for i, s in enumerate(S):
                by_pair[meta[s][0]].append(i)
            pset = {tuple(sorted(v)) for v in by_pair.values()}
            chk(len(pset) == 48 and all(len(p) == 2 for p in pset), f"{ck} pairs")

            def jac(a, b):
                u = flip_sets[a] | flip_sets[b]
                return len(flip_sets[a] & flip_sets[b]) / len(u) if u else None
            partner = [x for x in (jac(a, b) for a, b in pset) if x is not None]
            non = [x for x in (jac(i, j) for i in range(96) for j in range(i + 1, 96) if (i, j) not in pset) if x is not None]
            chk(R["coflip"]["partner"]["n"] == len(partner) and abs(R["coflip"]["partner"]["median"] - med(partner)) < 1e-12, f"{ck} partner median")
            chk(R["coflip"]["non_partner"]["n"] == len(non) and abs(R["coflip"]["non_partner"]["median"] - med(non)) < 1e-12, f"{ck} non-partner median")
            chk(R["coflip"]["non_partner_above_partner_median"] == sum(x > med(partner) for x in non), f"{ck} non above partner")
            for name, a in res["anchors"].items():
                chk(R["anchors"][name]["actions"] == act[a], f"{ck} anchor {name}")
            print(ck, "checked; discrepancies so far", len(D), flush=True)
        # bars recomputed from verified fields
        b = res["bars"]
        cks = res["checkpoints"]
        chk(b["B2"]["fires"] == all(res["checkpoint"][ck]["n_invariant"] == 0 for ck in cks), f"{cohort} B2")
        chk(b["B3"]["fires"] == all(res["checkpoint"][ck]["mean_flip_fraction"] < 0.15 for ck in cks), f"{cohort} B3")
        chk(b["B4"]["fires"] == all(res["checkpoint"][ck]["theta"]["SIN_LOW"]["boundary_fraction"]["median"] > res["checkpoint"][ck]["theta"]["COS_LOW"]["boundary_fraction"]["median"] for ck in cks), f"{cohort} B4")
        chk(b["B5"]["fires"] == all((res["checkpoint"][ck]["theta"]["SIN_LOW"]["top_wrong"], res["checkpoint"][ck]["theta"]["COS_LOW"]["top_wrong"]) == ("B", "A") for ck in cks), f"{cohort} B5")
        chk(b["B7"]["fires"] == all(res["checkpoint"][ck]["states_fragmented"] >= 48 for ck in cks), f"{cohort} B7")
        if "B8" in b:
            H = json.load(open(f"logs/mathworld1/morphology/{cohort}.json"))["H"]
            chk(b["B8"]["fires"] == all(res["checkpoint"][ck]["zero_flip_fraction"] >= 0.8 * H[ck]["T"]["edge_abs_delta"]["zero_fraction"] for ck in cks), f"{cohort} B8")
    rec = {"prereg": "MATH-CYBER-1-RENDER-ATLAS-DECISION-ATLAS-0", "verdict": "VERIFIED" if not D else "DISCREPANCIES",
           "discrepancies": D[:40], "n_discrepancies": len(D), "inputs": inputs,
           "instrument_receipt_sha256": hashlib.sha256((OUTDIR / "decisionatlas_receipt.json").read_bytes()).hexdigest(),
           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "verifier_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
    (OUTDIR / "verify_receipt.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1)[:1500])


if __name__ == "__main__":
    main()
