"""MATH-CYBER-1-RENDER-ATLAS-DECISION-ATLAS-0 — bank J of L65753 (prereg
RESULTS L66047): the state-level decision-boundary atlas of the twelve
booked render atlases, read from the untracked sha-anchored chunk streams.
For every checkpoint, renderer and frozen state the top action (strict
top-1 over the four candidate total-sums, exact tie = TIE) is recovered,
gated against the verifier-checked policy tables (T and top_census, 720 /
720), and analysed on the Cayley graph: per-state boundary edges,
render-invariant states, correct-set components, SIN_LOW v COS_LOW
anatomy, per-edge flip counts, pair-partner co-flip Jaccard, anchor
readouts. Streams are re-hashed at entry against the booked aggregates,
opened read-only, never rewritten. No model, no logit; torch is never
imported (the strict top-1 rule is a verbatim local copy of
scratch/mathworld1_prband2score.top1_of, whose module imports torch; the
copy is checked against that module's source text at entry).

Usage:
    .venv/bin/python scratch/mathworld1_decisionatlas.py
    DA_SMOKE=1 .venv/bin/python scratch/mathworld1_decisionatlas.py   # first 2,000 rows of one stream
"""
import ast
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mathworld1_cayley as C  # noqa: E402  (frozen instrument, imported not copied)
from llmopt.lab.provenance import completion_commit, start_provenance  # noqa: E402

gate, fsha, components = C.gate, C.fsha, C.components
SMOKE = os.environ.get("DA_SMOKE") == "1"
OUTDIR = Path("logs/mathworld1/decisionatlas_smoke" if SMOKE else "logs/mathworld1/decisionatlas")
STREAMS = {"DISCOVERY": "logs/mathworld1/prband2atlasscore", "FRESH": "logs/mathworld1/prband2atlasfresh"}
AGG_KEY = {"DISCOVERY": "chunk_shas", "FRESH": "chunk_shas"}
CODE = {"i_sum": "S", "A0": "A", "B0": "B", "I0/t5": "I", None: "T"}
NAMES = ("i_sum", "A0", "B0", "I0/t5")


def top1_of(scores):
    """scores: dict sem->float. Returns (winner or None, tie flag,
    runner-up margin)."""
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best = ranked[0][1]
    ties = [k for k, v in scores.items() if v == best]
    margin = best - ranked[1][1]
    return (ranked[0][0] if len(ties) == 1 else None), len(ties) > 1, margin


def check_top1_identity():
    src = Path("scratch/mathworld1_prband2score.py").read_text()
    theirs = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "top1_of"]
    mine = [n for n in ast.parse(Path(__file__).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "top1_of"]
    gate(len(theirs) == 1 and len(mine) == 1 and ast.dump(theirs[0]) == ast.dump(mine[0]), "top1_of SOURCE IDENTITY")


def sha_stream(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def parse_stream(path, limit=None):
    """Returns (states, sums, counts): states[s] = {pair_id, theta, gold};
    sums[(atlas, s)] = {name: sum} for FULL rows; counts of arms and of
    duplicate replay rows (asserted identical)."""
    states, sums, counts = {}, {}, Counter()
    with open(path) as f:
        for n, line in enumerate(f):
            if limit is not None and n >= limit:
                break
            r = json.loads(line)
            counts[(r["arm"], r["mask"])] += 1
            if r["arm"] != "FULL" or r["mask"] != 255:
                continue
            s = r["state"]
            st = states.setdefault(s, {"pair_id": r["pair_id"], "theta": r["theta"], "gold": tuple(r["gold"])})
            gate(st["pair_id"] == r["pair_id"] and st["theta"] == r["theta"] and st["gold"] == tuple(r["gold"]), "STATE CONSISTENCY")
            if tuple(r["candidate"]) == st["gold"]:
                st["gold_name"] = r["name"]
            key = (r["atlas_index"], s)
            d = sums.setdefault(key, {})
            if r["name"] in d:
                gate(d[r["name"]] == r["sum"] and r["atlas_index"] == 12, "DUPLICATE ROW NOT REPLAY-IDENTICAL")
                counts["replay_duplicates"] += 1
            else:
                d[r["name"]] = r["sum"]
    return states, sums, counts


def analyze_ck(ck, states, sums, adj, edges, pol_rows, anchors, smoke=False):
    S = sorted(states)
    gate(smoke or (len(S) == 96 and S == list(range(96))), "96 STATES")
    for s in S:
        gate("gold_name" in states[s] and states[s]["gold_name"] == ("A0" if states[s]["theta"] == "SIN_LOW" else "B0"), f"GOLD {s}")
    R = sorted({a for a, _ in sums})
    if smoke:   # a row-limited smoke keeps only renderers whose 96 states are complete
        R = [a for a in R if all((a, s) in sums and len(sums[(a, s)]) == 4 for s in S)]
    gate(smoke or R == list(range(720)), "720 RENDERERS")
    act = {}
    for r in R:
        row = []
        for s in S:
            d = sums[(r, s)]
            gate(set(d) == set(NAMES), f"FOUR CANDIDATES {r} {s}")
            w, _, _ = top1_of(d)
            row.append(CODE[w])
        act[r] = "".join(row)
    gold = "".join(CODE[states[s]["gold_name"]] for s in S)
    # gates against the policy table
    if not smoke:
        for r in R:
            corr = [act[r][i] == gold[i] for i in range(96)]
            T = sum(corr)
            gate(T == pol_rows[r][ck]["T"], f"T MISMATCH {r}")
            pairs = defaultdict(list)
            for i, s in enumerate(S):
                pairs[states[s]["pair_id"]].append(corr[i])
            gate(len(pairs) == 48 and all(len(v) == 2 for v in pairs.values()), "PAIR JOIN")
            gate(sum(all(v) for v in pairs.values()) == pol_rows[r][ck]["B"], f"B MISMATCH {r}")
            census = Counter(act[r][i] for i in range(96))
            tc = pol_rows[r][ck]["top_census"]
            exp = {CODE[k]: v for k, v in tc.items()} if tc else {}
            got = {k: v for k, v in census.items() if k != "T"}
            gate(got == {k: v for k, v in exp.items() if k != "T"} and census.get("T", 0) == pol_rows[r][ck]["ties"], f"TOP CENSUS MISMATCH {r} {got} v {exp}")
    ns = len(S)
    # per-state statistics
    per_state = []
    for i, s in enumerate(S):
        col = {r: act[r][i] for r in R}
        bnd = [(u, v) for u, v in edges if u in col and v in col and col[u] != col[v]]
        correct = [r for r in R if col[r] == gold[i]]
        comps = components(adj, correct) if correct else []
        wrong = Counter(col[r] for r in R if col[r] != gold[i])
        per_state.append({"state": s, "pair_id": states[s]["pair_id"], "theta": states[s]["theta"], "gold": gold[i],
                          "boundary_edges": len(bnd), "boundary_fraction": len(bnd) / len(edges),
                          "n_correct": len(correct), "correct_components": len(comps),
                          "correct_largest_fraction": (len(comps[0]) / len(correct)) if correct else None,
                          "action_census": dict(Counter(col.values())), "wrong_census": dict(wrong),
                          "top_wrong": (wrong.most_common(1)[0][0] if wrong else None),
                          "invariant": len(bnd) == 0,
                          "invariant_kind": (None if bnd else ("CORRECT" if len(correct) == len(R) else ("WRONG" if not correct else "MIXED-NO-EDGE")))})
    # per-edge flip counts
    flips = {}
    flip_sets = defaultdict(set)
    for u, v in edges:
        if u not in act or v not in act:
            continue
        k = 0
        for i in range(ns):
            if act[u][i] != act[v][i]:
                k += 1
                flip_sets[i].add((u, v))
        flips[(u, v)] = k
    hist = Counter(flips.values())
    ne = len(flips)
    # theta summaries
    def med(v):
        v = sorted(v)
        n = len(v)
        return None if n == 0 else (v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2)

    def summ(v):
        return {"n": len(v), "median": med(v), "min": (min(v) if v else None), "max": (max(v) if v else None),
                "mean": (round(sum(v) / len(v), 6) if v else None)}
    theta = {}
    for th in ("SIN_LOW", "COS_LOW"):
        ps = [p for p in per_state if p["theta"] == th]
        wrong_all = Counter()
        for p in ps:
            wrong_all.update(p["wrong_census"])
        theta[th] = {"boundary_fraction": summ([p["boundary_fraction"] for p in ps]),
                     "n_correct": summ([p["n_correct"] for p in ps]),
                     "correct_components": summ([p["correct_components"] for p in ps]),
                     "invariant_states": sum(p["invariant"] for p in ps),
                     "wrong_census_total": dict(wrong_all),
                     "top_wrong": (wrong_all.most_common(1)[0][0] if wrong_all else None),
                     "states_top_wrong": dict(Counter(p["top_wrong"] for p in ps))}
    # pair-partner co-flip Jaccard
    by_pair = defaultdict(list)
    for i, s in enumerate(S):
        by_pair[states[s]["pair_id"]].append(i)

    def jac(a, b):
        A, B = flip_sets[a], flip_sets[b]
        u = len(A | B)
        return (len(A & B) / u) if u else None
    partner = [jac(v[0], v[1]) for v in by_pair.values() if len(v) == 2]
    partner = [x for x in partner if x is not None]
    partner_set = {tuple(sorted(v)) for v in by_pair.values() if len(v) == 2}
    non = []
    for i in range(ns):
        for j in range(i + 1, ns):
            if (i, j) in partner_set:
                continue
            x = jac(i, j)
            if x is not None:
                non.append(x)
    pm = med(partner)
    coflip = {"partner": summ(partner), "non_partner": summ(non),
              "non_partner_above_partner_median": (sum(x > pm for x in non) if pm is not None else None),
              "n_non_partner": len(non)}
    # anchors
    anch = {}
    for name, a in anchors.items():
        if a in act:
            anch[name] = {"actions": act[a], "T": sum(act[a][i] == gold[i] for i in range(ns)),
                          "edge_flip_counts": {str(w): flips.get((min(a, w), max(a, w))) for w in adj[a]}}
    zero_flip = hist.get(0, 0) / ne if ne else None
    return {"gold": gold, "n_states": ns, "n_renderers": len(R), "decision_matrix": act if not smoke else {str(k): v for k, v in list(act.items())[:5]},
            "per_state": per_state,
            "invariant_states": [p["state"] for p in per_state if p["invariant"]],
            "n_invariant": sum(p["invariant"] for p in per_state),
            "boundary_fraction": summ([p["boundary_fraction"] for p in per_state]),
            "states_fragmented": sum(p["correct_components"] > 1 for p in per_state),
            "flip_hist": {str(k): v for k, v in sorted(hist.items())},
            "mean_flip_per_edge": (sum(flips.values()) / ne if ne else None),
            "mean_flip_fraction": (sum(flips.values()) / ne / ns if ne else None),
            "zero_flip_fraction": zero_flip, "max_flip": (max(flips.values()) if flips else None),
            "theta": theta, "coflip": coflip, "anchors": anch}


def main():
    gate("torch" not in sys.modules, "TORCH IMPORTED")
    check_top1_identity()
    for p, h in C.PINS.items():
        gate(fsha(p) == h, f"PIN DRIFT {p}")
    START = start_provenance(["scratch/mathworld1_decisionatlas.py", "scratch/mathworld1_cayley.py",
                              "scratch/mathworld1_prband2score.py", "llmopt/lab/provenance.py"])
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for f in ("decisionatlas_receipt.json", "DISCOVERY.json", "FRESH.json"):
        gate(not (OUTDIR / f).exists(), f"REFUSE OVERWRITE {f}")
    _, adj, edges, _ = C.build_graph()
    stream_shas, results = {}, {}
    cohorts = ["DISCOVERY"] if SMOKE else list(C.COHORTS)
    for cohort in cohorts:
        spec = C.COHORTS[cohort]
        agg = json.load(open(f"{STREAMS[cohort]}/aggregate.json"))[AGG_KEY[cohort]]
        pol = {r["atlas_index"]: r for r in map(json.loads, open(spec["table"]))}
        res = {"cohort": cohort, "checkpoints": spec["cks"], "anchors": spec["anchors"], "checkpoint": {}}
        cks = spec["cks"][:1] if SMOKE else spec["cks"]
        for ck in cks:
            seed, rep = ck.split("|")
            path = f"{STREAMS[cohort]}/chunk_{seed}_{rep}/scores.jsonl"
            gate(Path(path).exists(), f"STREAM ABSENT {path}")
            h = sha_stream(path)
            gate(h == agg[ck], f"STREAM SHA MISMATCH {ck}")
            stream_shas[ck] = {"path": path, "sha256": h}
            states, sums, counts = parse_stream(path, limit=2000 if SMOKE else None)
            gate(SMOKE or (counts[("FULL", 255)] == 276864 and counts[("MASK0", 0)] == 6144 and counts["replay_duplicates"] == 384), f"ROW CENSUS {ck} {counts}")
            r = analyze_ck(ck, states, sums, adj, edges, pol, spec["anchors"], smoke=SMOKE)
            r["row_census"] = {"FULL": counts[("FULL", 255)], "MASK0": counts[("MASK0", 0)], "replay_duplicates": counts["replay_duplicates"]}
            res["checkpoint"][ck] = r
            print(f"[{ck}] states {r['n_states']} invariant {r['n_invariant']} boundary median {r['boundary_fraction']['median']} "
                  f"mean flip {r['mean_flip_fraction']} zero-flip {r['zero_flip_fraction']} fragmented {r['states_fragmented']} "
                  f"theta bnd med SIN {r['theta']['SIN_LOW']['boundary_fraction']['median']} COS {r['theta']['COS_LOW']['boundary_fraction']['median']} "
                  f"top wrong SIN {r['theta']['SIN_LOW']['top_wrong']} COS {r['theta']['COS_LOW']['top_wrong']} "
                  f"coflip partner med {r['coflip']['partner']['median']} non {r['coflip']['non_partner']['median']} {time.time() - t0:.0f}s", flush=True)
        # bars
        cks_done = list(res["checkpoint"])
        H = None
        try:
            H = json.load(open("logs/mathworld1/morphology/" + cohort + ".json"))["H"]
        except FileNotFoundError:
            pass
        bars = {"B1": {"pass": True, "note": "T, B and top_census reproduced 720 / 720 on every checkpoint (gates; the run halts otherwise)"}}
        g = lambda k: {ck: res["checkpoint"][ck][k] for ck in cks_done}
        bars["B2"] = {"n_invariant": g("n_invariant"), "fires": all(v == 0 for v in g("n_invariant").values())}
        bars["B3"] = {"mean_flip_fraction": g("mean_flip_fraction"), "fires": all(v < 0.15 for v in g("mean_flip_fraction").values())}
        sin = {ck: res["checkpoint"][ck]["theta"]["SIN_LOW"]["boundary_fraction"]["median"] for ck in cks_done}
        cos = {ck: res["checkpoint"][ck]["theta"]["COS_LOW"]["boundary_fraction"]["median"] for ck in cks_done}
        bars["B4"] = {"median_boundary_SIN": sin, "median_boundary_COS": cos, "per_ck": {ck: sin[ck] > cos[ck] for ck in cks_done},
                      "fires": all(sin[ck] > cos[ck] for ck in cks_done)}
        tw = {ck: (res["checkpoint"][ck]["theta"]["SIN_LOW"]["top_wrong"], res["checkpoint"][ck]["theta"]["COS_LOW"]["top_wrong"]) for ck in cks_done}
        bars["B5"] = {"top_wrong_SIN_COS": tw, "per_ck": {ck: tw[ck] == ("B", "A") for ck in cks_done}, "fires": all(tw[ck] == ("B", "A") for ck in cks_done)}
        cf = {ck: (res["checkpoint"][ck]["coflip"]["partner"]["median"], res["checkpoint"][ck]["coflip"]["non_partner"]["median"]) for ck in cks_done}
        bars["B6"] = {"partner_v_non_partner_median": cf, "per_ck": {ck: (cf[ck][0] is not None and cf[ck][1] is not None and cf[ck][0] > cf[ck][1]) for ck in cks_done},
                      "fires": all(cf[ck][0] is not None and cf[ck][1] is not None and cf[ck][0] > cf[ck][1] for ck in cks_done)}
        bars["B7"] = {"states_fragmented": g("states_fragmented"), "fires": all(v >= 48 for v in g("states_fragmented").values())}
        if H is not None:
            zt = {ck: H[ck]["T"]["edge_abs_delta"]["zero_fraction"] for ck in cks_done}
            zf = g("zero_flip_fraction")
            bars["B8"] = {"zero_flip_fraction": zf, "zero_deltaT_fraction": zt, "per_ck": {ck: zf[ck] >= 0.8 * zt[ck] for ck in cks_done},
                          "fires": all(zf[ck] >= 0.8 * zt[ck] for ck in cks_done)}
        res["bars"] = bars
        (OUTDIR / f"{cohort}.json").write_text(json.dumps(res, indent=1))
        results[cohort] = res
        print(f"[{cohort}] bars {json.dumps(bars, default=str)[:1500]}", flush=True)
    receipt = {"prereg": "MATH-CYBER-1-RENDER-ATLAS-DECISION-ATLAS-0" + ("-SMOKE" if SMOKE else ""), "smoke": SMOKE,
               "pins": {p: fsha(p) for p in C.PINS}, "streams": stream_shas,
               "graph": {"V": 720, "E": len(edges), "degree": 5, "diameter": 15},
               "cohorts": {c: {"sha256": fsha(str(OUTDIR / f"{c}.json")), "checkpoints": list(results[c]["checkpoint"]),
                               "bars": results[c]["bars"]} for c in results},
               "semantic_beyond_all_surface_identifiable": False,
               "wall_s": round(time.time() - t0, 1), "start": START, "completion_commit": completion_commit()}
    (OUTDIR / "decisionatlas_receipt.json").write_text(json.dumps(receipt, indent=1))
    print("wall", receipt["wall_s"])


if __name__ == "__main__":
    main()
