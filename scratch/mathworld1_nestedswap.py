"""MATH-CYBER-1-RENDER-ATLAS-NESTED-SWAP-DECOMPOSITION-0 — bank M (prereg
RESULTS L66224): the 360 matched swaps of a role pair (orientation "X
before Y", delta = x(r') - x(r), L66198) split exactly into 15 cells of 24
by gap and endpoint slot; sums of squares partitioned into between-gap,
between-slot-within-gap and within-cell parts (identity gated), and within
each cell by the role occupying each free position; gap-5 interior-order
readout and gap-4 outside-role x interior-order tables for HI_D<->W; the
nested decomposition on B for all 15 pairs as the frame. Discovery and
fresh cohorts separately, never pooled. Graph builder, gates and pins
imported from the frozen scratch/mathworld1_cayley.py. No model, no
logit; torch is never imported.

Usage:
    .venv/bin/python scratch/mathworld1_nestedswap.py
    NS_SMOKE=1 .venv/bin/python scratch/mathworld1_nestedswap.py   # one checkpoint, own directory
"""
import itertools
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mathworld1_cayley as C  # noqa: E402  (frozen instrument, imported not copied)
from llmopt.lab.provenance import completion_commit, start_provenance  # noqa: E402

gate, fsha = C.gate, C.fsha
SMOKE = os.environ.get("NS_SMOKE") == "1"
OUTDIR = Path("logs/mathworld1/nestedswap_smoke" if SMOKE else "logs/mathworld1/nestedswap")
ROLES = list(C.ROLES)
FIELDS = ("B", "T", "A0_correct", "B0_correct")
MAIN = ("HI_D", "W")
PAIRS = [(a, b) for a, b in itertools.combinations(ROLES, 2)]
L66198_GAP_MEANS_KEY = "gap_mean"


def mean(v):
    return sum(v) / len(v)


def ss(v, m=None):
    m = mean(v) if m is None else m
    return sum((t - m) ** 2 for t in v)


def cells_for(roles, idx, X, Y):
    """{(g, s): [(r, r_prime, order_of_r_prime)]} in the orientation X before Y."""
    cells = defaultdict(list)
    for r, order in roles.items():
        px, py = order.index(X), order.index(Y)
        if py < px:
            q = list(order)
            q[px], q[py] = q[py], q[px]
            cells[(px - py, py)].append((r, idx[tuple(q)], tuple(q)))
    gate(sorted(cells) == [(g, s) for g in range(1, 6) for s in range(6 - g)] and all(len(v) == 24 for v in cells.values()), f"CELLS {X} {Y}")
    return cells


def decompose(cells, x, X, Y, full=False):
    d = {(g, s): [x[rp] - x[r] for r, rp, _ in v] for (g, s), v in cells.items()}
    allv = [t for v in d.values() for t in v]
    gate(len(allv) == 360, "360")
    m = mean(allv)
    gap_vals = {g: [t for (gg, s), v in d.items() if gg == g for t in v] for g in range(1, 6)}
    gap_m = {g: mean(v) for g, v in gap_vals.items()}
    ss_total = ss(allv, m)
    ss_gap = sum(len(v) * (gap_m[g] - m) ** 2 for g, v in gap_vals.items())
    ss_slot = sum(24 * (mean(v) - gap_m[g]) ** 2 for (g, s), v in d.items())
    ss_within = sum(ss(v) for v in d.values())
    gate(abs(ss_total - (ss_gap + ss_slot + ss_within)) < 1e-6, "SS IDENTITY")
    out = {"mean": m, "gap_mean": {str(g): gap_m[g] for g in gap_m},
           "ss": {"total": ss_total, "gap": ss_gap, "slot": ss_slot, "within": ss_within},
           "fraction": ({"gap": ss_gap / ss_total, "slot": ss_slot / ss_total, "within": ss_within / ss_total,
                         "position": (ss_gap + ss_slot) / ss_total} if ss_total > 0 else "CONSTANT"),
           "cells": {}}
    for (g, s), v in sorted(d.items()):
        orders = cells[(g, s)]
        free = [p for p in range(6) if p not in (s, s + g)]
        cm = mean(v)
        within = ss(v, cm)
        role_ss = {}
        for p in free:
            groups = defaultdict(list)
            for (r, rp, order), t in zip(orders, v):
                groups[order[p]].append(t)
            gate(len(groups) == 4 and all(len(gv) == 6 for gv in groups.values()), "ROLE GROUPS")
            role_ss[p] = sum(6 * (mean(gv) - cm) ** 2 for gv in groups.values())
        best = max(free, key=lambda p: (role_ss[p], -abs(p - s))) if within > 0 else None
        cell = {"gap": g, "slot": s, "endpoints": [s, s + g], "free_positions": free, "mean": cm,
                "sign": {"positive": sum(t > 0 for t in v), "zero": sum(t == 0 for t in v), "negative": sum(t < 0 for t in v)},
                "min": min(v), "max": max(v), "distinct_values": len(set(v)), "ss_within": within,
                "role_ss_fraction": ({str(p): role_ss[p] / within for p in free} if within > 0 else "CONSTANT"),
                "most_explanatory_position": best}
        if full or (g, s) in ((5, 0), (4, 0), (4, 1)):
            cell["deltas"] = [{"r": r, "r_prime": rp, "order": list(order), "d": t} for (r, rp, order), t in zip(orders, v)]
            cell["role_at_position_mean"] = {str(p): {role: mean([t for (r, rp, order), t in zip(orders, v) if order[p] == role]) for role in ROLES if role not in (X, Y)} for p in free}
            if g == 4:
                outside = 5 if s == 0 else 0
                interior = [p for p in free if p != outside]
                tab = defaultdict(dict)
                for (r, rp, order), t in zip(orders, v):
                    tab[order[outside]]["|".join(order[p] for p in interior)] = t
                cell["outside_position"] = outside
                cell["outside_role_x_interior_order"] = {k: dict(sorted(vv.items())) for k, vv in sorted(tab.items())}
                cell["outside_role_mean"] = {k: mean(list(vv.values())) for k, vv in sorted(tab.items())}
                inter = defaultdict(list)
                for k, vv in tab.items():
                    for o, t in vv.items():
                        inter[o].append(t)
                cell["interior_order_mean"] = {o: mean(vv) for o, vv in sorted(inter.items())}
        out["cells"][f"{g},{s}"] = cell
    return out


def main():
    gate("torch" not in sys.modules, "TORCH IMPORTED")
    for p, h in C.PINS.items():
        gate(fsha(p) == h, f"PIN DRIFT {p}")
    lock = json.load(open("docs/receipts.lock.json"))["receipts"]
    L = {"DISCOVERY": "logs/mathworld1/transposition/DISCOVERY.json", "FRESH": "logs/mathworld1/transposition/FRESH.json"}
    for p in L.values():
        gate(fsha(p) == lock[p]["sha256"], f"L66198 RECEIPT LOCK {p}")
    START = start_provenance(["scratch/mathworld1_nestedswap.py", "scratch/mathworld1_cayley.py", "llmopt/lab/provenance.py"] + list(L.values()))
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for f in ("nestedswap_receipt.json", "DISCOVERY.json", "FRESH.json"):
        gate(not (OUTDIR / f).exists(), f"REFUSE OVERWRITE {f}")
    man, _, edges, _ = C.build_graph()
    roles = {m["atlas_index"]: tuple(m["roles"]) for m in man}
    idx = {v: k for k, v in roles.items()}
    results = {}
    for cohort in (["DISCOVERY"] if SMOKE else list(C.COHORTS)):
        spec = C.COHORTS[cohort]
        rows = {r["atlas_index"]: r for r in map(json.loads, open(spec["table"]))}
        prev = json.load(open(L[cohort]))["checkpoint"]
        cks = spec["cks"][:1] if SMOKE else spec["cks"]
        res = {"cohort": cohort, "checkpoints": cks, "checkpoint": {}}
        for ck in cks:
            x = {f: [rows[i][ck][f] for i in range(720)] for f in FIELDS}
            R = {"main": {}, "frame_B": {}}
            cm = cells_for(roles, idx, *MAIN)
            for f in FIELDS:
                R["main"][f] = decompose(cm, x[f], *MAIN, full=True)
                # reproduce the L66198 gap means
                g_prev = prev[ck]["pairs"]["HI_D<W"][f][L66198_GAP_MEANS_KEY]
                gate(all(abs(R["main"][f]["gap_mean"][k] - g_prev[k]) < 1e-9 for k in g_prev), f"L66198 GAP MEANS {ck} {f}")
            for X, Y in PAIRS:
                dd = decompose(cells_for(roles, idx, X, Y), x["B"], X, Y)
                R["frame_B"][f"{X}<{Y}"] = {"fraction": dd["fraction"], "gap_mean": dd["gap_mean"],
                                           "most_explanatory_position": {k: v["most_explanatory_position"] for k, v in dd["cells"].items()}}
            res["checkpoint"][ck] = R
            mb = R["main"]["B"]
            c5, c40, c41 = mb["cells"]["5,0"], mb["cells"]["4,0"], mb["cells"]["4,1"]
            print(f"[{ck}] B fractions {mb['fraction']} gap5 mean {c5['mean']:.3f} sign {c5['sign']} distinct {c5['distinct_values']} role_ss {c5['role_ss_fraction']} best {c5['most_explanatory_position']} | gap4 (0,4) {c40['mean']:.3f} (1,5) {c41['mean']:.3f} outside means {c40['outside_role_mean']} {c41['outside_role_mean']}", flush=True)
        g = lambda fn: {ck: fn(res["checkpoint"][ck]["main"]["B"]) for ck in cks}
        pa = g(lambda m: m["fraction"]["position"] if m["fraction"] != "CONSTANT" else None)
        pb = g(lambda m: m["cells"]["5,0"]["sign"]["positive"])
        pc = g(lambda m: m["cells"]["5,0"]["most_explanatory_position"])
        pd = g(lambda m: (m["cells"]["4,1"]["mean"], m["cells"]["4,0"]["mean"]))
        bars = {"B0": {"pass": True, "note": "15 cells of 24 per pair, SS identity to 1e-6 and L66198 gap means to 1e-9 are halting gates"},
                "a": {"position_fraction_B": pa, "per_ck": {ck: (pa[ck] is not None and pa[ck] >= 0.5) for ck in cks}, "fires": all(pa[ck] is not None and pa[ck] >= 0.5 for ck in cks)},
                "b": {"gap5_positive_of_24": pb, "per_ck": {ck: pb[ck] >= 18 for ck in cks}, "fires": all(v >= 18 for v in pb.values())},
                "c": {"gap5_most_explanatory_position": pc, "per_ck": {ck: pc[ck] == 1 for ck in cks}, "fires": all(v == 1 for v in pc.values())},
                "d": {"gap4_cell_means_15_v_04": pd, "per_ck": {ck: pd[ck][0] > pd[ck][1] for ck in cks}, "fires": all(pd[ck][0] > pd[ck][1] for ck in cks)}}
        res["bars"] = bars
        (OUTDIR / f"{cohort}.json").write_text(json.dumps(res, indent=1))
        results[cohort] = res
        print(f"[{cohort}] bars {json.dumps({k: v for k, v in bars.items() if k != 'B0'}, default=str)[:1500]}", flush=True)
    receipt = {"prereg": "MATH-CYBER-1-RENDER-ATLAS-NESTED-SWAP-DECOMPOSITION-0" + ("-SMOKE" if SMOKE else ""), "smoke": SMOKE,
               "pins": {p: fsha(p) for p in C.PINS}, "l66198_receipts": {p: fsha(p) for p in L.values()}, "graph_edges": len(edges),
               "cohorts": {c: {"sha256": fsha(str(OUTDIR / f"{c}.json")), "checkpoints": results[c]["checkpoints"], "bars": results[c]["bars"]} for c in results},
               "semantic_beyond_all_surface_identifiable": False,
               "wall_s": round(time.time() - t0, 1), "start": START, "completion_commit": completion_commit()}
    (OUTDIR / "nestedswap_receipt.json").write_text(json.dumps(receipt, indent=1))
    print("wall", receipt["wall_s"])


if __name__ == "__main__":
    main()
