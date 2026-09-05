"""MATH-CYBER-1-RENDER-ATLAS-TRANSPOSITION-RESPONSE-0 — bank L (prereg
RESULTS L66175; design L66150): matched transposition response of the
twelve booked render atlases. For every unordered role pair the 360
matched in-place swaps (other four roles fixed) decompose the bank-G
precedence effect exactly into 120 direct adjacent transpositions (Cayley
edges) and 240 non-adjacent swaps; the instrument gates that identity,
then reports direct v non-adjacent means, sign counts, slot / context /
gap dependence, the state-level flip anatomy of the direct HI_D<->W edges
from the booked decision matrices, and per-edge checkpoint agreement with
the bank-D counts of L65907 reproduced as a gate. Discovery and fresh
cohorts separately, never pooled. Graph builder, gates and pins imported
from the frozen scratch/mathworld1_cayley.py. No model, no logit; torch is
never imported.

Usage:
    .venv/bin/python scratch/mathworld1_transposition.py
    TR_SMOKE=1 .venv/bin/python scratch/mathworld1_transposition.py   # one checkpoint, own directory
"""
import itertools
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

gate, fsha = C.gate, C.fsha
SMOKE = os.environ.get("TR_SMOKE") == "1"
OUTDIR = Path("logs/mathworld1/transposition_smoke" if SMOKE else "logs/mathworld1/transposition")
ROLES = list(C.ROLES)
FIELDS = ("B", "T", "A0_correct", "B0_correct")
DA = {"DISCOVERY": "logs/mathworld1/decisionatlas/DISCOVERY.json", "FRESH": "logs/mathworld1/decisionatlas/FRESH.json"}
MAIN = ("HI_D", "W")
RUNNERS = [("LO_D", "W"), ("LO_L", "W"), ("K", "W"), ("HI_L", "W"), ("HI_D", "HI_L"), ("HI_D", "LO_D")]
PAIRS = [(a, b) for a, b in itertools.combinations(ROLES, 2)]


def mean(v):
    return sum(v) / len(v) if v else None


def matched(roles, idx, X, Y):
    """All 360 matched swaps in the orientation 'X before Y': list of
    (r, r_prime, gap, slot) with r having Y before X."""
    out = []
    for r, order in roles.items():
        px, py = order.index(X), order.index(Y)
        if py < px:
            q = list(order)
            q[px], q[py] = q[py], q[px]
            rp = idx[tuple(q)]
            out.append((r, rp, px - py, py))
    gate(len(out) == 360, f"MATCHED {X} {Y}")
    return out


def analyze_ck(ck, x, roles, idx, edges, da, anchors):
    res = {"pairs": {}, "flips": None, "agreement": None}
    for X, Y in PAIRS:
        M = matched(roles, idx, X, Y)
        gate(Counter(g for _, _, g, _ in M) == Counter({1: 120, 2: 96, 3: 72, 4: 48, 5: 24}), "GAP PROFILE")
        P = {}
        for f in FIELDS:
            d = {(r, rp): x[f][rp] - x[f][r] for r, rp, g, s in M}
            before = [x[f][rp] for r, rp, g, s in M]
            after = [x[f][r] for r, rp, g, s in M]
            effect = mean(before) - mean(after)
            gate(abs(mean(list(d.values())) - effect) < 1e-9, f"IDENTITY {X} {Y} {f}")
            direct = [d[(r, rp)] for r, rp, g, s in M if g == 1]
            non = [d[(r, rp)] for r, rp, g, s in M if g > 1]
            gate(len(direct) == 120 and len(non) == 240, "SPLIT")
            slots = {s: [d[(r, rp)] for r, rp, g, ss in M if g == 1 and ss == s] for s in range(5)}
            gate(all(len(v) == 24 for v in slots.values()), "SLOTS")
            ctx_before, ctx_after = defaultdict(list), defaultdict(list)
            for r, rp, g, s in M:
                if g != 1:
                    continue
                o = roles[rp]
                ctx_before[o[s - 1] if s > 0 else "NONE"].append(d[(r, rp)])
                ctx_after[o[s + 2] if s + 2 < 6 else "NONE"].append(d[(r, rp)])
            P[f] = {"effect": effect, "direct_mean": mean(direct), "non_adjacent_mean": mean(non),
                    "direct_share": ((mean(direct) / 3) / effect) if effect != 0 else None,
                    "direct_sign": {"positive": sum(v > 0 for v in direct), "zero": sum(v == 0 for v in direct), "negative": sum(v < 0 for v in direct)},
                    "direct_min": min(direct), "direct_max": max(direct),
                    "gap_mean": {str(g): mean([d[(r, rp)] for r, rp, gg, s in M if gg == g]) for g in range(1, 6)},
                    "slot_mean": {str(s): mean(v) for s, v in slots.items()},
                    "slot_sign": {str(s): {"positive": sum(t > 0 for t in v), "zero": sum(t == 0 for t in v), "negative": sum(t < 0 for t in v)} for s, v in slots.items()},
                    "best_slot": max(range(5), key=lambda s: (mean(slots[s]), -s)),
                    "context_before_mean": {k: mean(v) for k, v in sorted(ctx_before.items())},
                    "context_after_mean": {k: mean(v) for k, v in sorted(ctx_after.items())}}
        res["pairs"][f"{X}<{Y}"] = P
        if (X, Y) == MAIN:
            # state anatomy on the direct edges from the decision matrix
            act, gold = da["decision_matrix"], da["gold"]
            per_state = da["per_state"]
            theta = [p["theta"] for p in per_state]
            pid = [p["pair_id"] for p in per_state]
            fl = {"toward": Counter(), "away": Counter(), "lateral": Counter(), "flips": Counter()}
            status = Counter()
            edge_rows = []
            for r, rp, g, s in M:
                if g != 1:
                    continue
                a0, a1 = act[str(r)], act[str(rp)]
                nf = 0
                for i in range(96):
                    if a0[i] != a1[i]:
                        nf += 1
                        fl["flips"][theta[i]] += 1
                        if a1[i] == gold[i]:
                            fl["toward"][theta[i]] += 1
                        elif a0[i] == gold[i]:
                            fl["away"][theta[i]] += 1
                        else:
                            fl["lateral"][theta[i]] += 1
                # pair status
                by = defaultdict(dict)
                for i in range(96):
                    by[pid[i]][theta[i]] = (a0[i] == gold[i], a1[i] == gold[i])
                for p, dd in by.items():
                    b0 = all(v[0] for v in dd.values())
                    b1 = all(v[1] for v in dd.values())
                    if b0 == b1:
                        continue
                    flipped = [th for th, v in dd.items() if v[0] != v[1]]
                    key = ("gain" if b1 else "loss") + "|" + ("both" if len(flipped) == 2 else flipped[0])
                    status[key] += 1
                edge_rows.append({"r": r, "r_prime": rp, "slot": s, "dB": x["B"][rp] - x["B"][r], "dT": x["T"][rp] - x["T"][r], "flips": nf})
            sin_only = status["gain|SIN_LOW"] + status["loss|SIN_LOW"]
            cos_only = status["gain|COS_LOW"] + status["loss|COS_LOW"]
            res["flips"] = {"by_theta": {k: dict(v) for k, v in fl.items()},
                            "pair_status_changes": dict(status), "sin_only": sin_only, "cos_only": cos_only,
                            "both": status["gain|both"] + status["loss|both"], "sin_binding": sin_only > cos_only,
                            "edges": edge_rows}
    # anchors: which direct HI_D<->W edge (if any) touches each anchor
    res["anchor_edges"] = {}
    M = matched(roles, idx, *MAIN)
    for name, a in anchors.items():
        hit = [(r, rp) for r, rp, g, s in M if g == 1 and a in (r, rp)]
        res["anchor_edges"][name] = [{"r": r, "r_prime": rp, "anchor_is_W_first": a == r, "dB": x["B"][rp] - x["B"][r]} for r, rp in hit]
    return res


def main():
    gate("torch" not in sys.modules, "TORCH IMPORTED")
    for p, h in C.PINS.items():
        gate(fsha(p) == h, f"PIN DRIFT {p}")
    lock = json.load(open("docs/receipts.lock.json"))["receipts"]
    for p in DA.values():
        gate(fsha(p) == lock[p]["sha256"], f"DECISION ATLAS LOCK {p}")
    START = start_provenance(["scratch/mathworld1_transposition.py", "scratch/mathworld1_cayley.py", "llmopt/lab/provenance.py"] + list(DA.values()))
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for f in ("transposition_receipt.json", "DISCOVERY.json", "FRESH.json"):
        gate(not (OUTDIR / f).exists(), f"REFUSE OVERWRITE {f}")
    man, adj, edges, _ = C.build_graph()
    roles = {m["atlas_index"]: tuple(m["roles"]) for m in man}
    idx = {v: k for k, v in roles.items()}
    results = {}
    for cohort in (["DISCOVERY"] if SMOKE else list(C.COHORTS)):
        spec = C.COHORTS[cohort]
        rows = {r["atlas_index"]: r for r in map(json.loads, open(spec["table"]))}
        da = json.load(open(DA[cohort]))
        cks = spec["cks"][:1] if SMOKE else spec["cks"]
        res = {"cohort": cohort, "checkpoints": cks, "anchors": spec["anchors"], "checkpoint": {}}
        X = {ck: {f: [rows[i][ck][f] for i in range(720)] for f in FIELDS} for ck in cks}
        for ck in cks:
            res["checkpoint"][ck] = analyze_ck(ck, X[ck], roles, idx, edges, da["checkpoint"][ck], spec["anchors"])
            P = res["checkpoint"][ck]["pairs"]["HI_D<W"]["B"]
            print(f"[{ck}] HI_D<W B effect {P['effect']:.3f} direct {P['direct_mean']:.3f} non {P['non_adjacent_mean']:.3f} sign {P['direct_sign']} slots {[round(v, 2) for v in P['slot_mean'].values()]} best {P['best_slot']} flips {res['checkpoint'][ck]['flips']['pair_status_changes']}", flush=True)
        # checkpoint agreement on the direct HI_D<->W edges + bank-D gate (discovery cohort has 4 cks; fresh 8)
        M = matched(roles, idx, *MAIN)
        direct = [(r, rp) for r, rp, g, s in M if g == 1]
        agree = Counter()
        for r, rp in direct:
            ds = [X[ck]["B"][rp] - X[ck]["B"][r] for ck in cks]
            if all(v > 0 for v in ds):
                agree["all_positive"] += 1
            elif all(v >= 0 for v in ds) and any(v > 0 for v in ds):
                agree["all_non_negative_one_positive"] += 1
            elif all(v == 0 for v in ds):
                agree["all_zero"] += 1
            elif all(v <= 0 for v in ds):
                agree["all_non_positive"] += 1
            else:
                agree["mixed"] += 1
        # bank-D gate under ascending-index orientation over all 1,800 edges
        strict_all = weak_fwd = weak_rev = 0
        for u, v in edges:
            ds = [X[ck]["B"][v] - X[ck]["B"][u] for ck in cks]
            if all(t > 0 for t in ds):
                strict_all += 1
            if all(t >= 0 for t in ds) and any(t > 0 for t in ds):
                weak_fwd += 1
            if all(t <= 0 for t in ds) and any(t < 0 for t in ds):
                weak_rev += 1
        if cohort == "DISCOVERY" and not SMOKE:
            gate((strict_all, weak_fwd, weak_rev) == (3, 256, 477), f"BANK-D COUNTS {(strict_all, weak_fwd, weak_rev)}")
        hidw_strict = sum(1 for r, rp in direct if all(X[ck]["B"][rp] - X[ck]["B"][r] > 0 for ck in cks))
        res["agreement"] = {"direct_HI_D_W": dict(agree), "bank_D_all_edges": {"strict_all": strict_all, "weak_all_forward": weak_fwd, "weak_all_reverse": weak_rev},
                            "HI_D_W_direct_strict_all": hidw_strict}
        g = lambda fn: {ck: fn(res["checkpoint"][ck]) for ck in cks}
        pa = g(lambda r: (r["pairs"]["HI_D<W"]["B"]["direct_mean"], r["pairs"]["HI_D<W"]["B"]["non_adjacent_mean"]))
        pb = g(lambda r: r["pairs"]["HI_D<W"]["B"]["direct_sign"])
        pc = g(lambda r: r["pairs"]["HI_D<W"]["B"]["best_slot"])
        pd = g(lambda r: (r["flips"]["sin_only"], r["flips"]["cos_only"], r["flips"]["both"]))
        bars = {"B0": {"pass": True, "identity_pairs": 15, "bank_D": res["agreement"]["bank_D_all_edges"]},
                "a": {"direct_v_non": pa, "per_ck": {ck: pa[ck][0] >= pa[ck][1] for ck in cks}, "fires": all(pa[ck][0] >= pa[ck][1] for ck in cks)},
                "b": {"sign": pb, "per_ck": {ck: (pb[ck]["positive"] + pb[ck]["zero"] >= 100 and pb[ck]["positive"] >= 60) for ck in cks},
                      "fires": all(pb[ck]["positive"] + pb[ck]["zero"] >= 100 and pb[ck]["positive"] >= 60 for ck in cks)},
                "c": {"best_slot": pc, "per_ck": {ck: pc[ck] == 4 for ck in cks}, "fires": all(v == 4 for v in pc.values())},
                "d": {"sin_only_cos_only_both": pd, "per_ck": {ck: pd[ck][0] > pd[ck][1] for ck in cks}, "fires": all(pd[ck][0] > pd[ck][1] for ck in cks)}}
        res["bars"] = bars
        (OUTDIR / f"{cohort}.json").write_text(json.dumps(res, indent=1))
        results[cohort] = res
        print(f"[{cohort}] agreement {res['agreement']} bars {json.dumps({k: v for k, v in bars.items() if k != 'B0'}, default=str)[:1500]}", flush=True)
    receipt = {"prereg": "MATH-CYBER-1-RENDER-ATLAS-TRANSPOSITION-RESPONSE-0" + ("-SMOKE" if SMOKE else ""), "smoke": SMOKE,
               "pins": {p: fsha(p) for p in C.PINS}, "decision_atlas": {p: fsha(p) for p in DA.values()}, "graph_edges": len(edges),
               "cohorts": {c: {"sha256": fsha(str(OUTDIR / f"{c}.json")), "checkpoints": results[c]["checkpoints"], "bars": results[c]["bars"],
                               "agreement": results[c]["agreement"]} for c in results},
               "semantic_beyond_all_surface_identifiable": False,
               "wall_s": round(time.time() - t0, 1), "start": START, "completion_commit": completion_commit()}
    (OUTDIR / "transposition_receipt.json").write_text(json.dumps(receipt, indent=1))
    print("wall", receipt["wall_s"])


if __name__ == "__main__":
    main()
