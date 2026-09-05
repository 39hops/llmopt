"""Independent verifier for MATH-CYBER-1-RENDER-ATLAS-TRANSPOSITION-RESPONSE-0
(prereg RESULTS L66175). Shares no code with
scratch/mathworld1_transposition.py: own manifest read, own matched-swap
construction (swap the two roles in place), own identity check against a
directly computed precedence effect, own direct / non-adjacent / slot /
gap means and sign counts, own decision-matrix flip and pair-status
anatomy, own checkpoint agreement and bank-D counts, own bar
recomputation; instrument source checked against its start provenance,
inputs against the instrument's entry pins and the receipt lock;
refuses to overwrite its receipt.

Usage:
    .venv/bin/python scratch/mathworld1_transpositionverify.py
"""
import hashlib
import itertools
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

OUTDIR = Path("logs/mathworld1/transposition")
ROLES = ["HI_D", "HI_L", "LO_D", "LO_L", "K", "W"]
FIELDS = ("B", "T", "A0_correct", "B0_correct")
TABLES = {"DISCOVERY": "logs/mathworld1/prband2atlasscore/policy_table.jsonl", "FRESH": "logs/mathworld1/prband2atlasfresh/policy_table.jsonl"}
DA = {"DISCOVERY": "logs/mathworld1/decisionatlas/DISCOVERY.json", "FRESH": "logs/mathworld1/decisionatlas/FRESH.json"}
D = []


def chk(c, m):
    if not c:
        D.append(m)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    chk(not (OUTDIR / "verify_receipt.json").exists(), "REFUSE OVERWRITE verify_receipt.json")
    if D:
        raise SystemExit(D[-1])
    inst = json.load(open(OUTDIR / "transposition_receipt.json"))
    chk(inst.get("smoke") is False and inst["prereg"] == "MATH-CYBER-1-RENDER-ATLAS-TRANSPOSITION-RESPONSE-0", "instrument receipt identity")
    src_sha = sha("scratch/mathworld1_transposition.py")
    chk(src_sha == inst["start"]["file_sha256"]["scratch/mathworld1_transposition.py"], "instrument source sha v receipt")
    chk(inst["start"]["start_commit"] == inst["completion_commit"], "instrument start v completion commit")
    for pth, h in inst["pins"].items():
        chk(sha(pth) == h, f"pin drift {pth}")
    lock = json.load(open("docs/receipts.lock.json"))["receipts"]
    for pth in DA.values():
        chk(sha(pth) == lock[pth]["sha256"] == inst["decision_atlas"][pth], f"decision atlas v lock {pth}")
    if D:
        raise SystemExit(D[-1])
    man = [json.loads(l) for l in open("logs/mathworld1/prband2atlas/atlas_manifest.jsonl")]
    roles = {m["atlas_index"]: tuple(m["roles"]) for m in man}
    idx = {v: k for k, v in roles.items()}
    edges = sorted({(min(i, idx[tuple(roles[i][:k] + (roles[i][k + 1], roles[i][k]) + roles[i][k + 2:])]),
                     max(i, idx[tuple(roles[i][:k] + (roles[i][k + 1], roles[i][k]) + roles[i][k + 2:])])) for i in roles for k in range(5)})
    chk(len(edges) == 1800, "edges")
    inputs = {}
    for cohort in ("DISCOVERY", "FRESH"):
        res = json.load(open(OUTDIR / f"{cohort}.json"))
        inputs[cohort] = sha(OUTDIR / f"{cohort}.json")
        chk(inst["cohorts"][cohort]["sha256"] == inputs[cohort], f"{cohort} cohort sha v instrument receipt")
        rows = {r["atlas_index"]: r for r in map(json.loads, open(TABLES[cohort]))}
        da = json.load(open(DA[cohort]))
        cks = res["checkpoints"]
        X = {ck: {f: [rows[i][ck][f] for i in range(720)] for f in FIELDS} for ck in cks}
        for ck in cks:
            R = res["checkpoint"][ck]
            for Xr, Yr in itertools.combinations(ROLES, 2):
                key = f"{Xr}<{Yr}"
                pairs = []
                for r, o in roles.items():
                    px, py = o.index(Xr), o.index(Yr)
                    if py < px:
                        q = list(o)
                        q[px], q[py] = q[py], q[px]
                        pairs.append((r, idx[tuple(q)], px - py, py))
                chk(len(pairs) == 360, f"{ck} {key} matched")
                for f in FIELDS:
                    x = X[ck][f]
                    before = [x[i] for i in range(720) if roles[i].index(Xr) < roles[i].index(Yr)]
                    after = [x[i] for i in range(720) if roles[i].index(Xr) > roles[i].index(Yr)]
                    effect = sum(before) / 360 - sum(after) / 360
                    d = [(x[rp] - x[r], g, s) for r, rp, g, s in pairs]
                    chk(abs(sum(t for t, _, _ in d) / 360 - effect) < 1e-9, f"{ck} {key} {f} identity")
                    P = R["pairs"][key][f]
                    direct = [t for t, g, _ in d if g == 1]
                    non = [t for t, g, _ in d if g > 1]
                    chk(abs(P["effect"] - effect) < 1e-9 and abs(P["direct_mean"] - sum(direct) / 120) < 1e-9 and abs(P["non_adjacent_mean"] - sum(non) / 240) < 1e-9, f"{ck} {key} {f} means")
                    chk(P["direct_sign"] == {"positive": sum(t > 0 for t in direct), "zero": sum(t == 0 for t in direct), "negative": sum(t < 0 for t in direct)}, f"{ck} {key} {f} sign")
                    for g in range(1, 6):
                        vals = [t for t, gg, _ in d if gg == g]
                        chk(abs(P["gap_mean"][str(g)] - sum(vals) / len(vals)) < 1e-9, f"{ck} {key} {f} gap {g}")
                    sm = {}
                    for s in range(5):
                        vals = [t for t, g, ss in d if g == 1 and ss == s]
                        chk(len(vals) == 24 and abs(P["slot_mean"][str(s)] - sum(vals) / 24) < 1e-9, f"{ck} {key} {f} slot {s}")
                        sm[s] = sum(vals) / 24
                    chk(P["best_slot"] == max(range(5), key=lambda s: (sm[s], -s)), f"{ck} {key} {f} best slot")
                    if effect != 0:
                        chk(abs(P["direct_share"] - (sum(direct) / 120 / 3) / effect) < 1e-9, f"{ck} {key} {f} share")
            # flips on the direct HI_D<->W edges
            A = da["checkpoint"][ck]
            act, gold, ps = A["decision_matrix"], A["gold"], A["per_state"]
            theta = [p["theta"] for p in ps]
            pid = [p["pair_id"] for p in ps]
            direct = []
            for r, o in roles.items():
                px, py = o.index("HI_D"), o.index("W")
                if py < px and px - py == 1:
                    q = list(o)
                    q[px], q[py] = q[py], q[px]
                    direct.append((r, idx[tuple(q)]))
            chk(len(direct) == 120, f"{ck} direct")
            fl = {"toward": Counter(), "away": Counter(), "lateral": Counter(), "flips": Counter()}
            status = Counter()
            for r, rp in direct:
                a0, a1 = act[str(r)], act[str(rp)]
                for i in range(96):
                    if a0[i] != a1[i]:
                        fl["flips"][theta[i]] += 1
                        fl["toward" if a1[i] == gold[i] else ("away" if a0[i] == gold[i] else "lateral")][theta[i]] += 1
                by = defaultdict(dict)
                for i in range(96):
                    by[pid[i]][theta[i]] = (a0[i] == gold[i], a1[i] == gold[i])
                for p, dd in by.items():
                    b0, b1 = all(v[0] for v in dd.values()), all(v[1] for v in dd.values())
                    if b0 != b1:
                        flipped = [th for th, v in dd.items() if v[0] != v[1]]
                        status[("gain" if b1 else "loss") + "|" + ("both" if len(flipped) == 2 else flipped[0])] += 1
            F = R["flips"]
            chk(F["by_theta"] == {k: dict(v) for k, v in fl.items()}, f"{ck} flips by theta")
            chk(F["pair_status_changes"] == dict(status), f"{ck} pair status")
            so, co = status["gain|SIN_LOW"] + status["loss|SIN_LOW"], status["gain|COS_LOW"] + status["loss|COS_LOW"]
            chk(F["sin_only"] == so and F["cos_only"] == co and F["sin_binding"] == (so > co), f"{ck} sin binding")
            # pair-status changes must reconcile with dB on each edge: gains - losses = dB
            for row in F["edges"]:
                chk(row["dB"] == X[ck]["B"][row["r_prime"]] - X[ck]["B"][row["r"]], f"{ck} edge dB {row['r']}")
            gains = sum(v for k, v in status.items() if k.startswith("gain"))
            losses = sum(v for k, v in status.items() if k.startswith("loss"))
            chk(gains - losses == sum(row["dB"] for row in F["edges"]), f"{ck} gains-losses v sum dB")
            print(ck, "checked; discrepancies so far", len(D), flush=True)
        # agreement + bank-D counts
        direct = []
        for r, o in roles.items():
            px, py = o.index("HI_D"), o.index("W")
            if py < px and px - py == 1:
                q = list(o)
                q[px], q[py] = q[py], q[px]
                direct.append((r, idx[tuple(q)]))
        agree = Counter()
        for r, rp in direct:
            ds = [X[ck]["B"][rp] - X[ck]["B"][r] for ck in cks]
            agree["all_positive" if all(v > 0 for v in ds) else "all_non_negative_one_positive" if all(v >= 0 for v in ds) and any(v > 0 for v in ds)
                  else "all_zero" if all(v == 0 for v in ds) else "all_non_positive" if all(v <= 0 for v in ds) else "mixed"] += 1
        chk(res["agreement"]["direct_HI_D_W"] == dict(agree), f"{cohort} agreement")
        sa = wf = wr = 0
        for u, v in edges:
            ds = [X[ck]["B"][v] - X[ck]["B"][u] for ck in cks]
            sa += all(t > 0 for t in ds)
            wf += all(t >= 0 for t in ds) and any(t > 0 for t in ds)
            wr += all(t <= 0 for t in ds) and any(t < 0 for t in ds)
        chk(res["agreement"]["bank_D_all_edges"] == {"strict_all": sa, "weak_all_forward": wf, "weak_all_reverse": wr}, f"{cohort} bank D")
        if cohort == "DISCOVERY":
            chk((sa, wf, wr) == (3, 256, 477), "bank D v L65907")
        b = res["bars"]
        chk(b["a"]["fires"] == all(res["checkpoint"][ck]["pairs"]["HI_D<W"]["B"]["direct_mean"] >= res["checkpoint"][ck]["pairs"]["HI_D<W"]["B"]["non_adjacent_mean"] for ck in cks), f"{cohort} bar a")
        chk(b["b"]["fires"] == all(res["checkpoint"][ck]["pairs"]["HI_D<W"]["B"]["direct_sign"]["positive"] + res["checkpoint"][ck]["pairs"]["HI_D<W"]["B"]["direct_sign"]["zero"] >= 100 and res["checkpoint"][ck]["pairs"]["HI_D<W"]["B"]["direct_sign"]["positive"] >= 60 for ck in cks), f"{cohort} bar b")
        chk(b["c"]["fires"] == all(res["checkpoint"][ck]["pairs"]["HI_D<W"]["B"]["best_slot"] == 4 for ck in cks), f"{cohort} bar c")
        chk(b["d"]["fires"] == all(res["checkpoint"][ck]["flips"]["sin_only"] > res["checkpoint"][ck]["flips"]["cos_only"] for ck in cks), f"{cohort} bar d")
    rec = {"prereg": "MATH-CYBER-1-RENDER-ATLAS-TRANSPOSITION-RESPONSE-0", "verdict": "VERIFIED" if not D else "DISCREPANCIES",
           "discrepancies": D[:40], "n_discrepancies": len(D), "inputs": inputs, "instrument_source_sha256": src_sha,
           "instrument_receipt_sha256": sha(OUTDIR / "transposition_receipt.json"), "instrument_run_commit": inst["start"]["start_commit"],
           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "status_porcelain": subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout,
           "verifier_sha256": sha(__file__)}
    (OUTDIR / "verify_receipt.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1)[:1500])


if __name__ == "__main__":
    main()
