"""Independent verifier for MATH-CYBER-1-RENDER-ATLAS-NESTED-SWAP-DECOMPOSITION-0
(prereg RESULTS L66224). Shares no code with scratch/mathworld1_nestedswap.py:
own manifest read, own matched-swap cells by gap and slot, own sums of
squares (total / gap / slot / within and role-at-position within cells),
own gap-5 and gap-4 tables, own bar recomputation; instrument source
checked against its start provenance, inputs against the instrument's
entry pins and the receipt lock; refuses to overwrite its receipt.

Usage:
    .venv/bin/python scratch/mathworld1_nestedswapverify.py
"""
import hashlib
import itertools
import json
import subprocess
from collections import defaultdict
from pathlib import Path

OUTDIR = Path("logs/mathworld1/nestedswap")
ROLES = ["HI_D", "HI_L", "LO_D", "LO_L", "K", "W"]
FIELDS = ("B", "T", "A0_correct", "B0_correct")
TABLES = {"DISCOVERY": "logs/mathworld1/prband2atlasscore/policy_table.jsonl", "FRESH": "logs/mathworld1/prband2atlasfresh/policy_table.jsonl"}
D = []


def chk(c, m):
    if not c:
        D.append(m)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def mean(v):
    return sum(v) / len(v)


def main():
    chk(not (OUTDIR / "verify_receipt.json").exists(), "REFUSE OVERWRITE verify_receipt.json")
    if D:
        raise SystemExit(D[-1])
    inst = json.load(open(OUTDIR / "nestedswap_receipt.json"))
    chk(inst.get("smoke") is False and inst["prereg"] == "MATH-CYBER-1-RENDER-ATLAS-NESTED-SWAP-DECOMPOSITION-0", "instrument receipt identity")
    src_sha = sha("scratch/mathworld1_nestedswap.py")
    chk(src_sha == inst["start"]["file_sha256"]["scratch/mathworld1_nestedswap.py"], "instrument source sha v receipt")
    chk(inst["start"]["start_commit"] == inst["completion_commit"], "instrument start v completion commit")
    for pth, h in inst["pins"].items():
        chk(sha(pth) == h, f"pin drift {pth}")
    lock = json.load(open("docs/receipts.lock.json"))["receipts"]
    for pth, h in inst["l66198_receipts"].items():
        chk(sha(pth) == h == lock[pth]["sha256"], f"L66198 receipt v lock {pth}")
    if D:
        raise SystemExit(D[-1])
    man = [json.loads(l) for l in open("logs/mathworld1/prband2atlas/atlas_manifest.jsonl")]
    roles = {m["atlas_index"]: tuple(m["roles"]) for m in man}
    idx = {v: k for k, v in roles.items()}
    inputs = {}
    for cohort in ("DISCOVERY", "FRESH"):
        res = json.load(open(OUTDIR / f"{cohort}.json"))
        inputs[cohort] = sha(OUTDIR / f"{cohort}.json")
        chk(inst["cohorts"][cohort]["sha256"] == inputs[cohort], f"{cohort} cohort sha v instrument receipt")
        rows = {r["atlas_index"]: r for r in map(json.loads, open(TABLES[cohort]))}
        prev = json.load(open(f"logs/mathworld1/transposition/{cohort}.json"))["checkpoint"]
        for ck in res["checkpoints"]:
            R = res["checkpoint"][ck]
            x = {f: [rows[i][ck][f] for i in range(720)] for f in FIELDS}
            for X, Y in itertools.combinations(ROLES, 2):
                key = f"{X}<{Y}"
                cells = defaultdict(list)
                for r, o in roles.items():
                    px, py = o.index(X), o.index(Y)
                    if py < px:
                        q = list(o)
                        q[px], q[py] = q[py], q[px]
                        cells[(px - py, py)].append((r, idx[tuple(q)], tuple(q)))
                chk(len(cells) == 15 and all(len(v) == 24 for v in cells.values()), f"{ck} {key} cells")
                fields = FIELDS if (X, Y) == ("HI_D", "W") else ("B",)
                for f in fields:
                    got = R["main"][f] if (X, Y) == ("HI_D", "W") else R["frame_B"][key]
                    d = {c: [x[f][rp] - x[f][r] for r, rp, _ in v] for c, v in cells.items()}
                    allv = [t for v in d.values() for t in v]
                    m = mean(allv)
                    tot = sum((t - m) ** 2 for t in allv)
                    gm = {g: mean([t for (gg, s), v in d.items() if gg == g for t in v]) for g in range(1, 6)}
                    sg = sum(len([t for (gg, s), v in d.items() if gg == g for t in v]) * (gm[g] - m) ** 2 for g in range(1, 6))
                    sl = sum(24 * (mean(v) - gm[g]) ** 2 for (g, s), v in d.items())
                    sw = sum(sum((t - mean(v)) ** 2 for t in v) for v in d.values())
                    chk(abs(tot - sg - sl - sw) < 1e-6, f"{ck} {key} {f} ss identity")
                    for g in range(1, 6):
                        chk(abs(got["gap_mean"][str(g)] - gm[g]) < 1e-9, f"{ck} {key} {f} gap {g}")
                    if tot > 0:
                        chk(got["fraction"] != "CONSTANT" and abs(got["fraction"]["gap"] - sg / tot) < 1e-9 and abs(got["fraction"]["slot"] - sl / tot) < 1e-9
                            and abs(got["fraction"]["within"] - sw / tot) < 1e-9 and abs(got["fraction"]["position"] - (sg + sl) / tot) < 1e-9, f"{ck} {key} {f} fractions")
                    else:
                        chk(got["fraction"] == "CONSTANT", f"{ck} {key} {f} constant")
                    # per-cell role-at-position SS and most explanatory position
                    for (g, s), v in d.items():
                        cm = mean(v)
                        within = sum((t - cm) ** 2 for t in v)
                        free = [p for p in range(6) if p not in (s, s + g)]
                        rs = {}
                        for p in free:
                            groups = defaultdict(list)
                            for (r, rp, o), t in zip(cells[(g, s)], v):
                                groups[o[p]].append(t)
                            rs[p] = sum(6 * (mean(gv) - cm) ** 2 for gv in groups.values())
                        best = max(free, key=lambda p: (rs[p], -abs(p - s))) if within > 0 else None
                        if (X, Y) == ("HI_D", "W"):
                            cell = got["cells"][f"{g},{s}"]
                            chk(abs(cell["mean"] - cm) < 1e-9 and cell["sign"] == {"positive": sum(t > 0 for t in v), "zero": sum(t == 0 for t in v), "negative": sum(t < 0 for t in v)}, f"{ck} {f} cell {g},{s}")
                            chk(cell["most_explanatory_position"] == best, f"{ck} {f} cell {g},{s} best")
                            if within > 0:
                                chk(all(abs(cell["role_ss_fraction"][str(p)] - rs[p] / within) < 1e-9 for p in free), f"{ck} {f} cell {g},{s} role ss")
                            chk(cell["distinct_values"] == len(set(v)) and cell["min"] == min(v) and cell["max"] == max(v), f"{ck} {f} cell {g},{s} stats")
                            if "deltas" in cell:
                                chk(sorted((e["r"], e["r_prime"], e["d"]) for e in cell["deltas"]) == sorted((r, rp, t) for (r, rp, o), t in zip(cells[(g, s)], v)), f"{ck} {f} cell {g},{s} deltas")
                            if g == 4:
                                outside = 5 if s == 0 else 0
                                om = defaultdict(list)
                                for (r, rp, o), t in zip(cells[(g, s)], v):
                                    om[o[outside]].append(t)
                                chk(all(abs(cell["outside_role_mean"][k] - mean(vv)) < 1e-9 for k, vv in om.items()), f"{ck} {f} cell {g},{s} outside means")
                        else:
                            chk(got["most_explanatory_position"][f"{g},{s}"] == best, f"{ck} {key} frame best {g},{s}")
            # L66198 gap means
            for f in FIELDS:
                chk(all(abs(R["main"][f]["gap_mean"][k] - vv) < 1e-9 for k, vv in prev[ck]["pairs"]["HI_D<W"][f]["gap_mean"].items()), f"{ck} {f} v L66198")
            print(ck, "checked; discrepancies so far", len(D), flush=True)
        b = res["bars"]
        cks = res["checkpoints"]
        M = lambda ck: res["checkpoint"][ck]["main"]["B"]
        chk(b["a"]["fires"] == all(M(ck)["fraction"] != "CONSTANT" and M(ck)["fraction"]["position"] >= 0.5 for ck in cks), f"{cohort} bar a")
        chk(b["b"]["fires"] == all(M(ck)["cells"]["5,0"]["sign"]["positive"] >= 18 for ck in cks), f"{cohort} bar b")
        chk(b["c"]["fires"] == all(M(ck)["cells"]["5,0"]["most_explanatory_position"] == 1 for ck in cks), f"{cohort} bar c")
        chk(b["d"]["fires"] == all(M(ck)["cells"]["4,1"]["mean"] > M(ck)["cells"]["4,0"]["mean"] for ck in cks), f"{cohort} bar d")
    rec = {"prereg": "MATH-CYBER-1-RENDER-ATLAS-NESTED-SWAP-DECOMPOSITION-0", "verdict": "VERIFIED" if not D else "DISCREPANCIES",
           "discrepancies": D[:40], "n_discrepancies": len(D), "inputs": inputs, "instrument_source_sha256": src_sha,
           "instrument_receipt_sha256": sha(OUTDIR / "nestedswap_receipt.json"), "instrument_run_commit": inst["start"]["start_commit"],
           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "status_porcelain": subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout,
           "verifier_sha256": sha(__file__)}
    (OUTDIR / "verify_receipt.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1)[:1500])


if __name__ == "__main__":
    main()
