"""Independent verifier for WRITER-TRAJECTORY-CENSUS-0 (pre-reg RESULTS
L67576, sealed by AMENDMENT -SEAL L67810). Shares no computation with
writertraj_census.py or writertraj_depend.py: its own literal tensor
law, its own flatten / cosine / divergence code, its own W_0
regeneration, its own bar evaluation. Recomputes every GLOBAL, BLOCK,
OUTSIDE and CLASS quantity at every step for every pair from the
checkpoint files, all four S0 bars and the matched-time set, the
dependence values from the gate rows (gate dicts summing, five levels,
each level <= 24), the swap losses and D-0 / D-1 / D-2, and checks the
instrument sources by git show at the census commit. Writes
logs/writertraj0/verify_receipt.json (refuses to overwrite).

Env: SMOKE=1 verifies the smoke rows and appends a kind=verify row to
logs/writertraj0/smoke.jsonl.

Usage: .venv/bin/python scratch/writertraj_verify.py
"""
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import torch  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
OUT = Path("logs/writertraj0")
T = 15_420
REPAIR = Path("/Users/artin/code/llmopt-repair")
SOURCES = ["scratch/writertraj_census.py", "scratch/writertraj_depend.py", "scratch/atomtraj_pins.py"]
D = []


def chk(c, m):
    if not c:
        D.append(m)


def close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def digest(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().to("cpu", torch.float32).contiguous()
        h.update(f"{k}:{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def keys_law():
    ks = []
    for l in range(8):
        ks += [f"blocks.{l}.{t}" for t in ("qkv.weight", "o.weight", "gate.weight", "up.weight", "down.weight", "n1.g", "n2.g")]
    ks += ["emb.weight", "head.weight", "norm.g"]
    sets = {"GLOBAL": sorted(ks), "OUTSIDE": ["emb.weight", "head.weight", "norm.g"]}
    for l in range(8):
        sets[f"BLOCK{l}"] = sorted(k for k in ks if k.startswith(f"blocks.{l}."))
    cls = {"qkv": [], "o": [], "gate": [], "up": [], "down": [], "norms": [], "emb": [], "head": []}
    for k in ks:
        if k.startswith("blocks."):
            t = k.split(".")[2]
            cls[t if t in ("qkv", "o", "gate", "up", "down") else "norms"].append(k)
        elif k == "emb.weight":
            cls["emb"].append(k)
        elif k == "head.weight":
            cls["head"].append(k)
        else:
            cls["norms"].append(k)
    for c, v in cls.items():
        sets[f"CLASS_{c}"] = sorted(v)
    return sorted(ks), sets


def load(p):
    d = torch.load(p, map_location="cpu")
    return d["model"] if isinstance(d, dict) and "model" in d else d


def flat(sd, keys):
    return torch.cat([sd[k].detach().double().reshape(-1) for k in keys])


def cosf(a, b):
    na, nb = float(a.norm()), float(b.norm())
    return float((a @ b) / (na * nb)) if na > 0 and nb > 0 else None


def regen(seed):
    import train_mathnative as TM
    tok = TM.MathTokenizer()
    torch.manual_seed(seed)
    m = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def main():
    if not SMOKE:
        chk(not (OUT / "verify_receipt.json").exists(), "REFUSE OVERWRITE")
        if D:
            raise SystemExit(D[-1])
        census = json.load(open(OUT / "census.json"))
        depend = json.load(open(OUT / "depend.json"))
        gates = [json.loads(l) for l in open(OUT / "gates.jsonl")]
    else:
        rows = [json.loads(l) for l in open(OUT / "smoke.jsonl")]
        census = [r for r in rows if r.get("kind") == "census"][-1]
        depend = [r for r in rows if r.get("kind") == "depend"][-1]
        gates = [r for r in rows if r.get("kind") == "gate"]
    KEYS, SETS = keys_law()
    chk(sorted(sum([SETS[f"BLOCK{l}"] for l in range(8)], []) + SETS["OUTSIDE"]) == KEYS, "BLOCK+OUTSIDE partition")
    chk(sorted(sum([SETS[f"CLASS_{c}"] for c in ("qkv", "o", "gate", "up", "down", "norms", "emb", "head")], [])) == KEYS, "CLASS partition")
    chk({k: len(v) for k, v in SETS.items()} == census["tensor_law"], "tensor law sizes v census")
    commit = census["commit"]
    chk(not census["tree_dirty"] and depend["commit"] == commit, "commit identity census v depend, clean tree")
    src = {}
    for p in SOURCES:
        r = subprocess.run(["git", "show", f"{commit}:{p}"], capture_output=True)
        chk(r.returncode == 0, f"source {p} absent at {commit}")
        src[p] = hashlib.sha256(r.stdout).hexdigest()
    # W_0 law
    w0_2 = regen(2)
    chk(digest(w0_2) == census["w0_seed2_state_digest"], "seed-2 W_0 digest v census")
    chk(census["w0_law"]["vocab_len"] == 40 and census["w0_law"]["VOCAB_EXTRA"] == "", "W_0 law fields")
    # recompute pairs
    def paths_of(pair):
        if pair == "A_B":
            A = {int(s): Path(p) for s, p in census["artifacts"]["A_paths"].items()}
            B = {int(s): Path(p) for s, p in census["artifacts"]["B_paths"].items()}
            return A, B, w0_2, w0_2
        cell = pair[len("NULL_"):]
        X = {s: Path("checkpoints/atomtraj1") / cell / f"step_{s:05d}.pt" for s in [463] + list(range(1028, T + 1, 1028))}
        Y = {s: REPAIR / "checkpoints/atomtraj1" / cell / f"step_{s:05d}.pt" for s in X}
        w0x, w0y = load(Path("checkpoints/atomtraj1") / cell / "step_00000.pt"), load(REPAIR / "checkpoints/atomtraj1" / cell / "step_00000.pt")
        chk(digest(w0x) == digest(w0y) == census["artifacts"][f"null_{cell}_w0_digest"], f"{cell} W_0 digests")
        return X, Y, w0x, w0y
    mine = {}
    for pair, pc in census["pairs"].items():
        X, Y, w0x, w0y = paths_of(pair)
        common = [s for s in sorted(X) if s in Y]
        chk(common == pc["steps"], f"{pair} step set")
        prev, prevv = {}, {}
        mine[pair] = {"steps": common, "per_set": {s: {} for s in SETS}}
        for s in common:
            sdX, sdY = load(X[s]), load(Y[s])
            for sname, keys in SETS.items():
                dX, dY = flat(sdX, keys) - flat(w0x, keys), flat(sdY, keys) - flat(w0y, keys)
                r = pc["per_set"][sname][str(s)]
                nX, nY = float(dX.norm()), float(dY.norm())
                got = {"C": cosf(dX, dY), "normX": nX, "normY": nY, "R": float((dX - dY).norm() / ((nX + nY) / 2))}
                if sname in prev:
                    s0, pX, pY = prev[sname]
                    vX, vY = (dX - pX) / (s - s0), (dY - pY) / (s - s0)
                    got.update({"h": s - s0, "V": cosf(vX, vY), "vnormX": float(vX.norm()), "vnormY": float(vY.norm())})
                    if sname in prevv:
                        s1, qX, qY = prevv[sname]
                        aX, aY = (vX - qX) / (s - s1), (vY - qY) / (s - s1)
                        got.update({"A": cosf(aX, aY), "anormX": float(aX.norm()), "anormY": float(aY.norm())})
                    prevv[sname] = (s, vX, vY)
                prev[sname] = (s, dX, dY)
                for k, v in got.items():
                    chk(close(v, r.get(k)) if k != "h" else r.get("h") == v, f"{pair} {sname} step {s} {k}")
                mine[pair]["per_set"][sname][s] = got
            print(f"[verify] {pair} step {s} ok; discrepancies so far {len(D)}", flush=True)
    if not SMOKE:
        # bars
        AB = mine["A_B"]; nulls = {p: m for p, m in mine.items() if p.startswith("NULL_")}
        fA = max(AB["steps"]); chk(fA == T, "writer final step 15,420")
        C_AB = AB["per_set"]["GLOBAL"][fA]["C"]; R_AB = AB["per_set"]["GLOBAL"][fA]["R"]
        C_null = [n["per_set"]["GLOBAL"][max(n["steps"])]["C"] for n in nulls.values()]
        R_null = [n["per_set"]["GLOBAL"][max(n["steps"])]["R"] for n in nulls.values()]
        match_set = [s for s in [463] + list(range(1028, T + 1, 1028)) if 0.1 <= s / T <= 0.9]
        chk(len(match_set) == 12, "12 matched times")
        def near(t, grid):
            return min(grid, key=lambda g: (abs(g - t), g))
        def mid(pair_m):
            g = pair_m["per_set"]["GLOBAL"]; own = [s for s in pair_m["steps"] if "V" in g[s]]
            return median([g[near(t, own)]["V"] for t in match_set])
        V_AB = mid(AB); V_null = [mid(n) for n in nulls.values()]
        b = census["bars"]
        chk(close(C_AB, b["C_AB_final"]) and all(close(x, y) for x, y in zip(C_null, b["C_null_final"])), "C finals v census")
        chk(close(R_AB, b["R_AB_final"]) and all(close(x, y) for x, y in zip(R_null, b["R_null_final"])), "R finals v census")
        chk(close(V_AB, b["V_AB_mid_median"]) and all(close(x, y) for x, y in zip(V_null, b["V_null_mid_median"])), "V medians v census")
        s01, s02, s03 = C_AB < min(C_null), V_AB < min(V_null), R_AB > max(R_null)
        chk(b["S0-1"] == s01 and b["S0-2"] == s02 and b["S0-3"] == s03 and b["TRAJECTORY"] == (s01 and s03), "S0 bar booleans")
        cnt = 0
        for l in range(8):
            cb = AB["per_set"][f"BLOCK{l}"][fA]["C"]; nb = min(n["per_set"][f"BLOCK{l}"][max(n["steps"])]["C"] for n in nulls.values())
            chk(close(cb, b["S0-4_blocks"][str(l)]["C_AB"]) and close(nb, b["S0-4_blocks"][str(l)]["null_min"]), f"block {l} S0-4")
            cnt += cb < nb
        chk(cnt == b["S0-4_count"], "S0-4 count")
        # dependence from gate rows
        g = {r["label"]: r for r in gates}
        for r in gates:
            chk(sum(r["solves"].values()) == r["total"] and sorted(r["solves"]) == ["3", "4", "5", "6", "7"] and all(0 <= v <= 24 for v in r["solves"].values()), f"gate dict {r['label']}")
            chk(r["code_commit"] == commit and not r["smoke"], f"gate provenance {r['label']}")
        full = {n: g[f"{n}/full"]["total"] for n in ("A", "B", "N1", "N2")}
        chk(full == depend["full"], "full gates v depend")
        revert_groups = [k for k in SETS if k != "GLOBAL"]      # 8 blocks + OUTSIDE + 8 classes = 17
        chk(len(revert_groups) == 17, "17 revert groups")
        dep = {n: {grp: full[n] - g[f"{n}/revert/{grp}"]["total"] for grp in revert_groups} for n in full}
        for n in full:
            for grp, v in dep[n].items():
                chk(depend["dependence"][n].get(grp) == v, f"dependence {n} {grp}")
        groups9 = [f"BLOCK{l}" for l in range(8)] + ["OUTSIDE"]; groups8 = [k for k in SETS if k.startswith("CLASS_")]
        def dist(x, y, gs):
            return sum((dep[x][k] - dep[y][k]) ** 2 for k in gs) ** 0.5
        band = max(full.values()) - min(full.values()); d0 = band <= 7
        swaps = {k: g[k]["total"] - full[k.split("<-")[0]] for k in depend["swap_loss"]}
        chk(swaps == depend["swap_loss"] and len(swaps) == 32, "swap losses v depend")
        cross = [v for k, v in swaps.items() if k.startswith("A<-B") or k.startswith("B<-A")]
        null = [v for k, v in swaps.items() if k.startswith("N1<-") or k.startswith("N2<-")]
        db = depend["bars"]
        chk(db["D-0"] == d0 and db["band"] == band, "D-0")
        chk(close(dist("A", "B", groups9), db["profile_dist_9group_AB"]) and close(dist("N1", "N2", groups9), db["profile_dist_9group_N1N2"]), "9-group distances")
        chk(close(dist("A", "B", groups8), db["profile_dist_8class_AB"]) and close(dist("N1", "N2", groups8), db["profile_dist_8class_N1N2"]), "8-class distances")
        d1 = (dist("A", "B", groups9) > dist("N1", "N2", groups9)) if d0 else None
        d2 = (statistics.median(cross) < min(null)) if d0 else None
        chk(db["D-1"] == d1 and db["D-2"] == d2, "D-1 / D-2 booleans")
        bars = {"S0-1": s01, "S0-2": s02, "S0-3": s03, "S0-4_count": cnt, "TRAJECTORY": s01 and s03, "D-0": d0, "D-1": d1, "D-2": d2,
                "C_AB_final": C_AB, "C_null_final": C_null, "R_AB_final": R_AB, "R_null_final": R_null, "V_AB": V_AB, "V_null": V_null}
    else:
        bars = {"smoke": True}
    rec = {"kind": "verify", "prereg": "WRITER-TRAJECTORY-CENSUS-0", "smoke": SMOKE, "verdict": "VERIFIED" if not D else "DISCREPANCIES",
           "n_discrepancies": len(D), "discrepancies": D[:60], "census_commit": commit, "source_sha256_at_commit": src, "bars": bars,
           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "status_porcelain": subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout,
           "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if SMOKE:
        with (OUT / "smoke.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
    else:
        (OUT / "verify_receipt.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("verdict", "n_discrepancies", "discrepancies")}, indent=1))


if __name__ == "__main__":
    main()
