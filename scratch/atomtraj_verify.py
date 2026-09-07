"""Independent verifier for ATOM-DIET-TRAJECTORY-1 (pre-reg RESULTS
L66546, sealed by AMENDMENT -SEAL L66822). Shares no code with
scratch/atomtraj_census.py: its own literal key law, its own digest of
each snapshot, its own distances, bars and descriptive tables. Checks
the instrument sources against the launch commit (git show, never the
working tree), recomputes the six stream digests and the counts from the
frozen curric functions, checks completeness, the state_digest equality
laws, the frozen execution order against recorded start times, the gate
dict sums, and every census number within 1e-9 relative. Writes
logs/atomtraj1/verify_receipt.json (refuses to overwrite).

Env: SMOKE=1 verifies the plumbing on the smoke rows and appends a
kind=verify row to logs/atomtraj1/smoke.jsonl.

Usage: .venv/bin/python scratch/atomtraj_verify.py
"""
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

import torch  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
LOGS = Path("logs/atomtraj1")
SEEDS = (5, 6, 7)
FINAL = 15_420
GRID = 1_028
LAGS = (1_028, 2_056, 3_084, 5_140)
SCHEDULE = [0, 463] + list(range(1_028, FINAL + 1, 1_028))
GATE_STEPS = (2_056, 5_140, 10_280, 15_420)
ORDER = ["stock_s5", "atoms_s5", "stock_s6", "atoms_s6", "atoms_s7", "stock_s7"]
SOURCES = ["scratch/birth19m_atoms_traj.py", "scratch/birth19m_atoms_trajgate.py", "scratch/atomtraj_census.py", "scratch/atomtraj_pins.py"]
PINNED = {"stock": ["18a6c14a6e7cd48e871e1645bfef1f6c89297681531ee34cc48740665ea003f7",
                    "a3bf1ab910faf09f1cdd1455dd54a16474eb4cf4433deb96b1e4607c7cd18324",
                    "7ff1b63ea45b343a94ba2092deeaf3da4abbd006085a190f0167781f579d433b"],
          "atoms": ["29f35e9931effbdb1db45378a2be80a4b0fbe44e1d26bc1e03492faa808699e8",
                    "b07e1e71ff24c90f8478f536bc95ff2b7aef2dc80761045b285b42da1032f222",
                    "3fc36a33bcc257eb6941bd237d1a7ff3d0f81758add787ea7a0f2c3517779a35"]}
BLOCK_KEYS = [f"blocks.{l}.{t}" for l in range(8) for t in ("qkv.weight", "o.weight", "gate.weight", "up.weight", "down.weight")]
CLASS_OF = {}
for _l in range(8):
    for _t in ("qkv", "o", "gate", "up", "down"):
        CLASS_OF[f"blocks.{_l}.{_t}.weight"] = _t
    CLASS_OF[f"blocks.{_l}.n1.g"] = "norms"
    CLASS_OF[f"blocks.{_l}.n2.g"] = "norms"
CLASS_OF["norm.g"] = "norms"
CLASS_OF["emb.weight"] = "emb"
CLASS_OF["head.weight"] = "head"
CLASS_ORDER = ["qkv", "o", "gate", "up", "down", "norms", "emb", "head"]
D = []


def chk(c, m):
    if not c:
        D.append(m)


def digest(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().to("cpu", torch.float32).contiguous()
        h.update(f"{k}:{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def my_objects(sd, sd0):
    e = [0.0] * 8
    ce = {c: 0.0 for c in CLASS_ORDER}
    chk(set(sd) == set(CLASS_OF), "state dict keys v literal class law")
    for k in sd:
        d = (sd[k].double() - sd0[k].double())
        en = float((d * d).sum())
        ce[CLASS_OF[k]] += en
        if k in BLOCK_KEYS:
            e[int(k.split(".")[1])] += en
    tot = sum(e)
    ct = sum(ce.values())
    if tot == 0 or ct == 0:
        return None, None, None
    p8 = [x / tot for x in e]
    return p8, [ce[c] / ct for c in CLASS_ORDER], sum(i * x for i, x in enumerate(p8))


def l2(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def main():
    receipt_path = LOGS / ("smoke.jsonl" if SMOKE else "verify_receipt.json")
    if not SMOKE:
        chk(not receipt_path.exists(), "REFUSE OVERWRITE verify_receipt.json")
        if D:
            raise SystemExit(D[-1])
        births = [json.loads(l) for l in (LOGS / "births.jsonl").open()]
        gates = [json.loads(l) for l in (LOGS / "gates.jsonl").open()]
        census = json.load(open(LOGS / "census.json"))
        tens = json.load(open(LOGS / "census_tensors.json"))
        chk(hashlib.sha256((LOGS / "census_tensors.json").read_bytes()).hexdigest() == census["census_tensors_sha256"], "census_tensors sha v census.json")
    else:
        rows = [json.loads(l) for l in (LOGS / "smoke.jsonl").open()]
        births = [r for r in rows if r.get("kind") == "birth" and r.get("emit") and r.get("device") != "cpu"][-1:]
        gates = [r for r in rows if r.get("kind") == "gate"]
        census = [r for r in rows if r.get("kind") == "census"][-1]
        tens = census["census_tensors"]
    # provenance and sources
    commits = {r["code_commit"] for r in births}
    chk(len(commits) == 1, f"code_commit not unique across births: {commits}")
    launch = births[0]["code_commit"]
    chk(all(not r["tree_dirty"] for r in births), "a birth ran on a dirty tree")
    chk(all(r["smoke"] == SMOKE for r in births), "smoke flag v mode")
    src_sha = {}
    for p in SOURCES:
        shown = subprocess.run(["git", "show", f"{launch}:{p}"], capture_output=True)
        chk(shown.returncode == 0, f"source {p} absent at launch commit {launch}")
        src_sha[p] = hashlib.sha256(shown.stdout).hexdigest()
    # stream pins recomputed from the frozen functions
    os.environ["ARM"] = "off"
    os.environ.setdefault("BIRTH_SEED", "3")
    import birth19m_curric as C
    import train_mathnative as TM
    from tenet_d2_revdiet import gate_band_exprs, norm
    tok = TM.MathTokenizer()
    stock_rows = C.load_excised_rows()
    enc_stock, _ = C.encode_with_levels(stock_rows, tok)
    shard = Path("data/micromodel_atoms_shard0.jsonl")
    atoms = [json.loads(l) for l in shard.open()]
    band = set(gate_band_exprs())
    atoms = [r for r in atoms if norm(str(r["cur"])) not in band and norm(str(r["nxt"])) not in band]
    trip = []
    for r in stock_rows + atoms:
        t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
        try:
            ids = tok.encode(t) + [tok.eos_id]
        except ValueError:
            continue
        if len(ids) <= 512:
            trip.append((ids, r.get("source") == "atom-oneply"))
    trip.sort(key=lambda p: len(p[0]))
    enc_atoms = [p[0] for p in trip]
    is_atom = [p[1] for p in trip]
    spe = len(C.stock_epoch_stream(len(enc_stock), 0))
    chk(len(enc_stock) == 164_490 and len(enc_atoms) == 170_490 and spe == 5_140, "enc sizes / steps per epoch")
    my_streams = {}
    for arm, enc, fl in (("stock", enc_stock, [False] * len(enc_stock)), ("atoms", enc_atoms, is_atom)):
        my_streams[arm] = []
        for ep in range(3):
            raw = C.stock_epoch_stream(len(enc), ep)
            st = raw[:spe]
            h = hashlib.sha256()
            for a, b in st:
                for j in range(a, b):
                    h.update(json.dumps(enc[j]).encode())
                h.update(b"|")
            my_streams[arm].append((h.hexdigest(), sum(1 for a, b in st for j in range(a, b) if fl[j]), len(raw) - spe))
        chk([x[0] for x in my_streams[arm]] == PINNED[arm], f"{arm} stream digests v pinned")
    chk([x[2] for x in my_streams["stock"]] == [0, 0, 0] and [x[2] for x in my_streams["atoms"]] == [187, 187, 187], "dropped counts v pinned")
    shard_sha = hashlib.sha256(shard.read_bytes()).hexdigest()
    for r in births:
        arm = r["arm"]
        chk(r["stream_sha256"] == [x[0] for x in my_streams[arm]], f"{arm} s{r['seed']} receipt stream digests")
        n_ep = len(r["atom_rows_per_epoch"])
        chk(n_ep == 3 or SMOKE, f"{arm} s{r['seed']} atom_rows_per_epoch length {n_ep}")
        chk(r["atom_rows_per_epoch"] == [x[1] for x in my_streams[arm]][:n_ep], f"{arm} s{r['seed']} atom rows")
        chk(r["dropped_per_epoch"] == [x[2] for x in my_streams[arm]], f"{arm} s{r['seed']} dropped counts")
        chk(r["shard_sha256"] == shard_sha, "shard sha in receipt")
    # completeness, equality laws, order
    by = {f"{r['arm']}_s{r['seed']}": r for r in births}
    if not SMOKE:
        chk(sorted(by) == sorted(ORDER), f"births present {sorted(by)}")
        starts = sorted(by, key=lambda k: by[k]["started_utc"])
        chk(starts == ORDER, f"execution order by start time {starts}")
        chk([by[k]["order_index"] for k in ORDER] == [str(i) for i in range(6)], "order_index fields")
        for s in SEEDS:
            chk(by[f"stock_s{s}"]["snapshots"]["0"]["state_digest"] == by[f"atoms_s{s}"]["snapshots"]["0"]["state_digest"], f"seed {s} step_0 state_digest across arms")
        chk(len(gates) == 24 and all(not g["smoke"] for g in gates), "24 real gate rows")
        chk(sorted((g["arm"], g["seed"], g["step"]) for g in gates) == sorted((a, s, st) for a in ("stock", "atoms") for s in SEEDS for st in GATE_STEPS), "gate cells")
    objs = {}
    for key, r in by.items():
        outdir = Path(r["outdir"])
        expect = [s for s in SCHEDULE if s <= r["steps"]] + ([r["steps"]] if r["steps"] not in SCHEDULE else [])
        chk(sorted(int(x) for x in r["snapshots"]) == sorted(expect), f"{key} snapshot set")
        chk(r["steps"] == (FINAL if not SMOKE else r["steps"]), f"{key} steps")
        sd0 = None
        objs[key] = {}
        flats = {}
        for s in sorted(int(x) for x in r["snapshots"]):
            p = outdir / f"step_{s:05d}.pt"
            chk(p.exists(), f"{key} missing {p}")
            if not p.exists():
                continue
            rec = r["snapshots"][str(s)]
            chk(hashlib.sha256(p.read_bytes()).hexdigest() == rec["file_sha256"], f"{key} step {s} file sha")
            sd = torch.load(p, map_location="cpu")
            chk(digest(sd) == rec["state_digest"], f"{key} step {s} state_digest")
            if s == 0:
                sd0 = sd
                chk(rec["state_digest"] == r["init_state_digest"], f"{key} init digest")
            p8, c8, cen = my_objects(sd, sd0)
            objs[key][s] = (p8, c8, cen)
            if s % GRID == 0 and s > 0:
                flats[s] = torch.cat([sd[k].double().reshape(-1) for k in BLOCK_KEYS])
            co = census["snapshot_objects"][key][str(s)]
            if p8 is None:
                chk(co["P8"] is None and co["C8"] is None and co["centroid"] is None, f"{key} step {s} zero-delta objects must be null")
            else:
                chk(co["P8"] is not None and all(close(a, b) for a, b in zip(p8, co["P8"])), f"{key} step {s} P8 v census")
                chk(co["C8"] is not None and all(close(c8[i], co["C8"][c]) for i, c in enumerate(CLASS_ORDER)), f"{key} step {s} C8 v census")
                chk(close(cen, co["centroid"]), f"{key} step {s} centroid v census")
            # per-tensor descriptive tables
            for k in BLOCK_KEYS:
                Dm = sd[k].double() - sd0[k].double()
                S = torch.linalg.svdvals(Dm)
                rank = int((S > 0).sum())
                te = tens[f"{key}/{s}"][k]
                chk(te["rank"] == rank and close(te["energy"], float((Dm * Dm).sum())), f"{key} {s} {k} energy/rank")
                if rank > 0:
                    U = torch.linalg.svd(Dm, full_matrices=False)[0]
                    kk = min(16, rank)
                    ipr = float((U[:, :kk] ** 4).sum(0).mean()) * Dm.shape[0]
                    pr = S[:rank] / S[:rank].sum()
                    eff = float(torch.exp(-(pr * pr.log()).sum()))
                    chk(close(te["ipr_top16_x_rows"], ipr, 1e-6) and close(te["effrank"], eff), f"{key} {s} {k} ipr/effrank")
        fp = outdir / "final.pt"
        chk(fp.exists() and hashlib.sha256(fp.read_bytes()).hexdigest() == r["final"]["file_sha256"], f"{key} final file sha")
        if fp.exists():
            chk(digest(torch.load(fp, map_location="cpu")) == r["final"]["state_digest"] == r["snapshots"][str(r["steps"])]["state_digest"], f"{key} state_digest(step_final) == state_digest(final)")
        for h in LAGS:
            for t in sorted(flats):
                if t + h in flats and t + 2 * h in flats:
                    d1, d2 = flats[t + h] - flats[t], flats[t + 2 * h] - flats[t + h]
                    n1, n2 = float(d1.norm()), float(d2.norm())
                    mine = float((d1 @ d2) / (n1 * n2)) if n1 > 0 and n2 > 0 else None
                    chk(close(mine, census["lag"][key][str(h)].get(str(t))), f"{key} lag {h} t {t}")
        flats.clear()
        print(f"[verify] {key} checked; discrepancies so far {len(D)}", flush=True)
    if not SMOKE:
        chk("0" not in census["bars_by_step"], "census must not evaluate bars at step 0 (zero delta)")
        for s in [st for st in SCHEDULE if st > 0]:
            P = {(a, sd_): objs[f"{a}_s{sd_}"][s] for a in ("stock", "atoms") for sd_ in SEEDS}
            T = {sd_: l2(P[("atoms", sd_)][0], P[("stock", sd_)][0]) for sd_ in SEEDS}
            W = {(a, x, y): l2(P[(a, x)][0], P[(a, y)][0]) for a in ("stock", "atoms") for x, y in itertools.combinations(SEEDS, 2)}
            Tc = {sd_: l2(P[("atoms", sd_)][1], P[("stock", sd_)][1]) for sd_ in SEEDS}
            Wc = {(a, x, y): l2(P[(a, x)][1], P[(a, y)][1]) for a in ("stock", "atoms") for x, y in itertools.combinations(SEEDS, 2)}
            dp = {sd_: [x - y for x, y in zip(P[("atoms", sd_)][0], P[("stock", sd_)][0])] for sd_ in SEEDS}
            cs = []
            for x, y in itertools.combinations(SEEDS, 2):
                na, nb = math.sqrt(sum(v * v for v in dp[x])), math.sqrt(sum(v * v for v in dp[y]))
                cs.append(None if na == 0 or nb == 0 else sum(u * v for u, v in zip(dp[x], dp[y])) / (na * nb))
            s1 = min(T.values()) > max(W.values())
            s1b = all(any(v != 0 for v in dp[sd_]) for sd_ in SEEDS) and all(c is not None and c > 0 for c in cs)
            s2 = min(Tc.values()) > max(Wc.values())
            sg = {sd_: (P[("atoms", sd_)][2] > P[("stock", sd_)][2]) - (P[("atoms", sd_)][2] < P[("stock", sd_)][2]) for sd_ in SEEDS}
            s3 = len(set(sg.values())) == 1 and 0 not in sg.values()
            cb = census["bars_by_step"][str(s)]
            chk(cb["S1"] == s1 and cb["S1b"] == s1b and cb["S2"] == s2 and cb["S3"] == s3, f"step {s} bar booleans v census")
            chk(all(close(T[sd_], cb["T"][str(sd_)]) for sd_ in SEEDS) and all(close(W[(a, x, y)], cb["W"][f"{a}_{x}_{y}"]) for (a, x, y) in W), f"step {s} T/W v census")
            chk(all(close(Tc[sd_], cb["T_C8"][str(sd_)]) for sd_ in SEEDS) and all(close(Wc[(a, x, y)], cb["W_C8"][f"{a}_{x}_{y}"]) for (a, x, y) in Wc), f"step {s} C8 T/W v census")
            chk(all(close(c, cb["cosines"][f"{x}_{y}"]) for c, (x, y) in zip(cs, itertools.combinations(SEEDS, 2))), f"step {s} cosines v census")
            chk({str(k): v for k, v in sg.items()} == cb["centroid_sign"], f"step {s} centroid signs v census")
            if s == FINAL:
                final_bars = {"S1": s1, "S1b": s1b, "S2": s2, "S3": s3, "T": T, "W": {f"{a}_{x}_{y}": v for (a, x, y), v in W.items()}}
        g = {}
        for r in gates:
            chk(sum(r["solves"].values()) == r["total"] and sorted(r["solves"]) == ["3", "4", "5", "6", "7"], f"gate dict {r['arm']} s{r['seed']} {r['step']}")
            chk(all(0 <= v <= 24 for v in r["solves"].values()), f"gate level counts within 24 items {r['arm']} s{r['seed']} {r['step']}")
            chk(r["birth_code_commit"] == launch and r["code_commit"] == launch, "gate row commits v launch commit")
            g[(r["arm"], r["seed"], r["step"])] = r
        deltas = {sd_: g[("atoms", sd_, FINAL)]["total"] - g[("stock", sd_, FINAL)]["total"] for sd_ in SEEDS}
        f1 = all(d > 0 for d in deltas.values()) and sum(deltas.values()) / 3 >= 5
        f2 = all(g[("atoms", sd_, FINAL)]["solves"]["4"] >= g[("stock", sd_, FINAL)]["solves"]["4"] for sd_ in SEEDS)
        cbars = census["bars"]
        chk(cbars["F1"] == f1 and cbars["F2"] == f2 and cbars["paired_deltas"] == {str(k): v for k, v in deltas.items()}, "F1/F2 v census")
        structural = final_bars["S1"] and final_bars["S1b"]
        chk(cbars["STRUCTURAL"] == structural and cbars["FUNCTION"] == f1 and cbars["JOINT"] == (structural and f1), "claim axes v census")
        final_bars.update({"F1": f1, "F2": f2, "paired_deltas": deltas, "STRUCTURAL": structural, "FUNCTION": f1, "JOINT": structural and f1})
    else:
        final_bars = {"smoke": True}
    rec = {"kind": "verify", "prereg": "ATOM-DIET-TRAJECTORY-1", "smoke": SMOKE, "verdict": "VERIFIED" if not D else "DISCREPANCIES",
           "n_discrepancies": len(D), "discrepancies": D[:60], "launch_commit": launch, "source_sha256_at_launch": src_sha,
           "bars": {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in final_bars.items()},
           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "status_porcelain": subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout,
           "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if SMOKE:
        with receipt_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    else:
        receipt_path.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("verdict", "n_discrepancies", "discrepancies", "launch_commit")}, indent=1))


if __name__ == "__main__":
    main()
