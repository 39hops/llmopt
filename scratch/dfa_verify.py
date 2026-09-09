"""Independent verifier for WRITER-DFA-1 (pre-reg RESULTS L68321, sealed
L68543 / L68644). Shares no computation with the instruments: its own
literal tensor law, flatten / cosine / divergence, W_0 regeneration,
feedback regeneration, bar evaluation against the literal thresholds.
Checks, from the files: the two discovery birth receipts (17 snapshot
file shas and state digests re-read, final == step_15420, shared W_0 ==
seed-2 regeneration, stream digests == the pinned stock digests, launch
commit == code_commit and clean tree, feedback matrices regenerated
from the seed law with the frozen s and digest-matched, the DFA (s, lr)
== the frozen qualification selection, control lr 3e-4); the
qualification receipts (four births, gate dicts, the selection law
re-applied); census.json (C and R at 15,420 recomputed, T-1); gates.jsonl
(dicts sum, five levels, unique labels, provenance, band recomputed,
dependence profiles and distances, swap losses and median, DEP-DEPTH /
DEP-CLASS / COMPAT against 9.85 / 10.34 / -7, one revert and one swap
reconstructed by digest); depthclass.json when present (80 rows, cells
recomputed, one cell reconstructed); act.json (distance and ACT-1
recomputed against act_envelope.json, self-checks, digests);
align.json (digests, feedback digest, cosines in [-1, 1]). Writes
logs/writerdfa1/verify_receipt.json (refuses to overwrite).
"""
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import torch  # noqa: E402

OUT = Path("logs/writerdfa1")
T = 15_420
STOCK_STREAMS = ["18a6c14a6e7cd48e871e1645bfef1f6c89297681531ee34cc48740665ea003f7",
                 "a3bf1ab910faf09f1cdd1455dd54a16474eb4cf4433deb96b1e4607c7cd18324",
                 "7ff1b63ea45b343a94ba2092deeaf3da4abbd006085a190f0167781f579d433b"]
THRESH = {"C_max": 0.7771, "R_min": 0.6677, "depth": 9.85, "cls": 10.34, "swap_median": -7, "band": 7}
SOURCES = ["scratch/birth19m_dfa.py", "scratch/dfa_credit.py", "scratch/dfa_act.py", "scratch/dfa_align.py", "scratch/dfa_trajcensus.py",
           "scratch/writertraj_depend.py", "scratch/dfa_depthclass.py", "scratch/dfa_qualgate.py", "scratch/dfa_probe.py", "scratch/atomtraj_pins.py"]
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


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    d = torch.load(p, map_location="cpu")
    return d["model"] if isinstance(d, dict) and "model" in d else d


def regen(seed):
    import train_mathnative as TM
    torch.manual_seed(seed)
    m = TM.build_model(len(TM.MathTokenizer().vocab), d=384, layers=8, heads=6, ffn=1536)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def feedback(s):
    h = hashlib.sha256()
    for l in range(8):
        g = torch.Generator("cpu").manual_seed(31_000_000 + l)
        B = (torch.rand(384, 40, generator=g) * 2 - 1) * (s / math.sqrt(40))
        h.update(f"B{l}:{tuple(B.shape)}|".encode())
        h.update(B.contiguous().numpy().tobytes())
    return h.hexdigest()


def keys_law():
    keys = ["emb.weight", "head.weight", "norm.g"]
    for l in range(8):
        keys += [f"blocks.{l}.{t}" for t in ("qkv.weight", "o.weight", "gate.weight", "up.weight", "down.weight", "n1.g", "n2.g")]
    keys = sorted(keys)
    sets = {"GLOBAL": keys, "OUTSIDE": sorted(k for k in keys if not k.startswith("blocks."))}
    for l in range(8):
        sets[f"BLOCK{l}"] = sorted(k for k in keys if k.startswith(f"blocks.{l}."))
    for c in ("qkv", "o", "gate", "up", "down"):
        sets[f"CLASS_{c}"] = sorted(k for k in keys if k.endswith(f".{c}.weight"))
    sets["CLASS_norms"] = sorted(k for k in keys if k.endswith(".g"))
    sets["CLASS_emb"], sets["CLASS_head"] = ["emb.weight"], ["head.weight"]
    return keys, sets


def flat(sd, keys):
    return torch.cat([sd[k].double().reshape(-1) for k in keys])


def main():
    chk(not (OUT / "verify_receipt.json").exists(), "REFUSE OVERWRITE")
    if D:
        raise SystemExit(D[-1])
    KEYS, SETS = keys_law()
    births = [json.loads(l) for l in open(OUT / "births.jsonl") if '"kind": "birth"' in l]
    disc = {b["mode"]: b for b in births if b["phase"] == "disc"}
    chk(set(disc) == {"bp", "dfa"} and len(births) == 2, "exactly two discovery births")
    sel = json.load(open(OUT / "qual_selection.json"))
    commit = disc["dfa"]["code_commit"]
    for m_, b in disc.items():
        chk(b["seed"] == 2 and not b["smoke"] and b["emit"] and b["steps"] == T and b["steps_total"] == T, f"{m_} birth law")
        chk(b["code_commit"] == b["launch_commit"] == commit and not b["tree_dirty"], f"{m_} commit identity / clean tree")
        chk(b["stream_sha256"] == STOCK_STREAMS, f"{m_} stream digests")
        chk(sorted(int(s) for s in b["snapshots"]) == [0, 463] + list(range(1028, T + 1, 1028)), f"{m_} 17-snapshot schedule")
        for s, r in b["snapshots"].items():
            p = Path(b["outdir"]) / f"step_{int(s):05d}.pt"
            chk(p.exists() and sha(p) == r["file_sha256"], f"{m_} snapshot {s} sha")
            chk(digest(load(p)) == r["state_digest"], f"{m_} snapshot {s} state digest")
        chk(b["final"]["state_digest"] == b["snapshots"][str(T)]["state_digest"] == digest(load(Path(b["outdir"]) / "final.pt")), f"{m_} final == step_15420")
        chk(b["init_state_digest"] == b["snapshots"]["0"]["state_digest"], f"{m_} init digest")
    w0d = digest(regen(2))
    chk(disc["bp"]["init_state_digest"] == disc["dfa"]["init_state_digest"] == w0d, "shared W_0 == seed-2 regeneration")
    chk(disc["bp"]["peak_lr"] == 3e-4 and disc["bp"]["s"] is None and disc["bp"]["feedback"] is None, "control = stock recipe, no feedback")
    fb = disc["dfa"]["feedback"]
    chk(fb and fb["seed_base"] == 31_000_000 and feedback(fb["s"]) == fb["digest"], "feedback regenerated from the seed law")
    fbp = Path(disc["dfa"]["outdir"]) / "feedback.pt"
    chk(fbp.exists() and sha(fbp) == fb["file_sha256"], "feedback.pt sha")
    chk(sel["selected"] and (disc["dfa"]["s"], disc["dfa"]["peak_lr"]) == (sel["selected"]["s"], sel["selected"]["lr"]), "DFA (s, lr) == frozen selection")
    # qualification law re-applied
    qual = [json.loads(l) for l in open(OUT / "qual.jsonl")]
    qb = [r for r in qual if r.get("kind") == "birth"]
    qg = {(r["s"], r["peak_lr"]): r for r in qual if r.get("kind") == "gate"}
    chk(len(qb) == 4 and all(r["seed"] == 21 and r["mode"] == "dfa" and not r["smoke"] for r in qb), "four qualification births at seed 21")
    cells = [(1.0, 3e-4), (1.0, 1e-4), (0.25, 3e-4), (4.0, 1e-4)]
    chk(sorted((r["s"], r["peak_lr"]) for r in qb) == sorted(cells), "qualification cells")
    for r in qg.values():
        chk(sum(r["solves"].values()) == r["total"] and sorted(r["solves"]) == ["3", "4", "5", "6", "7"] and all(0 <= v <= 24 for v in r["solves"].values()), f"qual gate dict {r['cell']}")
    stable = [c for c in cells if c in qg and qg[c]["total"] > 0]
    order = {(1.0, 3e-4): 0, (0.25, 3e-4): 1, (1.0, 1e-4): 2, (4.0, 1e-4): 3}
    if stable:
        best = max(qg[c]["total"] for c in stable)
        pick = min([c for c in stable if qg[c]["total"] == best], key=lambda c: order[c])
        chk((sel["selected"]["s"], sel["selected"]["lr"], sel["selected"]["gate"]) == (pick[0], pick[1], best), "selection law re-applied")
    else:
        chk(sel["selected"] is None and sel["stop"] == "DFA-UNSTABLE", "no stable cell -> DFA-UNSTABLE")
    src = {}
    for p in SOURCES:
        r = subprocess.run(["git", "show", f"{commit}:{p}"], capture_output=True)
        chk(r.returncode == 0, f"source {p} absent at {commit}")
        src[p] = hashlib.sha256(r.stdout).hexdigest()
    # trajectory census
    census = json.load(open(OUT / "census.json"))
    pc = census["pairs"]["DFA_CTRL"]
    w0 = load(Path(disc["dfa"]["outdir"]) / "step_00000.pt")
    sx, sy = load(Path(disc["dfa"]["outdir"]) / f"step_{T:05d}.pt"), load(Path(disc["bp"]["outdir"]) / f"step_{T:05d}.pt")
    w0f = flat(w0, KEYS)
    dX, dY = flat(sx, KEYS) - w0f, flat(sy, KEYS) - w0f
    C = float((dX @ dY) / (dX.norm() * dY.norm()))
    R = float((dX - dY).norm() / ((float(dX.norm()) + float(dY.norm())) / 2))
    chk(close(C, census["bars"]["C_final"], 1e-6) and close(R, census["bars"]["R_final"], 1e-6), "C / R at 15,420 recomputed")
    t1 = C < THRESH["C_max"] and R > THRESH["R_min"]
    chk(census["bars"]["T-1"] == t1 and census["bars"]["thresholds"] == {"C_max": 0.7771, "R_min": 0.6677}, "T-1 boolean / thresholds")
    for l in range(8):
        cb = flat(sx, SETS[f"BLOCK{l}"]) - flat(w0, SETS[f"BLOCK{l}"]); cc = flat(sy, SETS[f"BLOCK{l}"]) - flat(w0, SETS[f"BLOCK{l}"])
        chk(close(float((cb @ cc) / (cb.norm() * cc.norm())), pc["per_set"][f"BLOCK{l}"][str(T)]["C"], 1e-6), f"BLOCK{l} C at 15,420")
    # dependence, band, swaps
    depend = json.load(open(OUT / "depend.json"))
    gates = [json.loads(l) for l in open(OUT / "gates.jsonl")]
    g = {r["label"]: r for r in gates}
    chk(len(g) == len(gates), "unique gate labels")
    for r in gates:
        chk(sum(r["solves"].values()) == r["total"] and sorted(r["solves"]) == ["3", "4", "5", "6", "7"] and all(0 <= v <= 24 for v in r["solves"].values()), f"gate dict {r['label']}")
        chk(r["code_commit"] == depend["commit"] and not r["smoke"], f"gate provenance {r['label']}")
    full = {n: g[f"{n}/full"]["total"] for n in ("DFA", "CTRL")}
    chk(full == depend["full"], "full gates")
    chk(g["DFA/full"]["state_digest"] == disc["dfa"]["final"]["state_digest"] and g["CTRL/full"]["state_digest"] == disc["bp"]["final"]["state_digest"], "full gates on the final states")
    band_pass = full["CTRL"] - THRESH["band"] <= full["DFA"] <= full["CTRL"] + THRESH["band"]
    bd = depend["band"]
    chk(bd["g_dfa"] == full["DFA"] and bd["c_ctrl"] == full["CTRL"] and bd["band_pass"] == band_pass and bd["half_width"] == 7, "FUNCTION-BAND recomputed")
    bars = {"T-1": t1, "C_final": C, "R_final": R, "band_pass": band_pass, "g": full["DFA"], "c": full["CTRL"]}
    groups9 = [f"BLOCK{l}" for l in range(8)] + ["OUTSIDE"]; groups8 = [k for k in SETS if k.startswith("CLASS_")]
    n_expected = 2 + (1 + 2 * 17 + 16 if band_pass else 0)
    dc_rows = [r for r in gates if r.get("op") == "revert_cell"]
    chk(len(gates) - len(dc_rows) == n_expected, f"gate count {len(gates) - len(dc_rows)} v expected {n_expected}")
    if band_pass:
        dep = {n: {grp: full[n] - g[f"{n}/revert/{grp}"]["total"] for grp in groups9 + groups8} for n in full}
        for n in full:
            chk(dep[n] == depend["dependence"][n], f"dependence {n}")
        d9 = sum((dep["DFA"][k] - dep["CTRL"][k]) ** 2 for k in groups9) ** 0.5
        d8 = sum((dep["DFA"][k] - dep["CTRL"][k]) ** 2 for k in groups8) ** 0.5
        swaps = {k: g[k]["total"] - full[k.split("<-")[0]] for k in depend["swap_loss"]}
        chk(swaps == depend["swap_loss"] and len(swaps) == 16, "swap losses")
        med = statistics.median(list(swaps.values()))
        db = depend["bars"]
        chk(close(d9, db["profile_dist_9group"]) and close(d8, db["profile_dist_8class"]) and close(med, db["cross_swap_median"]), "distances / median")
        chk(db["DEP-DEPTH"] == (d9 > THRESH["depth"]) and db["DEP-CLASS"] == (d8 > THRESH["cls"]) and db["COMPAT"] == (med < THRESH["swap_median"]), "DEP / COMPAT booleans")
        chk(db["thresholds"] == {"DEP-DEPTH": 9.85, "DEP-CLASS": 10.34, "COMPAT": -7}, "literal thresholds")
        w0g = g["W0_seed2"]
        chk(w0g["state_digest"] == w0d, "W_0 gate on the seed-2 regeneration")
        for n, sd in (("DFA", sx), ("CTRL", sy)):
            rv = {k: (w0[k].clone() if k in SETS["BLOCK3"] else v.clone()) for k, v in sd.items()}
            chk(digest(rv) == g[f"{n}/revert/BLOCK3"]["state_digest"], f"{n} revert BLOCK3 reconstruction")
        for rec_n, don_n, sr, sdn in (("DFA", "CTRL", sx, sy), ("CTRL", "DFA", sy, sx)):
            hy = {k: v.clone() for k, v in sr.items()}
            for k in SETS["BLOCK5"]:
                hy[k] = w0[k] + (sdn[k] - w0[k])
            chk(digest(hy) == g[f"{rec_n}<-{don_n}/BLOCK5"]["state_digest"], f"{rec_n}<-{don_n} swap BLOCK5 reconstruction")
        bars.update({"d9": d9, "d8": d8, "swap_median": med, "DEP-DEPTH": d9 > THRESH["depth"], "DEP-CLASS": d8 > THRESH["cls"], "COMPAT": med < THRESH["swap_median"]})
        # depth x class census (mandatory on band pass)
        dcp = OUT / "depthclass.json"
        chk(dcp.exists(), "depthclass.json present on band pass")
        if dcp.exists():
            dc = json.load(open(dcp))
            chk(len(dc_rows) == 80 and dc["n_gates"] == 80, "80 census gates")
            for n in ("DFA", "CTRL"):
                cells_ = {f"B{l}_{c}": full[n] - g[f"{n}/revert_cell/B{l}_{c}"]["total"] for l in range(8) for c in ("qkv", "o", "gate", "up", "down")}
                chk(cells_ == dc["table"][n], f"depth x class table {n}")
            rv = {k: v.clone() for k, v in sx.items()}
            rv["blocks.4.up.weight"] = w0["blocks.4.up.weight"].clone()
            chk(digest(rv) == g["DFA/revert_cell/B4_up"]["state_digest"], "cell revert reconstruction B4_up")
    else:
        chk(depend["bars"].get("ACCESSIBILITY_ONLY") is True and not (OUT / "depthclass.json").exists(), "band fail: accessibility only, no census")
    # ACT
    act = json.load(open(OUT / "act.json")); env = json.load(open(OUT / "act_envelope.json"))
    for n in ("DFA", "CTRL"):
        v = act["specimens"][n]
        chk(v["selfcheck_ok"] and len(v["act"]) == 16 and v["state_digest"] == disc["dfa" if n == "DFA" else "bp"]["final"]["state_digest"], f"ACT {n} self-check / digest")
    u, v = act["specimens"]["DFA"]["act"], act["specimens"]["CTRL"]["act"]
    if all(x is not None for x in u + v):
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))
        e1 = dist > env["observation"]["ACT_null_envelope"] and dist > env["observation"]["ACT_dist_A_B"]
        chk(close(dist, act["bars"]["ACT_dist_DFA_CTRL"]) and act["bars"]["ACT-1"] == e1, "ACT distance / ACT-1 recomputed")
        chk(act["bars"]["ACT_null_envelope_booked"] == env["observation"]["ACT_null_envelope"] and act["bars"]["ACT_dist_A_B_booked"] == env["observation"]["ACT_dist_A_B"], "ACT booked numbers")
        bars.update({"ACT_dist": dist, "ACT-1": e1})
    else:
        chk(act["bars"]["ACT-1"] == "NOT-RESOLVABLE", "ACT NOT-RESOLVABLE propagated")
    chk(act["probe_token_digest"] == json.load(open(OUT / "probe.json"))["token_digest"], "ACT probe digest")
    # alignment
    al = json.load(open(OUT / "align.json"))
    chk(al["feedback"]["digest"] == fb["digest"] and al["probe_token_digest"] == json.load(open(OUT / "probe.json"))["token_digest"], "align feedback / probe digests")
    for n, b in (("DFA", disc["dfa"]), ("CTRL", disc["bp"])):
        snaps = al["arms"][n]["snapshots"]
        chk(sorted(int(s) for s in snaps) == sorted(int(s) for s in b["snapshots"]), f"align {n} snapshot set")
        for s, r in snaps.items():
            chk(r["state_digest"] == b["snapshots"][s]["state_digest"], f"align {n} {s} digest")
            chk(all(c is None or -1 - 1e-12 <= c <= 1 + 1e-12 for c in r["cos"]) and len(r["cos"]) == 8, f"align {n} {s} cosines")
    if band_pass and "DEP-DEPTH" in bars and "ACT-1" in bars:
        dd, dcl, a1 = bars["DEP-DEPTH"], bars["DEP-CLASS"], bars["ACT-1"]
        bars["ladder"] = "A" if (dd and not dcl) else "B" if (dd and dcl) else "C" if (not dd and not dcl and not a1) else "E"
    elif not band_pass:
        bars["ladder"] = "D"
    rec = {"kind": "verify", "prereg": "WRITER-DFA-1", "verdict": "VERIFIED" if not D else "DISCREPANCIES", "n_discrepancies": len(D),
           "discrepancies": D[:80], "birth_commit": commit, "source_sha256_at_commit": src, "bars": bars,
           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "status_porcelain": subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout,
           "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "verify_receipt.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("verdict", "n_discrepancies", "discrepancies", "bars")}, indent=1))


if __name__ == "__main__":
    main()
