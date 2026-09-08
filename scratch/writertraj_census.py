"""WRITER-TRAJECTORY-CENSUS-0, STAGE 0 (pre-reg RESULTS L67576): the
zero-training trajectory-derivative census. Loads the forward (phase19m)
and backward (backsched19m) OneCycle milestones of the shared seed-2
initialization and the four same-writer rerun pairs of
ATOM-DIET-TRAJECTORY-1 (first run v repair), flattens every checkpoint
under the frozen tensor law (GLOBAL, BLOCK 0..7, OUTSIDE, eight CLASSES,
sorted-key float64), and computes for each pair at each matched step:
cumulative cosine C, velocity cosine V (native grid spacing h),
velocity norms and ratio, relative divergence R, acceleration cosine
and norms. W_0 for the writer pair is the canonical seed-2 model
construction (torch.manual_seed(2) then build_model on CPU); its
canonical state digest is recorded, and the step-1 milestone distances
are recorded as consistency checks only. Writes
logs/writertraj0/census.json (refuses to overwrite) with the S0 bar
booleans computed from the sealed laws. No gate runs here.

Env: SMOKE=1 runs two milestones of family A only and appends a
kind=census row to logs/writertraj0/smoke.jsonl.

Usage: .venv/bin/python scratch/writertraj_census.py
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

import torch  # noqa: E402

import train_mathnative as TM  # noqa: E402
from atomtraj_pins import CLASSES, state_digest  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
OUT = Path("logs/writertraj0")
T_TOTAL = 15_420
MAIN = Path(".")
REPAIR = Path("/Users/artin/code/llmopt-repair")
WRITER_GRID = list(range(900, 15_301, 900))
NULL_GRID = [463] + list(range(1_028, T_TOTAL + 1, 1_028))
NULL_CELLS = ["stock_s6", "atoms_s6", "atoms_s7", "stock_s7"]
KEYS = sorted(sum(CLASSES.values(), []))
BLOCK = {l: sorted(k for k in KEYS if k.startswith(f"blocks.{l}.")) for l in range(8)}
OUTSIDE = sorted(k for k in KEYS if not k.startswith("blocks."))
SETS = {"GLOBAL": KEYS, "OUTSIDE": OUTSIDE}
SETS.update({f"BLOCK{l}": BLOCK[l] for l in range(8)})
SETS.update({f"CLASS_{c}": sorted(ks) for c, ks in CLASSES.items()})
assert sorted(sum(BLOCK.values(), []) + OUTSIDE) == KEYS


def load_model_sd(p):
    d = torch.load(p, map_location="cpu")
    return d["model"] if isinstance(d, dict) and "model" in d else d


def flat(sd, keys):
    return torch.cat([sd[k].detach().double().reshape(-1) for k in keys])


def cos(a, b):
    na, nb = float(a.norm()), float(b.norm())
    return float((a @ b) / (na * nb)) if na > 0 and nb > 0 else None


def w0_seed(seed):
    tok = TM.MathTokenizer()
    torch.manual_seed(seed)
    m = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def series(paths_by_step, w0):
    """paths_by_step: {step: path}. Returns per-set dict of {step: Delta flat} lazily via generator of (step, sd)."""
    steps = sorted(paths_by_step)
    return steps, {s: paths_by_step[s] for s in steps}


def pair_census(name, stepsX, pathsX, stepsY, pathsY, w0X, w0Y, sets):
    """Compute all pair quantities at common steps. Memory: two flats per set per step."""
    common = [s for s in stepsX if s in pathsY]
    out = {"pair": name, "steps": common, "per_set": {}}
    prev = {}   # set -> (step, DeltaX, DeltaY)
    prevv = {}  # set -> (step, vX, vY)
    w0fX = {sname: flat(w0X, keys) for sname, keys in sets.items()}
    w0fY = {sname: flat(w0Y, keys) for sname, keys in sets.items()}
    for s in common:
        sdX, sdY = load_model_sd(pathsX[s]), load_model_sd(pathsY[s])
        for sname, keys in sets.items():
            dX = flat(sdX, keys) - w0fX[sname]
            dY = flat(sdY, keys) - w0fY[sname]
            rec = out["per_set"].setdefault(sname, {})
            nX, nY = float(dX.norm()), float(dY.norm())
            r = {"C": cos(dX, dY), "normX": nX, "normY": nY,
                 "R": float((dX - dY).norm() / ((nX + nY) / 2)) if (nX + nY) > 0 else None}
            if sname in prev:
                s0, pX, pY = prev[sname]
                h = s - s0
                vX, vY = (dX - pX) / h, (dY - pY) / h
                r.update({"h": h, "V": cos(vX, vY), "vnormX": float(vX.norm()), "vnormY": float(vY.norm())})
                r["vratio"] = r["vnormX"] / r["vnormY"] if r["vnormY"] > 0 else None
                if sname in prevv:
                    s1, qX, qY = prevv[sname]
                    aX, aY = (vX - qX) / h, (vY - qY) / h      # sealed: a(t; h) = [v(t+h) - v(t)] / h
                    r.update({"A": cos(aX, aY), "anormX": float(aX.norm()), "anormY": float(aY.norm())})
                prevv[sname] = (s, vX, vY)
            prev[sname] = (s, dX, dY)
            rec[str(s)] = r
        print(f"[census] {name} step {s}: C {out['per_set']['GLOBAL'][str(s)]['C']:.4f} R {out['per_set']['GLOBAL'][str(s)]['R']:.4f}", flush=True)
    return out


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


MATCH_SET = [s for s in NULL_GRID if 0.1 <= s / T_TOTAL <= 0.9]   # the 12 enumerating times (1,028 grid)


def nearest(step, grid):
    """Nearest grid step; ties broken to the smaller step."""
    return min(grid, key=lambda g: (abs(g - step), g))


def mid_median(pc, key):
    """Median of the GLOBAL quantity over the 12 matched times of the 1,028
    grid mapped to this pair's own grid (nearest step, ties to the smaller)."""
    g = pc["per_set"]["GLOBAL"]
    own = [s for s in pc["steps"] if str(s) in g and key in g[str(s)]]
    vals = [g[str(nearest(t, own))].get(key) for t in MATCH_SET]
    return median(vals)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / ("smoke.jsonl" if SMOKE else "census.json")
    if not SMOKE and target.exists():
        raise SystemExit(f"REFUSING: {target} exists")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rec = {"prereg": "WRITER-TRAJECTORY-CENSUS-0", "stage": 0, "smoke": SMOKE, "commit": commit, "tree_dirty": dirty,
           "started_utc": started, "tensor_law": {k: len(v) for k, v in SETS.items()}, "pairs": {}, "artifacts": {}}
    # writer pair
    assert not os.environ.get("VOCAB_EXTRA"), "W_0 law requires VOCAB_EXTRA unset"
    w0_2 = w0_seed(2)
    rec["w0_law"] = {"seed": 2, "vocab_len": len(TM.MathTokenizer().vocab), "VOCAB_EXTRA": os.environ.get("VOCAB_EXTRA", ""),
                     "arch": {"d": 384, "layers": 8, "heads": 6, "ffn": 1536}, "n_params": sum(v.numel() for v in w0_2.values())}
    rec["w0_seed2_state_digest"] = state_digest(w0_2)
    A = {s: MAIN / f"checkpoints/phase19m/m{s:06d}.pt" for s in WRITER_GRID}
    B = {s: MAIN / f"checkpoints/backsched19m/m{s:06d}.pt" for s in WRITER_GRID}
    A[T_TOTAL] = MAIN / "checkpoints/gallery19m_phase_s2.pt"
    B[T_TOTAL] = MAIN / "checkpoints/gallery19m_backsched_s2.pt"
    if SMOKE:
        A = {s: A[s] for s in WRITER_GRID[:2]}
        B = {s: B[s] for s in WRITER_GRID[:2]}
    for fam, paths in (("A", A), ("B", B)):
        for s, p in paths.items():
            if not p.exists():
                raise SystemExit(f"NOT-RUN: missing {p}")
    # consistency checks on the step-1 milestones (never a proof of shared W_0)
    for fam, p in (("A", MAIN / "checkpoints/phase19m/m000001.pt"), ("B", MAIN / "checkpoints/backsched19m/m000001.pt")):
        if p.exists():
            d1 = flat(load_model_sd(p), KEYS) - flat(w0_2, KEYS)
            rec["artifacts"][f"{fam}_step1_distance_to_w0"] = float(d1.norm())
    rec["artifacts"]["A_paths"] = {str(s): str(p) for s, p in A.items()}
    rec["artifacts"]["B_paths"] = {str(s): str(p) for s, p in B.items()}
    rec["artifacts"]["sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in list(A.values()) + list(B.values())}
    rec["pairs"]["A_B"] = pair_census("A_B", sorted(A), A, sorted(B), B, w0_2, w0_2, SETS)
    # null pairs
    if not SMOKE:
        for cell in NULL_CELLS:
            seed = int(cell[-1])
            main_dir = MAIN / "checkpoints/atomtraj1" / cell
            rep_dir = REPAIR / "checkpoints/atomtraj1" / cell
            X = {s: main_dir / f"step_{s:05d}.pt" for s in NULL_GRID}
            Y = {s: rep_dir / f"step_{s:05d}.pt" for s in NULL_GRID}
            for p in list(X.values()) + list(Y.values()):
                if not p.exists():
                    raise SystemExit(f"NOT-RUN: missing {p}")
            w0x = load_model_sd(main_dir / "step_00000.pt")
            w0y = load_model_sd(rep_dir / "step_00000.pt")
            dx, dy = state_digest(w0x), state_digest(w0y)
            assert dx == dy, f"{cell}: step_0 digests differ across runs"
            regen = state_digest(w0_seed(seed))
            rec["artifacts"][f"null_{cell}_w0_digest"] = dx
            rec["artifacts"][f"null_{cell}_w0_matches_seed_regeneration"] = (regen == dx)
            rec["pairs"][f"NULL_{cell}"] = pair_census(f"NULL_{cell}", sorted(X), X, sorted(Y), Y, w0x, w0y, SETS)
        # bars
        AB = rec["pairs"]["A_B"]
        fA = str(max(AB["steps"]))
        nulls = [rec["pairs"][f"NULL_{c}"] for c in NULL_CELLS]
        nf = [str(max(n["steps"])) for n in nulls]
        C_AB = AB["per_set"]["GLOBAL"][fA]["C"]
        R_AB = AB["per_set"]["GLOBAL"][fA]["R"]
        C_null = [n["per_set"]["GLOBAL"][f]["C"] for n, f in zip(nulls, nf)]
        R_null = [n["per_set"]["GLOBAL"][f]["R"] for n, f in zip(nulls, nf)]
        V_AB = mid_median(AB, "V")
        V_null = [mid_median(n, "V") for n in nulls]
        blocks = {}
        for l in range(8):
            cb = AB["per_set"][f"BLOCK{l}"][fA]["C"]
            nb = min(n["per_set"][f"BLOCK{l}"][f]["C"] for n, f in zip(nulls, nf))
            blocks[str(l)] = {"C_AB": cb, "null_min": nb, "law_holds": cb < nb}
        rec["bars"] = {
            "final_step_writer": int(fA), "final_steps_null": [int(f) for f in nf],
            "C_AB_final": C_AB, "C_null_final": C_null, "S0-1": C_AB < min(C_null),
            "V_AB_mid_median": V_AB, "V_null_mid_median": V_null, "S0-2": (V_AB is not None and all(v is not None for v in V_null) and V_AB < min(V_null)),
            "R_AB_final": R_AB, "R_null_final": R_null, "S0-3": R_AB > max(R_null),
            "S0-4_blocks": blocks, "S0-4_count": sum(b["law_holds"] for b in blocks.values()),
        }
        rec["bars"]["TRAJECTORY"] = bool(rec["bars"]["S0-1"] and rec["bars"]["S0-3"])
    rec["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    if SMOKE:
        with target.open("a") as f:
            f.write(json.dumps({"kind": "census", **rec}) + "\n")
    else:
        target.write_text(json.dumps(rec, indent=1))
        print("[census] written; bars:", json.dumps({k: rec["bars"][k] for k in ("S0-1", "S0-2", "S0-3", "S0-4_count", "TRAJECTORY")}), flush=True)


if __name__ == "__main__":
    main()
