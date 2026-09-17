"""PERTURBATION-RESPONSE-GRAM-DESK-0: zero-training Gram desk on the booked
RANDOM-DIRECTION-CONTROL-1 snapshots (C / R1 / R2 / R3) and the locked
FIRST-MOMENT-ERASURE-LADDER-2 eps = 1e-1 snapshots (M), writer A, anchor 7200,
the 12 registered horizons.

Inputs are read only: every snapshot is opened once, its file sha and canonical
state digest asserted against the locked receipts (logs/rdc1/control.json and
logs/fmel2/ladder.json, both sha-pinned below) before any number is derived.
No optimizer state is read, nothing is trained, nothing is written except the
receipt under logs/prgd0/.

Quantities (all descriptive; no bar, no label):
  deviations      dev_a(h) = W_a(h) - W_C(h) for a in {M, R1, R2, R3}, float64
                  flats under the WRITER-TRAJECTORY-CENSUS-0 sorted-key law
                  (59 tensors, d = 18,911,616, groups BLOCK0..7 / OUTSIDE).
  Gram(h)         the 4 x 4 normalized Gram (cosine) matrix of the deviations at
                  h, its eigenvalues (descending), top eigenvalue / trace,
                  participation rank (sum l)^2 / sum l^2 and effective rank
                  exp(H(l / sum l)); globally and per group; also the 3 x 3
                  random-only Gram (R1, R2, R3).
  increments      D_a(k) = dev_a(h_{k+1}) - dev_a(h_k) over the 11 consecutive
                  grid windows; their norms and the same Gram readouts.
  lag matrices    cos(D_a(i), D_b(j)) for every ordered arm pair and every pair
                  of windows (11 x 11), globally; the named cells (the
                  moment-axis growth window against the panel's later window)
                  are reported explicitly, globally and per group, together with
                  the adjacent alternatives so no single grid choice is read
                  alone; and cos(dev_a(h_i), dev_b(h_j)) over the 12 x 12
                  horizon pairs.
Zero-norm law: a cosine with a zero operand is UNDEFINED = null + reason, never
NaN; eigen readouts of a Gram with a zero-norm row are null + reason.

SMOKE=1 runs the same mechanism on the first three grid horizons only and writes
to logs/prgd0/smoke<TAG>_desk.json (path-isolated; the real receipt path is
refused when it exists). A mechanical wall cap (SIGALRM) books NOT-RUN and
exits 3.
"""
import datetime
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from atomtraj_pins import CLASSES, state_digest  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"
SMOKE_TAG = os.environ.get("SMOKE_TAG", "")

PREREG = "PERTURBATION-RESPONSE-GRAM-DESK-0"
RDC1 = Path("logs/rdc1/control.json")
RDC1_SHA = "f85cc4a5539ea725440e820d107a1520e4e5ff743e351a99751f2fdfbb9f0be5"
FMEL2 = Path("logs/fmel2/ladder.json")
FMEL2_SHA = "0faffc61bd65c96d9581aaddd8b47c2379e27ba0cd5400153e9b6bb61015c917"
LOCK = Path("docs/receipts.lock.json")
WRITER = "A"
ANCHOR = 7200
M_ARM = "e1e-1"
R_ARMS = ["R1", "R2", "R3"]
ARMS = ["M"] + R_ARMS
GRID_FULL = [1, 5, 20, 100, 300, 900, 1800, 3080, 4500, 6000, 7200, 8220]
GRID = GRID_FULL if not SMOKE else GRID_FULL[:3]
KEYS = sorted(sum(CLASSES.values(), []))
GROUPS = {f"BLOCK{layer}": sorted(k for k in KEYS if k.startswith(f"blocks.{layer}.")) for layer in range(8)}
GROUPS["OUTSIDE"] = sorted(k for k in KEYS if not k.startswith("blocks."))
D_EXPECTED = 18_911_616
N_TENSORS = 59
WALL_S = 3600 if not SMOKE else int(os.environ.get("SMOKE_MAX_WALL_S", "3600"))
OUT_DIR = Path("logs/prgd0")
RECEIPT = OUT_DIR / (f"smoke{SMOKE_TAG}_desk.json" if SMOKE else "desk.json")

# The named lag cells: (arm a, window of a, arm b, window of b); windows as (h_from, h_to) on the registered grid.
# PRIMARY: the moment axis's 10x-to-157x growth window (300 -> 900) against each random arm's later large step (900 -> 1800).
NAMED_CELLS = {
    "primary_M300-900_v_R900-1800": ("M", (300, 900), "R", (900, 1800)),
    "alt_M100-300_v_R300-900": ("M", (100, 300), "R", (300, 900)),
    "alt_M300-900_v_R1800-3080": ("M", (300, 900), "R", (1800, 3080)),
    "alt_M900-1800_v_R900-1800_lag0": ("M", (900, 1800), "R", (900, 1800)),
    "alt_M900-1800_v_R1800-3080": ("M", (900, 1800), "R", (1800, 3080)),
    "alt_M300-900_v_R300-900_lag0": ("M", (300, 900), "R", (300, 900)),
    "two_window_M300-1800_v_R900-3080": ("M", (300, 1800), "R", (900, 3080)),
}


class Abort(Exception):
    pass


class WallLimit(Exception):
    pass


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


# ---------------------------------------------------------------- flat law (the census's sorted-key law, shapes from the loaded state)
def segments(sd):
    """[(key, start, end, group)] over the 59 registered tensors in sorted-key order, shapes from the state dict itself."""
    missing = [k for k in KEYS if k not in sd]
    assert not missing, f"state lacks registered tensors: {missing[:5]}"
    g_of = {k: g for g, ks in GROUPS.items() for k in ks}
    segs, off = [], 0
    for k in KEYS:
        n = int(sd[k].numel())
        segs.append((k, off, off + n, g_of[k]))
        off += n
    assert len(segs) == N_TENSORS
    if not SMOKE:
        assert off == D_EXPECTED, off
    digest = hashlib.sha256(json.dumps([(k, tuple(sd[k].shape)) for k in KEYS]).encode()).hexdigest()
    return segs, off, digest


def flat(sd, segs, d):
    x = np.empty(d, dtype=np.float64)
    for k, a, b, _ in segs:
        x[a:b] = sd[k].detach().to(torch.float64).reshape(-1).numpy()
    return x


def group_index(segs):
    return {g: np.concatenate([np.arange(a, b) for _, a, b, gg in segs if gg == g]) for g in sorted(set(s[3] for s in segs))}


# ---------------------------------------------------------------- Gram readouts (pure numpy; UNDEFINED law)
def cosine(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if not (math.isfinite(na) and math.isfinite(nb)) or na == 0.0 or nb == 0.0:
        return None
    c = float(a @ b) / (na * nb)
    return c if math.isfinite(c) else None


def gram_stats(vecs, names):
    """Normalized Gram of the named vectors: cosine matrix, eigenvalues (descending), top / trace, participation rank
    (sum l)^2 / sum l^2, effective rank exp(H). Any zero / non-finite norm -> eigen readouts null with a reason."""
    n = len(names)
    norms = [float(np.linalg.norm(v)) for v in vecs]
    out = {"names": list(names), "norms": [(x if math.isfinite(x) else None) for x in norms], "cos": None, "eig": None,
           "top_over_trace": None, "participation_rank": None, "effective_rank": None, "undefined": None}
    bad = [names[i] for i, x in enumerate(norms) if not math.isfinite(x) or x == 0.0]
    if bad:
        out["undefined"] = f"zero or non-finite norm in {bad}"
        return out
    G = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            G[i, j] = G[j, i] = (1.0 if i == j else float(vecs[i] @ vecs[j]) / (norms[i] * norms[j]))
    if not np.all(np.isfinite(G)):
        out["undefined"] = "non-finite cosine"
        return out
    lam = np.linalg.eigvalsh(G)[::-1]
    lam = np.clip(lam, 0.0, None)
    s1, s2 = float(lam.sum()), float((lam * lam).sum())
    p = lam / s1
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    out.update({"cos": G.tolist(), "eig": lam.tolist(), "top_over_trace": float(lam[0] / s1),
                "participation_rank": (s1 * s1 / s2 if s2 > 0 else None), "effective_rank": float(math.exp(ent))})
    return out


def gram_block(vecs_by_arm, gidx):
    """Global + per-group + random-only Gram readouts for {arm: vec}."""
    arms = [a for a in ARMS if a in vecs_by_arm]
    rs = [a for a in R_ARMS if a in vecs_by_arm]
    rec = {"global": gram_stats([vecs_by_arm[a] for a in arms], arms),
           "random_only": gram_stats([vecs_by_arm[a] for a in rs], rs) if len(rs) >= 2 else None, "groups": {}}
    for g, idx in gidx.items():
        rec["groups"][g] = {"all": gram_stats([vecs_by_arm[a][idx] for a in arms], arms),
                            "random_only": gram_stats([vecs_by_arm[a][idx] for a in rs], rs) if len(rs) >= 2 else None}
    return rec


def windows(grid):
    return [(grid[k], grid[k + 1]) for k in range(len(grid) - 1)]


def window_vector(dev_of, arm, h_from, h_to):
    """dev_arm(h_to) - dev_arm(h_from); None when either horizon is absent from the loaded grid."""
    if h_from not in dev_of[arm] or h_to not in dev_of[arm]:
        return None
    return dev_of[arm][h_to] - dev_of[arm][h_from]


def lag_matrix(inc_a, inc_b):
    """cos(D_a(i), D_b(j)) over all window pairs (rows i = a's windows, columns j = b's windows)."""
    return [[cosine(x, y) for y in inc_b] for x in inc_a]


def named_cells(dev_of, gidx):
    """The registered lag cells: for each R arm, the cosine of the two window vectors globally and per group."""
    out = {}
    for name, (aa, wa, bb, wb) in NAMED_CELLS.items():
        cell = {"a": aa, "window_a": list(wa), "b": bb, "window_b": list(wb), "per_arm": {}}
        va = window_vector(dev_of, aa, *wa)
        for r in R_ARMS:
            vb = window_vector(dev_of, r, *wb)
            if va is None or vb is None:
                cell["per_arm"][r] = {"cos": None, "undefined": "window horizon not in the loaded grid", "groups": None}
                continue
            cell["per_arm"][r] = {"cos": cosine(va, vb), "norm_a": float(np.linalg.norm(va)), "norm_b": float(np.linalg.norm(vb)),
                                  "groups": {g: cosine(va[idx], vb[idx]) for g, idx in gidx.items()}}
        out[name] = cell
    return out


# ---------------------------------------------------------------- receipts, pins, snapshots
def load_receipts():
    lock = json.loads(LOCK.read_text())
    rdc1_sha, fmel2_sha = sha256_file(RDC1), sha256_file(FMEL2)
    assert rdc1_sha == RDC1_SHA == lock["receipts"][str(RDC1)]["sha256"], "RDC1 receipt v the source literal / the lock"
    assert fmel2_sha == FMEL2_SHA == lock["receipts"][str(FMEL2)]["sha256"], "FMEL2 receipt v the source literal / the lock"
    rdc1, fmel2 = json.loads(RDC1.read_text()), json.loads(FMEL2.read_text())
    assert rdc1["status"] == "DONE" and rdc1["smoke"] is False and rdc1["writer"] == WRITER
    assert fmel2["status"] == "DONE" and fmel2["smoke"] is False and fmel2["writer"] == WRITER
    assert rdc1["pins"]["fmel2_sha256"] == FMEL2_SHA, "RDC1 was bound to another FMEL2 receipt"
    assert rdc1["anchor"]["state_digest"] == fmel2["anchor"]["state_digest"]
    assert rdc1["comparison_arm"] == M_ARM
    return rdc1, fmel2, {"rdc1_sha256": rdc1_sha, "fmel2_sha256": fmel2_sha, "rdc1_commit": rdc1["commit"], "fmel2_commit": fmel2["commit"],
                         "rdc1_label": rdc1["label"], "fmel2_label": fmel2["label"], "anchor": dict(rdc1["anchor"])}


def snapshot_refs(rdc1, fmel2, grid):
    """{arm: {h: {path, sha256, state_digest}}} for C / R1 / R2 / R3 from RDC1 and M from FMEL2's eps = 1e-1 arm; every
    RDC1 path under checkpoints/rdc1/, every M path under checkpoints/fmel2/A/e1e-1/. The RDC1 receipt's own reference
    digests for C (the locked FMEL2 C) must equal FMEL2's C digests at every horizon (the BAR 0 identity, re-asserted)."""
    refs = {}
    for arm in ["C"] + R_ARMS:
        refs[arm] = {}
        for h in grid:
            e = rdc1["arms"][arm]["snapshots"][str(h)]
            assert e["path"].startswith("checkpoints/rdc1/A/") and e["path"].endswith(f"h{h:04d}.pt"), e["path"]
            refs[arm][h] = {"path": e["path"], "sha256": e["sha256"], "state_digest": e["state_digest"]}
    refs["M"] = {}
    for h in grid:
        e = fmel2["arms"][M_ARM]["snapshots"][str(h)]
        assert e["path"].startswith("checkpoints/fmel2/A/e1e-1/") and e["path"].endswith(f"h{h:04d}.pt"), e["path"]
        refs["M"][h] = {"path": e["path"], "sha256": e["sha256"], "state_digest": e["state_digest"]}
        assert rdc1["reference_digests"]["C"][str(h)] == fmel2["arms"]["C"]["snapshots"][str(h)]["state_digest"] == refs["C"][h]["state_digest"], h
    return refs


def verify_files(refs):
    out, bad = {}, []
    for arm, hs in refs.items():
        out[arm] = {}
        for h, e in hs.items():
            p = Path(e["path"])
            if not p.exists():
                out[arm][str(h)] = {"path": e["path"], "status": "MISSING"}; bad.append(f"{arm}@{h}:MISSING"); continue
            s = sha256_file(p)
            st = "OK" if s == e["sha256"] else "DRIFTED"
            out[arm][str(h)] = {"path": e["path"], "file_sha256": s, "status": st}
            if st != "OK":
                bad.append(f"{arm}@{h}:{st}")
    return out, bad


def load_flat(ref, arm, h, segs, d):
    ck = torch.load(ref["path"], map_location="cpu")
    assert int(ck["step"]) == ANCHOR + h, (ref["path"], ck.get("step"))
    sd = ck["model"]
    dg = state_digest(sd)
    if dg != ref["state_digest"]:
        raise Abort(f"{arm} h = {h}: state digest {dg[:16]} v locked {ref['state_digest'][:16]}")
    law = None
    if segs is None:
        segs, d, law = segments(sd)
    return flat(sd, segs, d), segs, d, law


# ---------------------------------------------------------------- wall cap
def install_wall_limit(seconds):
    def handler(signum, frame):
        raise WallLimit(f"wall cap {seconds} s")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    if RECEIPT.exists():
        raise SystemExit(f"REFUSING: {RECEIPT} exists")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    install_wall_limit(WALL_S)
    rec = {"prereg": PREREG, "kind": "desk", "smoke": SMOKE, "smoke_tag": SMOKE_TAG, "commit": git("rev-parse", "HEAD"),
           "tree_dirty": bool(git("status", "--porcelain")), "self_sha256": sha256_file(__file__), "torch_version": torch.__version__,
           "numpy_version": np.__version__, "writer": WRITER, "anchor_step": ANCHOR, "grid": GRID, "arms": ARMS, "comparison_arm": M_ARM,
           "wall_limit_s": WALL_S, "started_utc": now(), "status": "RUNNING", "training": "none", "optimizer_state_read": False}
    phase = "receipts"
    try:
        rdc1, fmel2, pins = load_receipts()
        rec["pins"] = pins
        refs = snapshot_refs(rdc1, fmel2, GRID)
        rec["snapshots"] = {a: {str(h): dict(e) for h, e in hs.items()} for a, hs in refs.items()}
        rec["locked_rdc1_metrics"] = {str(h): {"abs": rdc1["metrics"][str(h)]["abs"], "abs_M": rdc1["metrics"][str(h)]["abs_M"],
                                              "cos_M": rdc1["metrics"][str(h)]["cos_M"], "pair_cos": rdc1["metrics"][str(h)]["pair_cos"]} for h in GRID}
        phase = "file-sha"
        files, bad = verify_files(refs)
        rec["snapshot_files"] = files
        if bad:
            rec["status"] = "NOT-RUN"; rec["reason"] = "snapshot missing or drifted: " + ", ".join(bad)
            return finish(rec, t0)
        phase = "load"
        segs = d = gidx = None
        dev_of = {a: {} for a in ARMS}
        per_h = {}
        for h in GRID:
            WC, segs, d, law = load_flat(refs["C"][h], "C", h, segs, d)
            if h == GRID[0]:
                gidx = group_index(segs)
                rec["flat_law"] = {"d": d, "n_tensors": len(segs), "shape_digest": law, "groups": {g: len(ks) for g, ks in GROUPS.items()}}
            vec = {}
            for a in ARMS:
                W, _, _, _ = load_flat(refs[a][h], a, h, segs, d)
                vec[a] = W - WC
                dev_of[a][h] = vec[a]
            del WC
            per_h[str(h)] = gram_block(vec, gidx)
            # consistency with the locked RDC1 readouts (the same deviations, re-derived here): norms and cos to M
            chk = {a: {"abs_here": float(np.linalg.norm(vec[a])), "abs_locked": (rdc1["metrics"][str(h)]["abs_M"] if a == "M" else rdc1["metrics"][str(h)]["abs"][a])} for a in ARMS}
            for a in R_ARMS:
                chk[a]["cos_M_here"] = cosine(vec[a], vec["M"]); chk[a]["cos_M_locked"] = rdc1["metrics"][str(h)]["cos_M"][a]
            per_h[str(h)]["locked_consistency"] = chk
            print(f"[prgd0] h={h} top/trace {per_h[str(h)]['global']['top_over_trace']:.4f} PR {per_h[str(h)]['global']['participation_rank']:.3f} "
                  f"eig {np.round(per_h[str(h)]['global']['eig'], 4).tolist()} ({time.time() - t0:.0f} s)", flush=True)
        rec["gram_by_horizon"] = per_h
        phase = "increments"
        wins = windows(GRID)
        inc = {a: [dev_of[a][h2] - dev_of[a][h1] for h1, h2 in wins] for a in ARMS}
        rec["windows"] = [list(w) for w in wins]
        rec["gram_by_window"] = {}
        for k, w in enumerate(wins):
            rec["gram_by_window"][f"{w[0]}-{w[1]}"] = gram_block({a: inc[a][k] for a in ARMS}, gidx)
            g = rec["gram_by_window"][f"{w[0]}-{w[1]}"]["global"]
            print(f"[prgd0] window {w[0]}->{w[1]} norms {np.round(g['norms'], 5).tolist()} top/trace {g['top_over_trace']} PR {g['participation_rank']}", flush=True)
        phase = "lag"
        rec["lag_cos_windows"] = {f"{a}|{b}": lag_matrix(inc[a], inc[b]) for a in ARMS for b in ARMS}
        rec["lag_cos_horizons"] = {f"{a}|{b}": [[cosine(dev_of[a][hi], dev_of[b][hj]) for hj in GRID] for hi in GRID] for a in ARMS for b in ARMS}
        rec["named_cells"] = named_cells(dev_of, gidx)
        for name, c in rec["named_cells"].items():
            print(f"[prgd0] {name}: " + ", ".join(f"{r} {c['per_arm'][r]['cos']}" for r in R_ARMS), flush=True)
        rec["status"] = "DONE"
    except WallLimit as e:
        rec["status"] = "NOT-RUN"; rec["reason"] = f"{e} in phase {phase}"
        finish(rec, t0); sys.exit(3)
    except Abort as e:
        rec["status"] = "NOT-RUN"; rec["reason"] = f"{e} (phase {phase})"
    return finish(rec, t0)


def finish(rec, t0):
    rec["wall_s"] = time.time() - t0
    rec["ended_utc"] = now()
    rec["self_sha256_at_end"] = sha256_file(__file__)
    RECEIPT.write_text(json.dumps(rec, indent=1))
    print(f"[prgd0] {rec['status']} {rec.get('reason', '')} receipt {RECEIPT} wall {rec['wall_s']:.1f} s", flush=True)
    return rec


if __name__ == "__main__":
    r = main()
    sys.exit(0 if r["status"] == "DONE" else 2)
