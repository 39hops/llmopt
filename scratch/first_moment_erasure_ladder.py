"""FIRST-MOMENT-ERASURE-LADDER-1 instrument (PRE-REG RESULTS L75312, AMENDMENT
-PRE-INSTRUMENT L75568): a perturbation-magnitude ladder on the first
moment of writer A at the validated A@7200 arena, run as FIVE fresh
continuous CPU-deterministic legs 7201..15420 (the full remaining
schedule): the native control C and exp_avg <- (1 - eps) exp_avg for eps
in {1e-3, 1e-2, 1e-1, 1}. Writer A only; no writer-B code path.

Thin sibling of scratch/first_moment_erasure.py: every continuation /
intervention mechanic is IMPORTED from scratch/optimizer_memory_ablation.py
(bind, scheduler resume, stream law, run_leg, readout, snapshots) and the
FME1 pin law is REUSED (its source-pin set plus its own source, its
leg-path symbol sha, the locked Stage-0 / desk / FME1 receipt shas). A
mismatch refuses before any state is created.

Registered mode (MODE=ladder; the only one), in this order:
  1. pins, locked receipt shas, anchor sha / digest, leg digest, disk
     preflight (refuse below 16 GiB free), reference digests loaded from
     the LOCKED receipts (Stage-0 C, FME1 Z / E at h = 1 / 5 / 20 / 100 /
     900) -- the historical snapshots are reference DIGESTS only, never
     data; nothing is resumed from them.
  2. ONE-STEP PREFLIGHT: every arm bound fresh, intervention applied, one
     step on the first future slice; |R_eps(1) - 1| <= 1e-3,
     cos(dW_eps(1), dW_1(1)) >= 0.999, |n_1(1) - n_pred| <= 0.02 for every
     eps. Failure -> receipt with label REGIME-UNRESOLVED, STOP (no long
     leg is paid for).
  3. the five long legs (order C, e1, e1e-2, e1e-1, e1e-3), each from a
     fresh bind; in-line BAR 0: at every qualification horizon the fresh
     C / e1 / e1e-2 state digest must equal the locked Stage-0 C / FME1
     Z / E digest bit-exact (mismatch -> receipt status NOT-RUN, abort);
     every arm's h = 1 digest must equal its preflight digest; model-only
     snapshots at the 12 grid horizons, the optimizer blob at H only;
     HELD-32 CE per horizon; C's in-line descriptive substrate readout at
     step 15300 against the booked m015300.pt.
  4. readouts per grid horizon from the fresh snapshots (n_eps, R_eps,
     alpha, cosmin, pair cosines, rotation, group share, dCE), the
     descriptive substrate readout of C(15420) against the booked
     gallery19m_phase_s2.pt (rho, CE, gate), the descriptive gate on C(H)
     and e1(H), BAR 2 regime (two-sided band, REGIME-UNRESOLVED on any
     UNDEFINED terminal value), BAR 3 locus, BAR 4 absolute function bar,
     BAR 5 cooled tail.                              -> logs/fmel1/ladder.json
No follow-up is launched. Snapshots under checkpoints/fmel1/A/<arm>/.

Zero-norm law: a zero deviation gives n = 0, R = 0 (UNDEFINED when the
eps = 1 deviation is zero), an UNDEFINED logarithm and an UNDEFINED
cosine; alpha needs all four logarithms, cosmin all six cosines; every
UNDEFINED is JSON null with a reason. No NaN is ever written.

SMOKE=1: the FME1 smoke pipeline builds the synthetic non-target arena
(OMA smoke Stage 0 + FME1 smoke treat, tag SMOKE_TAG), then this
instrument runs its ladder on it with the smoke grid [1, 2, 3], receipts
under logs/fmel1/smoke<TAG>_* and snapshots under checkpoints/fmel1_smoke.
Never writes a real path.

Usage: MODE=ladder .venv/bin/python scratch/first_moment_erasure_ladder.py
       SMOKE=1 SMOKE_TAG=lad .venv/bin/python scratch/first_moment_erasure_ladder.py
"""
import datetime
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
os.environ.setdefault("ARM", "off")
os.environ.setdefault("BIRTH_SEED", "0")

SMOKE = os.environ.get("SMOKE", "0") == "1"
SMOKE_TAG = os.environ.get("SMOKE_TAG", "")
MODE = os.environ.get("MODE", "smoke" if SMOKE else "")
# SMOKE ONLY: the synthetic step-3 arena cannot resolve eps = 1e-3 above float32 rounding, so the preflight law refuses there by
# design (that refusal is the first smoke); a second smoke bypasses it to exercise the long legs. Ignored outside SMOKE.
PREFLIGHT_BYPASS = SMOKE and os.environ.get("SMOKE_PREFLIGHT_BYPASS", "0") == "1"

import first_moment_erasure as FME  # noqa: E402  (module import asserts the FME1 pins + leg-path sha; refuses on drift)
import numpy as np  # noqa: E402
import torch  # noqa: E402

OMA = FME.OMA
UG = FME.UG
TM = FME.TM
state_digest = FME.state_digest
gate_eval = FME.gate_eval
sha256_file = FME.sha256_file

# ---------------------------------------------------------------- pins (the FME1 set + the FME1 instrument itself + its locked receipt)
FME1_SOURCE_SHA = "173f46ff4715df5e9a74cd8c370423e44fadf2b076f4cacd3e45d97383a2b8cc"    # scratch/first_moment_erasure.py at the seal
FME1_SHA = "a3d8ae834091dbc8071048aa25a8358ae0563f2ba33aead626e5b7ba10209856"           # logs/fme1/treat.json (locked)
PINS = dict(FME.PINS, **{"scratch/first_moment_erasure.py": FME1_SOURCE_SHA})
_bad = FME.check_pins(PINS)
if _bad:
    raise SystemExit("PIN MISMATCH: " + json.dumps({k: {"expected": v[0][:16], "actual": (v[1] or "MISSING")[:16]} for k, v in _bad.items()}))

WRITER = FME.WRITER
N_PRED = FME.N_PRED
ANCHOR_SHA = FME.ANCHOR_SHA
ANCHOR = OMA.ANCHOR
LEG_FULL = OMA.TOTAL - ANCHOR if not SMOKE else OMA.LEG                      # 8220 real; 3 in smoke
H_END = LEG_FULL
GRID = [1, 5, 20, 100, 300, 900, 1800, 3080, 4500, 6000, 7200, 8220] if not SMOKE else [1, 2, 3]
QUAL = OMA.HORIZONS                                                          # [1, 5, 20, 100, 900] real; [1, 2, 3] smoke
LADDER = [1e-3, 1e-2, 1e-1, 1.0]
ARMS = {"C": 0.0, "e1e-3": 1e-3, "e1e-2": 1e-2, "e1e-1": 1e-1, "e1": 1.0}
ARM_ORDER = ["C", "e1", "e1e-2", "e1e-1", "e1e-3"]
QUALIFIED = {"C": "C", "e1": "Z", "e1e-2": "E"}                              # fresh arm -> reference arm in the locked receipts
ARENA = {"anchor": 7200, "leg": 8220, "grid": GRID, "qual": [1, 5, 20, 100, 900], "threads": 8, "ladder": LADDER}
# law (literal; AMENDMENT -PRE-INSTRUMENT L75568)
BAR1_TOL, R1_TOL, COS1_MIN = 0.02, 1e-3, 0.999
FORGOTTEN, PERSISTENT = 0.05, 0.25
ALPHA_BAND, COS_SHARED, ALPHA_FLAT, COS_DECOR = (0.80, 1.20), 0.90, 0.20, 0.50
CE_ABS = 0.005
TAIL_H = 6000 if not SMOKE else 2
MIN_FREE_BYTES = 16 * (1 << 30)
# descriptive substrate provenance targets (no threshold)
_FAM_A = Path(OMA.WRITERS[WRITER]["anchor"]).parent
END_MODEL = Path("checkpoints/gallery19m_phase_s2.pt") if not SMOKE else Path(OMA.WRITERS[WRITER]["target"])
END_DIGEST = "4633efe5d376f911" if not SMOKE else None
END_SHA = "e7207b3bd4df541a" if not SMOKE else None
MID_STEP = 15300 if not SMOKE else ANCHOR + LEG_FULL
MID_MODEL = _FAM_A / f"m{MID_STEP:06d}.pt"
MID_DIGEST = "d97b19e0ff3c84e5" if not SMOKE else None
MID_SHA = "6a715beb6394e3b3" if not SMOKE else None

STAGE0 = FME.STAGE0
DESK = FME.DESK
FME1 = Path(f"logs/fme1/smoke{FME.ARENA_TAG}_treat.json" if SMOKE else "logs/fme1/treat.json")
LOCK = FME.LOCK
OUT_DIR = Path("logs/fmel1")
CK_DIR = Path("checkpoints/fmel1_smoke" if SMOKE else "checkpoints/fmel1")
RECEIPT = OUT_DIR / (f"smoke{SMOKE_TAG}_ladder.json" if SMOKE else "ladder.json")
STREAM = OUT_DIR / (f"smoke{SMOKE_TAG}_ladder.jsonl" if SMOKE else "ladder.jsonl")
if not SMOKE:
    assert (ANCHOR, LEG_FULL, GRID, QUAL, OMA.THREADS) == (ARENA["anchor"], ARENA["leg"], ARENA["grid"], ARENA["qual"], ARENA["threads"]), "arena literals"
    assert ANCHOR + LEG_FULL == OMA.TOTAL == 15_420
assert GRID[-1] == H_END and all(h in GRID for h in QUAL) and TAIL_H in GRID
OMA.CK_DIR = CK_DIR                            # this instrument's snapshots never land under checkpoints/oma1 or checkpoints/fme1


# ---------------------------------------------------------------- pure laws
def undefined(reason):
    return {"value": None, "reason": reason}


def apply_eps(opt, eps):
    """exp_avg <- (1 - eps) exp_avg on every tensor; eps = 1 and eps = EPS_TWIN go through OMA.apply_arm (Z / E) so the fresh e1 /
    e1e-2 arms are the FME1 float operations exactly. Returns the number of tensors touched (0 for the control)."""
    if eps == 0.0:
        return 0
    if eps == 1.0:
        return OMA.apply_arm(opt, "Z")
    if eps == OMA.EPS_TWIN:
        return OMA.apply_arm(opt, "E")
    n = 0
    with torch.no_grad():
        for p in opt.param_groups[0]["params"]:
            opt.state[p]["exp_avg"].mul_(1.0 - eps)
            n += 1
    return n


def cosine(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0 or not (math.isfinite(na) and math.isfinite(nb)):
        return None
    c = float(a @ b / (na * nb))
    return c if math.isfinite(c) else None


def finite_or_none(x):
    return x if (x is not None and math.isfinite(x)) else None


def horizon_metrics(W0, WC, W, eps_of=None):
    """Per-horizon readouts from flats: W0 anchor, WC control, W = {arm: flat} for the four eps arms. Zero-norm law applied;
    UNDEFINED values are None with a reason under `undefined`."""
    eps_of = eps_of or {a: e for a, e in ARMS.items() if e > 0}
    dC = WC - W0
    legn = float(np.linalg.norm(dC))
    dev = {a: W[a] - WC for a in eps_of}
    absn = {a: float(np.linalg.norm(dev[a])) for a in eps_of}
    nonfinite = [a for a in eps_of if not math.isfinite(absn[a])] + (["control"] if not math.isfinite(legn) else [])
    if nonfinite:
        # non-finite weights: every derived value UNDEFINED (the zero-norm law extended; never a NaN in the receipt)
        return {"leg_abs": finite_or_none(legn), "abs": {a: finite_or_none(absn[a]) for a in eps_of}, "n": {a: None for a in eps_of}, "R": {a: None for a in eps_of},
                "ln": {a: None for a in eps_of}, "group_share": {a: None for a in eps_of}, "leg_cos": {a: None for a in eps_of}, "alpha": None, "cosmin": None,
                "pair_cos": {}, "undefined": {"nonfinite": f"non-finite deviation norm in {nonfinite}"}}
    out = {"leg_abs": legn, "abs": absn, "n": {}, "R": {}, "ln": {}, "group_share": {}, "leg_cos": {}, "undefined": {}}
    ref = [a for a, e in eps_of.items() if e == 1.0][0]
    for a in eps_of:
        out["n"][a] = absn[a] / legn if legn > 0 else None
        if legn == 0:
            out["undefined"][f"n:{a}"] = "control leg displacement is zero"
        if absn[ref] > 0:
            out["R"][a] = absn[a] / (eps_of[a] * absn[ref])
        else:
            out["R"][a] = None; out["undefined"][f"R:{a}"] = "eps = 1 deviation norm is zero"
        if absn[a] > 0:
            out["ln"][a] = math.log(absn[a])
        else:
            out["ln"][a] = None; out["undefined"][f"ln:{a}"] = "deviation norm is zero"
        out["leg_cos"][a] = cosine(W[a] - W0, dC)
        out["group_share"][a] = None                                 # filled by the caller with the segment law
    # alpha: least squares of ln||dW|| against ln eps over the four eps, defined only when all four logs are defined
    if all(out["ln"][a] is not None for a in eps_of):
        x = np.array([math.log(eps_of[a]) for a in eps_of]); y = np.array([out["ln"][a] for a in eps_of])
        xm, ym = x.mean(), y.mean()
        out["alpha"] = float(((x - xm) @ (y - ym)) / ((x - xm) @ (x - xm)))
    else:
        out["alpha"] = None; out["undefined"]["alpha"] = "a deviation norm is zero (fewer than four defined logarithms)"
    arms = list(eps_of)
    pairs = {}
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            pairs[f"{arms[i]}|{arms[j]}"] = cosine(dev[arms[i]], dev[arms[j]])
    out["pair_cos"] = pairs
    if all(v is not None for v in pairs.values()):
        out["cosmin"] = float(min(pairs.values()))
    else:
        out["cosmin"] = None; out["undefined"]["cosmin"] = "a pair cosine is UNDEFINED (zero deviation)"
    return out


def group_shares(dev, segs):
    tot = float(dev @ dev)
    if tot == 0.0:
        return None
    share = {}
    for grp in sorted(set(s[3] for s in segs)):
        share[grp] = sum(float(dev[a:b] @ dev[a:b]) for k, a, b, g in segs if g == grp) / tot
    return share


def preflight_law(m1, n_pred=N_PRED):
    """BAR 1 + directional resolution on the one-step metrics dict (horizon_metrics at h = 1). Returns (ok, detail)."""
    detail = {"n_1": m1["n"].get("e1"), "R": m1["R"], "cos_to_e1": {}, "fails": []}
    n1 = m1["n"].get("e1")
    if n1 is None or abs(n1 - n_pred) > BAR1_TOL:
        detail["fails"].append(f"n_1(1) {n1} v n_pred {n_pred} (tol {BAR1_TOL})")
    for a, e in ARMS.items():
        if e == 0:
            continue
        r = m1["R"][a]
        if r is None or abs(r - 1.0) > R1_TOL:
            detail["fails"].append(f"R_{a}(1) {r} (tol {R1_TOL})")
        c = 1.0 if a == "e1" else m1["pair_cos"].get(f"{a}|e1", m1["pair_cos"].get(f"e1|{a}"))
        detail["cos_to_e1"][a] = c
        if c is None or c < COS1_MIN:
            detail["fails"].append(f"cos(dW_{a}(1), dW_1(1)) {c} < {COS1_MIN}")
    return not detail["fails"], detail


def regime(mH, preflight_ok):
    """BAR 2 at H (first match wins): FORGOTTEN / REGIME-UNRESOLVED / MAGNITUDE-SCALED PERSISTENT / NONLINEAR DIRECTION-SHARED /
    TRAJECTORY-SENSITIVE / INTERMEDIATE. FORGOTTEN uses only n_eps (always defined when the control moved)."""
    ns = [mH["n"][a] for a in mH["n"]]
    if all(v is not None for v in ns) and all(v <= FORGOTTEN for v in ns):
        return "FORGOTTEN"
    if any(v is None for v in ns):
        return "REGIME-UNRESOLVED"
    if not preflight_ok or mH["alpha"] is None or mH["cosmin"] is None or mH["n"].get("e1") is None:
        return "REGIME-UNRESOLVED"
    al, cm, n1 = mH["alpha"], mH["cosmin"], mH["n"]["e1"]
    in_band = ALPHA_BAND[0] <= al <= ALPHA_BAND[1]
    if in_band and cm >= COS_SHARED and n1 >= PERSISTENT:
        return "MAGNITUDE-SCALED PERSISTENT"
    if cm >= COS_SHARED and not in_band:
        return "NONLINEAR DIRECTION-SHARED"
    if al <= ALPHA_FLAT and cm <= COS_DECOR:
        return "TRAJECTORY-SENSITIVE"
    return "INTERMEDIATE"


def locus(metrics_by_h):
    """BAR 3: h_lin = last grid h in the two-sided band with cosmin >= COS_SHARED; h_dec = first grid h with cosmin <= COS_DECOR.
    UNDEFINED horizons are excluded. None = not reached."""
    h_lin = h_dec = None
    for h in sorted(metrics_by_h):
        m = metrics_by_h[h]
        if m["cosmin"] is None:
            continue                                   # h_dec needs cosmin only; h_lin needs alpha as well
        if m["alpha"] is not None and ALPHA_BAND[0] <= m["alpha"] <= ALPHA_BAND[1] and m["cosmin"] >= COS_SHARED:
            h_lin = h
        if h_dec is None and m["cosmin"] <= COS_DECOR:
            h_dec = h
    return {"h_lin": h_lin, "h_dec": h_dec}


def function_bar(dce_H):
    """BAR 4 per arm, absolute: NEUTRAL iff |dCE| <= CE_ABS, HARMED above positively, HELPED above negatively."""
    per = {}
    for a, v in dce_H.items():
        per[a] = "NEUTRAL" if abs(v) <= CE_ABS else ("HARMED" if v > 0 else "HELPED")
    return per, ("FUNCTION-NEUTRAL" if all(v == "NEUTRAL" for v in per.values()) else "FUNCTION-" + ",".join(f"{a}:{v}" for a, v in per.items() if v != "NEUTRAL"))


def label(reg, func):
    return f"{reg}+{func} [writer A only, one anchor, one seed lineage, CPU deterministic; gate descriptive]"


def reference_digests(stage0, fme1, horizons=None):
    """{fresh arm: {h: locked state digest}} for the three qualified arms, read from the LOCKED receipts (Stage-0 C; FME1 Z / E)."""
    horizons = QUAL if horizons is None else horizons
    refs = {"C": {h: stage0["cells"][WRITER]["C"]["snapshots"][str(h)]["state_digest"] for h in horizons}}
    for arm, ref in (("e1", "Z"), ("e1e-2", "E")):
        refs[arm] = {h: fme1["arms"][ref]["snapshots"][str(h)]["state_digest"] for h in horizons}
    return refs


def qualify(arm, h, digest, refs):
    """BAR 0 for one (arm, h): None when the arm is not a qualified arm or h is not a qualification horizon; else (ok, expected)."""
    if arm not in refs or h not in refs[arm]:
        return None
    return digest == refs[arm][h], refs[arm][h]


def disk_preflight(path=".", min_free=MIN_FREE_BYTES):
    free = shutil.disk_usage(path).free
    return free >= min_free, free


def load_model_sd(path, step=None):
    ck = torch.load(path, map_location="cpu")
    if step is not None:
        assert int(ck["step"]) == step, (path, ck.get("step"), step)
    return ck["model"] if isinstance(ck, dict) and "model" in ck else ck


def assert_fme1_provenance(fme1, lock, stage0, horizons=None):
    """The FME1 receipt this rung pins: sha equal to the source literal and the lock; bound to the same Stage-0 receipt and anchor;
    the sealed n_pred; the Z / E snapshot digests present at every qualification horizon."""
    s = sha256_file(FME1)
    assert s == FME1_SHA == FME.locked_sha(lock, str(FME1)), "FME1 receipt v the source literal / the receipt lock"
    assert fme1["pins"]["stage0_sha256"] == FME.STAGE0_SHA == sha256_file(STAGE0), "FME1 was bound to another Stage-0 receipt"
    assert fme1["anchor"]["file_sha256"] == ANCHOR_SHA == stage0["cells"][WRITER]["C"]["bind"]["file_sha256"]
    assert fme1["n_pred_literal"] == N_PRED and fme1["writer"] == WRITER and fme1["smoke"] is False
    for ref in ("Z", "E"):
        assert all(str(h) in fme1["arms"][ref]["snapshots"] for h in (QUAL if horizons is None else horizons)), ref
    return {"fme1_sha256": s, "fme1_commit": fme1["commit"], "fme1_label": fme1["label"]}


# ---------------------------------------------------------------- legs
def one_step(arm, eps, tok, enc, slices, segs, d):
    """Preflight: fresh bind, intervention, exactly one step on the first future slice. Returns (flat, digest, binfo, touched, group_next)."""
    model, opt, _ck, binfo = OMA.bind(WRITER, tok, "cpu")
    _sched, grp = OMA.resume_sched(OMA.WRITERS[WRITER]["kind"], opt, ANCHOR, binfo["serialized"])
    touched = apply_eps(opt, eps)
    snaps, _losses, _ = OMA.run_leg(model, opt, _sched, tok, enc, slices[:1], "cpu", [1])
    sd = snaps[1]
    return OMA.flat(sd, segs, d), state_digest(sd), binfo, touched, grp


class Abort(SystemExit):
    pass


def long_leg(arm, eps, tok, enc, slices, segs, d, held, refs, preflight_digest, stream, cell, mid):
    """One fresh continuous leg with in-line BAR 0 and the preflight-digest assertion; snapshots at GRID (opt at H); HELD-32 CE
    per grid horizon; for C the in-line descriptive substrate readout at MID_STEP. Raises Abort (after filling `cell`) on any
    digest mismatch."""
    model, opt, _ck, binfo = OMA.bind(WRITER, tok, "cpu")
    sched, grp = OMA.resume_sched(OMA.WRITERS[WRITER]["kind"], opt, ANCHOR, binfo["serialized"])
    touched = apply_eps(opt, eps)
    assert touched == (0 if eps == 0.0 else len(UG.KEYS)), (arm, touched)
    cell.update({"arm": arm, "eps": eps, "bind": binfo, "group_next": grp, "tensors_touched": touched, "qualification": {}, "sched_at_grid": {}})
    check_at = set(QUAL) | {1} | ({mid["h"]} if arm == "C" and mid else set())
    grid_set = set(GRID)

    def on_step(i, loss):
        stream({"writer": WRITER, "arm": arm, "eps": eps, "device": "cpu", "step": ANCHOR + i, "loss": loss})
        if i in grid_set:
            g = opt.param_groups[0]
            cell["sched_at_grid"][str(i)] = {"lr_next": float(g["lr"]), "beta1_next": float(g["betas"][0])}   # the group AFTER step h (row anchor + h + 1)
        if i in check_at:
            sd = OMA.sd_cpu(model)
            dg = state_digest(sd)
            if i == 1:
                cell["preflight_digest_match"] = dg == preflight_digest
                if not cell["preflight_digest_match"]:
                    cell["abort"] = f"h = 1 digest {dg[:16]} v preflight {preflight_digest[:16]}"
                    raise Abort(3)
            q = qualify(arm, i, dg, refs)
            if q is not None:
                ok, exp = q
                cell["qualification"][str(i)] = {"digest": dg, "expected": exp, "ok": ok}
                if not ok:
                    cell["abort"] = f"BAR 0 mismatch at h = {i}: {dg[:16]} v locked {exp[:16]}"
                    raise Abort(3)
            if arm == "C" and mid and i == mid["h"]:
                Wm = OMA.flat(sd, segs, d)
                dm = Wm - mid["W"]
                cell["substrate_mid"] = {"step": ANCHOR + i, "booked": str(MID_MODEL), "booked_digest": mid["digest"], "booked_sha256": mid["sha256"],
                                         "fresh_digest": dg, "rho": float(np.linalg.norm(dm) / max(np.linalg.norm(mid["W"] - mid["W0"]), 1e-300)),
                                         "abs": float(np.linalg.norm(dm)), "ce_fresh": OMA.held_ce(model, tok, held, "cpu")[0], "ce_booked": mid["ce"]}

    t0 = time.time()
    snaps, losses, opt_digest_end = OMA.run_leg(model, opt, sched, tok, enc, slices, "cpu", GRID, on_step=on_step)
    wall = time.time() - t0
    cell.update({"opt_state_digest_end": opt_digest_end, "wall_s": round(wall, 1), "it_per_s": round(len(slices) / max(wall, 1e-9), 2),
                 "losses_first5": losses[:5], "loss_last": losses[-1], "steps": len(slices)})
    cell["snapshots"] = {str(h): OMA.save_snap(WRITER, arm, h, sd, opt if h == H_END else None) for h, sd in snaps.items()}
    ce = {}
    for h in GRID:
        model.load_state_dict(snaps[h]); ce[str(h)] = OMA.held_ce(model, tok, held, "cpu")[0]
    cell["ce_held"] = ce
    return snaps


# ---------------------------------------------------------------- the registered mode
def mode_ladder(tok, enc, starts, info, segs, d, held, rec, stream):
    stage0 = json.loads(STAGE0.read_text()); desk = json.loads(DESK.read_text()); fme1 = json.loads(FME1.read_text())
    rec["pins"] = {"sources": dict(PINS), "leg_path_symbols": FME.LEG_PATH_SYMBOLS, "leg_path_sha_measured": FME.leg_path_sha(), "leg_path_sha": FME.LEG_PATH_SHA,
                   "stage0_receipt": str(STAGE0), "stage0_sha256": sha256_file(STAGE0), "desk_receipt": str(DESK), "desk_sha256": sha256_file(DESK),
                   "fme1_receipt": str(FME1), "fme1_sha256": sha256_file(FME1), "data_files": FME.data_file_shas()}
    if not SMOKE:
        lock = json.loads(LOCK.read_text())
        rec["pins"].update(FME.assert_provenance(stage0, desk, lock, sha256_file(OMA.WRITERS[WRITER]["anchor"]), torch.get_num_threads(), torch.__version__, np.__version__))
        rec["pins"].update(assert_fme1_provenance(fme1, lock, stage0))
    n_pred = float(desk["cells"][WRITER]["law"]["n_pred"])
    assert abs(n_pred - stage0["cells"][WRITER]["bar1_law"]["n_pred"]) <= 1e-12
    if not SMOKE:
        assert n_pred == N_PRED
    slices = OMA.leg_slices(starts, info["n_enc"], ANCHOR + 1, LEG_FULL)
    assert len(slices) == LEG_FULL
    leg900 = hashlib.sha256(json.dumps(slices[:OMA.LEG]).encode()).hexdigest()
    assert leg900 == stage0["leg_slices_digest"], "future stream v the Stage-0 leg digest (first 900 slices)"
    rec["leg_slices_digest_first_leg"] = leg900
    rec["leg_slices_digest_full"] = hashlib.sha256(json.dumps(slices).encode()).hexdigest()
    rec["leg"] = {"first_step": ANCHOR + 1, "last_step": ANCHOR + LEG_FULL, "n_steps": LEG_FULL, "first_slice": list(slices[0]), "last_slice": list(slices[-1]),
                  "epoch_position_first": OMA.epoch_position(ANCHOR + 1, info["n_enc"]), "epoch_position_last": OMA.epoch_position(ANCHOR + LEG_FULL, info["n_enc"])}
    ok_disk, free = disk_preflight()
    rec["disk_preflight"] = {"free_bytes": free, "min_free_bytes": MIN_FREE_BYTES, "ok": ok_disk}
    if not ok_disk:
        raise SystemExit(f"DISK PREFLIGHT REFUSED: {free / (1 << 30):.1f} GiB free < {MIN_FREE_BYTES / (1 << 30):.0f} GiB")
    refs = reference_digests(stage0, fme1)
    if SMOKE and os.environ.get("SMOKE_TAMPER_REF") == "1":
        refs["C"][QUAL[1]] = "0" * 64                            # abort smoke: a corrupted reference digest must book NOT-RUN
        rec["smoke_tampered_ref"] = f"C@{QUAL[1]}"
    rec["reference_digests"] = {a: {str(h): v for h, v in hs.items()} for a, hs in refs.items()}
    anchor_sd = load_model_sd(OMA.WRITERS[WRITER]["anchor"])
    rec["anchor"] = {"path": OMA.WRITERS[WRITER]["anchor"], "file_sha256": sha256_file(OMA.WRITERS[WRITER]["anchor"]), "state_digest": state_digest(anchor_sd)}
    if not SMOKE:
        assert rec["anchor"]["state_digest"] == stage0["cells"][WRITER]["C"]["bind"]["state_digest"] == fme1["anchor"]["state_digest"]
    W0 = OMA.flat(anchor_sd, segs, d)
    # booked end / mid models for the descriptive substrate readout (loaded now so a missing file refuses before any leg)
    end_sd = load_model_sd(END_MODEL); mid_sd = load_model_sd(MID_MODEL, step=MID_STEP)
    end_dg, mid_dg = state_digest(end_sd), state_digest(mid_sd)
    if END_DIGEST:
        assert end_dg.startswith(END_DIGEST) and mid_dg.startswith(MID_DIGEST), (end_dg[:16], mid_dg[:16])
        assert sha256_file(END_MODEL).startswith(END_SHA) and sha256_file(MID_MODEL).startswith(MID_SHA), "booked substrate files v the pre-reg shas"
    mC = UG.build(tok, "cpu")
    mC.load_state_dict(mid_sd); mid_ce = OMA.held_ce(mC, tok, held, "cpu")[0]
    mC.load_state_dict(end_sd); end_ce = OMA.held_ce(mC, tok, held, "cpu")[0]
    mid = {"h": MID_STEP - ANCHOR, "W": OMA.flat(mid_sd, segs, d), "W0": W0, "digest": mid_dg, "sha256": sha256_file(MID_MODEL), "ce": mid_ce}
    assert 1 <= mid["h"] <= LEG_FULL
    # ---- ONE-STEP PREFLIGHT (no long leg before it passes)
    t0 = time.time()
    pf = {"arms": {}}
    W1 = {}
    for arm in ARM_ORDER:
        f, dg, binfo, touched, grp = one_step(arm, ARMS[arm], tok, enc, slices, segs, d)
        pf["arms"][arm] = {"digest": dg, "tensors_touched": touched, "group_next": grp}
        if not SMOKE:
            assert touched == (0 if arm == "C" else 59), (arm, touched)
        W1[arm] = f
    m1 = horizon_metrics(W0, W1["C"], {a: W1[a] for a in ARMS if a != "C"})
    for a in m1["abs"]:
        m1["group_share"][a] = group_shares(W1[a] - W1["C"], segs)
    pf["metrics"] = m1
    pf["ok"], pf["detail"] = preflight_law(m1, n_pred)
    pf["wall_s"] = round(time.time() - t0, 1)
    rec["preflight"] = pf
    print(f"[fmel1] preflight {'PASS' if pf['ok'] else 'FAIL'} n_1(1) {m1['n'].get('e1')} R {m1['R']} cos {pf['detail']['cos_to_e1']}", flush=True)
    del W1
    rec["smoke_preflight_bypassed"] = bool(PREFLIGHT_BYPASS and not pf["ok"])
    assert not (rec["smoke_preflight_bypassed"] and not SMOKE)
    if not pf["ok"] and not PREFLIGHT_BYPASS:
        rec["status"] = "PREFLIGHT-FAILED"
        rec["regime"] = "REGIME-UNRESOLVED"
        rec["label"] = label("REGIME-UNRESOLVED", "FUNCTION-NOT-MEASURED")
        return rec
    # ---- the five long legs
    rec["arms"] = {}
    rec["status"] = "RUNNING"
    for arm in ARM_ORDER:
        cell = {}
        rec["arms"][arm] = cell
        try:
            snaps = long_leg(arm, ARMS[arm], tok, enc, slices, segs, d, held, refs, pf["arms"][arm]["digest"], stream, cell, mid)
        except Abort:
            rec["status"] = "NOT-RUN"
            rec["regime"] = "NOT-RUN"
            rec["label"] = f"NOT-RUN ({cell.get('abort')})"
            print(f"[fmel1] ABORT {arm}: {cell.get('abort')}", flush=True)
            return rec
        del snaps
        print(f"[fmel1] {arm} leg done {cell['wall_s']} s ({cell['it_per_s']} it/s) loss_last {cell['loss_last']:.4f}", flush=True)
    # ---- readouts per grid horizon from the fresh snapshots
    metrics, dce, ce = {}, {}, {}
    prev_dev = {}
    W_end_C = None
    for h in GRID:
        WC = OMA.flat(load_model_sd(rec["arms"]["C"]["snapshots"][str(h)]["path"]), segs, d)
        W = {a: OMA.flat(load_model_sd(rec["arms"][a]["snapshots"][str(h)]["path"]), segs, d) for a in ARMS if a != "C"}
        m = horizon_metrics(W0, WC, W)
        m["rotation"] = {}
        for a in W:
            dev = W[a] - WC
            m["group_share"][a] = group_shares(dev, segs)
            m["rotation"][a] = cosine(dev, prev_dev[a]) if a in prev_dev else None
            prev_dev[a] = dev
        m["sched"] = rec["arms"]["C"]["sched_at_grid"].get(str(h))
        ce[str(h)] = {a: rec["arms"][a]["ce_held"][str(h)] for a in ARMS}
        dce[str(h)] = {a: rec["arms"][a]["ce_held"][str(h)] - rec["arms"]["C"]["ce_held"][str(h)] for a in ARMS if a != "C"}
        metrics[h] = m
        if h == H_END:
            W_end_C = WC
    rec["metrics"] = {str(h): m for h, m in metrics.items()}
    rec["ce_held"] = ce
    rec["dce"] = dce
    # ---- descriptive substrate provenance at H
    d_end = W_end_C - OMA.flat(end_sd, segs, d)
    rec["substrate_end"] = {"step": ANCHOR + H_END, "booked": str(END_MODEL), "booked_digest": end_dg, "booked_sha256": sha256_file(END_MODEL),
                            "fresh_digest": rec["arms"]["C"]["snapshots"][str(H_END)]["state_digest"],
                            "rho": float(np.linalg.norm(d_end) / max(np.linalg.norm(OMA.flat(end_sd, segs, d) - W0), 1e-300)), "abs": float(np.linalg.norm(d_end)),
                            "ce_fresh": rec["arms"]["C"]["ce_held"][str(H_END)], "ce_booked": end_ce}
    rec["substrate_mid"] = rec["arms"]["C"].get("substrate_mid")
    dev_gate = "mps" if torch.backends.mps.is_available() else "cpu"
    rec["gate_device"] = dev_gate
    gate = {}
    for name, sd in (("C", load_model_sd(rec["arms"]["C"]["snapshots"][str(H_END)]["path"])), ("e1", load_model_sd(rec["arms"]["e1"]["snapshots"][str(H_END)]["path"])), ("booked_end", end_sd)):
        gm = UG.build(tok, dev_gate); gm.load_state_dict(sd); gm.eval()
        solves, valid = gate_eval(gm, tok, dev_gate, n=(2 if SMOKE else None))
        gate[name] = {"solves": solves, "total": int(sum(solves.values())), "valid_pct": round(valid, 2)}
    rec["gate"] = gate
    # ---- adjudication
    mH = metrics[H_END]
    reg = regime(mH, pf["ok"])
    per, func = function_bar(dce[str(H_END)])
    tail = {a: (mH["abs"][a] / metrics[TAIL_H]["abs"][a] if metrics[TAIL_H]["abs"][a] > 0 else None) for a in mH["abs"]}
    over = [(a, str(h)) for h in GRID for a in dce[str(h)] if abs(dce[str(h)][a]) > CE_ABS]
    for a in QUALIFIED:
        assert set(rec["arms"][a]["qualification"]) == {str(h) for h in QUAL}, (a, list(rec["arms"][a]["qualification"]))   # BAR 0 never passes vacuously
    rec["adjudication"] = {"bar0": {a: all(q["ok"] for q in rec["arms"][a]["qualification"].values()) for a in QUALIFIED},
                           "bar1_preflight": pf["ok"], "bar2_regime": reg, "alpha_H": mH["alpha"], "cosmin_H": mH["cosmin"], "n_H": mH["n"], "R_H": mH["R"],
                           "bar3_locus": locus(metrics), "bar4_function": {"per_arm": per, "label": func, "dce_H": dce[str(H_END)], "over_threshold": over},
                           "bar5_tail": tail, "undefined_H": mH["undefined"]}
    rec["regime"] = reg
    rec["status"] = "DONE"
    rec["label"] = label(reg, func)
    print(f"[fmel1] alpha(H) {mH['alpha']} cosmin(H) {mH['cosmin']} n {mH['n']} -> {rec['label']}; gate {{ {', '.join(f'{k} {v['total']}' for k, v in gate.items())} }}", flush=True)
    return rec


def main():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(OMA.THREADS)
    torch.manual_seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if RECEIPT.exists() or STREAM.exists():
        raise SystemExit(f"REFUSING: {RECEIPT} or {STREAM} exists")
    if CK_DIR.exists() and any(CK_DIR.rglob("*.pt")):
        raise SystemExit(f"REFUSING: {CK_DIR} holds snapshots")
    assert MODE in ("ladder", "smoke"), MODE
    tok = TM.MathTokenizer()
    assert len(tok.vocab) == 40 and not os.environ.get("VOCAB_EXTRA") and not os.environ.get("SEQ_CAP") and not os.environ.get("BIRTH_BS") and TM.BS == OMA.BS
    OMA.OA.assert_verbatim()
    segs, d, flat_digest = UG.flatten_law(UG.build(tok, "cpu"))
    assert d == 18_911_616
    enc, starts, info = OMA.future_stream(tok)
    ugc0 = json.loads(OMA.UGC0.read_text())
    assert info["n_enc"] == 164_490 and info["steps_per_epoch"] == 5_140 and len(starts) == 5_140
    assert info["probe64_digest"] == ugc0["probe"]["digest"]
    info["flatten_law_digest"] = flat_digest
    batches, probe = UG.probe_batches(tok)
    held = [batches[i] for i in probe["held"]]
    info["held_panel"] = {"n": len(held), "probe_digest": probe["digest"], "ugc0_digest": ugc0["probe"]["digest"]}
    if not SMOKE:
        assert probe["digest"] == ugc0["probe"]["digest"]
    if SMOKE:
        # the FME1 smoke pipeline builds the whole non-target arena (OMA smoke anchors + Stage 0 + desk + FME1 treat) under the tag
        if os.environ.get("SMOKE_ARENA_TAG"):
            assert FME1.exists() and STAGE0.exists(), "SMOKE_ARENA_TAG needs an existing smoke arena"      # reuse (second smoke)
        else:
            if FME1.exists():
                raise SystemExit(f"REFUSING: smoke FME1 arena {FME1} exists")
            FME.main()
        OMA.CK_DIR = CK_DIR
    rec = OMA.base_record("fmel1-ladder", tok, info, "cpu", "mps" if torch.backends.mps.is_available() else "cpu")
    rec["oma_source_sha256"] = rec.pop("source_sha256")          # OMA.base_record's field is OMA's own source; this instrument's is self_sha256
    rec.update({"prereg": "FIRST-MOMENT-ERASURE-LADDER-1", "kind": "first_moment_erasure_ladder/ladder", "writer": WRITER, "n_pred_literal": N_PRED,
                "self_sha256": sha256_file(__file__), "fme1_source_sha256": sha256_file(FME.__file__), "ck_dir": str(CK_DIR), "leg_steps": LEG_FULL, "target": ANCHOR + LEG_FULL,
                "horizons": GRID, "qualification_horizons": QUAL, "ladder": LADDER, "arms": {}, "arm_order": ARM_ORDER,
                "law": {"BAR1_TOL": BAR1_TOL, "R1_TOL": R1_TOL, "COS1_MIN": COS1_MIN, "FORGOTTEN": FORGOTTEN, "PERSISTENT": PERSISTENT, "ALPHA_BAND": list(ALPHA_BAND),
                        "COS_SHARED": COS_SHARED, "ALPHA_FLAT": ALPHA_FLAT, "COS_DECOR": COS_DECOR, "CE_ABS": CE_ABS, "TAIL_H": TAIL_H, "MIN_FREE_BYTES": MIN_FREE_BYTES,
                        "regime_order": ["FORGOTTEN", "REGIME-UNRESOLVED", "MAGNITUDE-SCALED PERSISTENT", "NONLINEAR DIRECTION-SHARED", "TRAJECTORY-SENSITIVE", "INTERMEDIATE"],
                        "zero_norm": "n = 0, R = 0 (UNDEFINED if ||dW_1|| = 0), ln UNDEFINED, cosine UNDEFINED; alpha needs four ln, cosmin six cosines; UNDEFINED -> null + reason"}})
    rec.pop("eps_twin", None)
    rec["writers"] = {WRITER: rec["writers"][WRITER]}           # the single writer this instrument knows

    def stream(row):
        with STREAM.open("a") as f:
            f.write(json.dumps(row) + "\n")

    t0 = time.time()
    try:
        rec = mode_ladder(tok, enc, starts, info, segs, d, held, rec, stream)
    finally:
        rec["wall_s"] = round(time.time() - t0, 1); rec["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        RECEIPT.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"[fmel1] {rec.get('status')} in {rec['wall_s']} s -> {RECEIPT}", flush=True)
    if rec.get("status") != "DONE":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
