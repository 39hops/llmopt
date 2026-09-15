"""FIRST-MOMENT-ERASURE-LADDER-2 instrument (PRE-REG RESULTS L75924): the
resolved-scale ladder. Writer A at the validated A@7200 arena, FOUR fresh
continuous CPU-deterministic legs 7201..15420: the native control C and
exp_avg <- (1 - eps) exp_avg for eps in {1e-2, 1e-1, 1} (the amplitudes
the FMEL1 target preflight resolved with large margin). Writer A only; no
writer-B code path. FMEL1 (VERDICT L75795) is untouched and permanently
NOT-ADJUDICABLE; this instrument only READS its locked receipt.

Thin sibling of scratch/first_moment_erasure_ladder.py (L1): the leg
mechanics (one_step, long_leg with in-line BAR 0, apply_eps, cosine,
group shares, disk preflight, model loading) are IMPORTED from L1, whose
import asserts the whole FME1 / OMA1 pin law; this module adds L1's own
source and the locked FMEL1 receipt to the pins. What is NEW here and
registered for this rung: the three-point ladder, the local decade slopes
alpha_low = log10(||dW_1e-1|| / ||dW_1e-2||) and alpha_high =
log10(||dW_1|| / ||dW_1e-1||) beside an INDEPENDENT least-squares
alpha_global, the mechanical identity alpha_global == (alpha_low +
alpha_high) / 2 asserted whenever all three are defined (a log-uniform
ladder makes the regression slope the mean of the local slopes; the
regression is computed on its own so the check is real), the regime law
requiring BOTH local slopes in the band for any magnitude-scaled claim,
h_curve, and the preflight receipt that separates construction
identities (R_1 = 1, cos(e1, e1) = 1) from the informative checks
(e1e-2, e1e-1, and the sealed n_1 endpoint).

Registered mode (MODE=ladder; the only one), in this order:
  1. pins, locked receipt shas (Stage-0, desk, FME1, FMEL1), anchor sha /
     digest, leg digest, disk preflight (16 GiB), reference digests read
     from the LOCKED RECEIPTS ONLY (Stage-0 C and FME1 Z / E at h = 1 / 5 /
     20 / 100 / 900; the FMEL1 preflight h = 1 digests for every arm) --
     no historical checkpoint is opened as state.
  2. ONE-STEP PREFLIGHT: every arm bound fresh, one step; the registered
     law unchanged (|R - 1| <= 1e-3, cos to dW_1 >= 0.999, |n_1(1) -
     n_pred| <= 0.02); every arm's digest must equal its locked FMEL1
     preflight digest. Failure -> REGIME-UNRESOLVED, stop, exit 3.
  3. four long legs (C, e1, e1e-2, e1e-1), each a fresh bind; in-line
     BAR 0 (mismatch -> NOT-RUN); snapshots at the 12 grid horizons, opt
     at H; HELD-32 CE; C's in-line substrate readout at 15300.
  4. readouts, substrate readout at 15420, gates, BAR 2 regime (local +
     global slopes), BAR 3 locus (h_lin / h_dec / h_curve), BAR 4
     absolute function bar, BAR 5 cooled tail.   -> logs/fmel2/ladder.json

SMOKE=1: L1's smoke pipeline builds the synthetic non-target arena (OMA
smoke Stage 0 + FME1 smoke treat + the L1 smoke preflight receipt, which
refuses on that arena by design and so supplies the h = 1 reference
digests), then this instrument runs on it with the smoke grid [1, 2, 3].
Receipts logs/fmel2/smoke<TAG>_*, snapshots checkpoints/fmel2_smoke.
SMOKE_ARENA_TAG reuses an existing smoke arena; SMOKE_PREFLIGHT_BYPASS
and SMOKE_TAMPER_REF are SMOKE-only knobs (ignored otherwise).

Usage: MODE=ladder .venv/bin/python scratch/first_moment_erasure_ladder2.py
       SMOKE=1 SMOKE_TAG=l2a .venv/bin/python scratch/first_moment_erasure_ladder2.py
"""
import datetime
import hashlib
import json
import math
import os
import signal
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
PREFLIGHT_BYPASS = SMOKE and os.environ.get("SMOKE_PREFLIGHT_BYPASS", "0") == "1"
# Registered whole-run wall cap (PRE-REG L75924 stop law (a): killed and NOT-RUN at 7 h), armed at the top of main() so setup,
# preflight, legs, readouts and the receipt write are all inside it. Real mode is FIXED at 25200 s: no environment variable
# extends or disables it. SMOKE may shorten it (SMOKE_MAX_WALL_S) solely to qualify the handler.
MAX_WALL_S_REAL = 25200
MAX_WALL_S = int(os.environ["SMOKE_MAX_WALL_S"]) if (SMOKE and os.environ.get("SMOKE_MAX_WALL_S")) else MAX_WALL_S_REAL
assert SMOKE or MAX_WALL_S == MAX_WALL_S_REAL
WALL_GRACE_S = 120                    # if the run is still alive this long after the limit fired (stuck in a C call), hard-exit
PHASE = {"name": "setup"}             # the phase the wall limit interrupted, for the receipt

import first_moment_erasure_ladder as L1  # noqa: E402  (asserts the FME1 / OMA1 pins + leg-path sha at import)
import numpy as np  # noqa: E402
import torch  # noqa: E402

FME = L1.FME
OMA = L1.OMA
UG = L1.UG
TM = L1.TM
state_digest = L1.state_digest
gate_eval = L1.gate_eval
sha256_file = L1.sha256_file
apply_eps = L1.apply_eps
cosine = L1.cosine
group_shares = L1.group_shares
disk_preflight = L1.disk_preflight
load_model_sd = L1.load_model_sd
one_step = L1.one_step
long_leg = L1.long_leg
Abort = L1.Abort
qualify = L1.qualify

# ---------------------------------------------------------------- pins (L1's set + L1 itself + the locked FMEL1 receipt)
L1_SOURCE_SHA = "fc4a0927c3ae2bb1f388408ab785697e7312d66d9e8cf073a0cded50450c64de"      # scratch/first_moment_erasure_ladder.py at the seal
FMEL1_SHA = "9005d2548b76b0f7cf8c3ae39de1c4a02ff3b9cc7f8a6af29015af3c11919351"         # logs/fmel1/ladder.json (locked)
PINS = dict(L1.PINS, **{"scratch/first_moment_erasure_ladder.py": L1_SOURCE_SHA})
_bad = FME.check_pins(PINS)
if _bad:
    raise SystemExit("PIN MISMATCH: " + json.dumps({k: {"expected": v[0][:16], "actual": (v[1] or "MISSING")[:16]} for k, v in _bad.items()}))

WRITER = L1.WRITER
N_PRED = L1.N_PRED
ANCHOR_SHA = L1.ANCHOR_SHA
ANCHOR = L1.ANCHOR
LEG_FULL = L1.LEG_FULL
H_END = L1.H_END
GRID = L1.GRID
QUAL = L1.QUAL
LADDER = [1e-2, 1e-1, 1.0]
ARMS = {"C": 0.0, "e1e-2": 1e-2, "e1e-1": 1e-1, "e1": 1.0}
EPS_ARMS = {a: e for a, e in ARMS.items() if e > 0}
ARM_ORDER = ["C", "e1", "e1e-2", "e1e-1"]
QUALIFIED = {"C": "C", "e1": "Z", "e1e-2": "E"}                              # fresh arm -> reference arm in the locked Stage-0 / FME1 receipts
ARENA = {"anchor": 7200, "leg": 8220, "grid": [1, 5, 20, 100, 300, 900, 1800, 3080, 4500, 6000, 7200, 8220], "qual": [1, 5, 20, 100, 900], "threads": 8, "ladder": LADDER}
# law (literal; PRE-REG L75924, inherited thresholds from AMENDMENT L75568)
BAR1_TOL, R1_TOL, COS1_MIN = L1.BAR1_TOL, L1.R1_TOL, L1.COS1_MIN
FORGOTTEN, PERSISTENT = L1.FORGOTTEN, L1.PERSISTENT
ALPHA_BAND, COS_SHARED, ALPHA_FLAT, COS_DECOR = L1.ALPHA_BAND, L1.COS_SHARED, L1.ALPHA_FLAT, L1.COS_DECOR
LOCAL_FLAT = 0.40                     # TRAJECTORY-SENSITIVE: max(alpha_low, alpha_high) <= LOCAL_FLAT
CURVE_GAP = 0.20                      # h_curve: |alpha_low - alpha_high| > CURVE_GAP
ALPHA_IDENTITY_TOL = 1e-9             # alpha_global == (alpha_low + alpha_high) / 2 on a log-uniform ladder (floating reduction only)
CE_ABS = L1.CE_ABS
TAIL_H = L1.TAIL_H
MIN_FREE_BYTES = L1.MIN_FREE_BYTES
END_MODEL, END_DIGEST, END_SHA = L1.END_MODEL, L1.END_DIGEST, L1.END_SHA
MID_STEP, MID_MODEL, MID_DIGEST, MID_SHA = L1.MID_STEP, L1.MID_MODEL, L1.MID_DIGEST, L1.MID_SHA
assert (BAR1_TOL, R1_TOL, COS1_MIN, FORGOTTEN, PERSISTENT, ALPHA_BAND, COS_SHARED, ALPHA_FLAT, COS_DECOR, CE_ABS) == (0.02, 1e-3, 0.999, 0.05, 0.25, (0.80, 1.20), 0.90, 0.20, 0.50, 0.005)

STAGE0 = L1.STAGE0
DESK = L1.DESK
FME1 = L1.FME1
FMEL1 = Path(f"logs/fmel1/smoke{FME.ARENA_TAG}_ladder.json" if SMOKE else "logs/fmel1/ladder.json")
LOCK = L1.LOCK
OUT_DIR = Path("logs/fmel2")
CK_DIR = Path("checkpoints/fmel2_smoke" if SMOKE else "checkpoints/fmel2")
RECEIPT = OUT_DIR / (f"smoke{SMOKE_TAG}_ladder.json" if SMOKE else "ladder.json")
STREAM = OUT_DIR / (f"smoke{SMOKE_TAG}_ladder.jsonl" if SMOKE else "ladder.jsonl")
if not SMOKE:
    assert (ANCHOR, LEG_FULL, GRID, QUAL, OMA.THREADS) == (ARENA["anchor"], ARENA["leg"], ARENA["grid"], ARENA["qual"], ARENA["threads"]), "arena literals"
    assert ANCHOR + LEG_FULL == OMA.TOTAL == 15_420
    assert str(L1.FME1) == "logs/fme1/treat.json" and str(FMEL1) == "logs/fmel1/ladder.json"
assert GRID[-1] == H_END and all(h in GRID for h in QUAL) and TAIL_H in GRID
# L1.long_leg reads L1's module globals (GRID, QUAL, H_END, MID_MODEL); this rung binds the same objects above and asserts the ARENA literals
OMA.CK_DIR = CK_DIR                            # this instrument's snapshots never land under checkpoints/oma1, fme1 or fmel1


# ---------------------------------------------------------------- pure laws
def finite(x):
    return x is not None and math.isfinite(x)


def alpha_regression(eps_of, ln):
    """Independent least-squares slope of ln||dW|| against ln eps over the arms in `ln` (all defined); None with fewer than two points."""
    arms = [a for a in eps_of if ln.get(a) is not None]
    if len(arms) < 2:
        return None
    x = np.array([math.log(eps_of[a]) for a in arms]); y = np.array([ln[a] for a in arms])
    xm, ym = x.mean(), y.mean()
    return float(((x - xm) @ (y - ym)) / ((x - xm) @ (x - xm)))


def decade_slope(ln, hi, lo):
    """log10(||dW_hi|| / ||dW_lo||) from the natural logs; None if either is UNDEFINED."""
    if ln.get(hi) is None or ln.get(lo) is None:
        return None
    return float((ln[hi] - ln[lo]) / math.log(10.0))


def horizon_metrics(W0, WC, W, eps_of=None, strict_identity=False):
    """Per-horizon readouts for the three-point ladder from flats: W0 anchor, WC control, W = {arm: flat}. Zero-norm and non-finite
    law: UNDEFINED values are None with a reason under `undefined`; never NaN. alpha_global is an independent regression; alpha_low /
    alpha_high are the decade slopes; when all three are defined the log-uniform identity alpha_global == (alpha_low + alpha_high) / 2
    is asserted to ALPHA_IDENTITY_TOL (a genuine implementation check, recorded under `alpha_identity_gap`)."""
    eps_of = eps_of or EPS_ARMS
    dC = WC - W0
    legn = float(np.linalg.norm(dC))
    dev = {a: W[a] - WC for a in eps_of}
    absn = {a: float(np.linalg.norm(dev[a])) for a in eps_of}
    empty = {"leg_abs": legn if math.isfinite(legn) else None, "abs": {a: (absn[a] if math.isfinite(absn[a]) else None) for a in eps_of},
             "n": {a: None for a in eps_of}, "R": {a: None for a in eps_of}, "ln": {a: None for a in eps_of}, "group_share": {a: None for a in eps_of},
             "leg_cos": {a: None for a in eps_of}, "alpha_global": None, "alpha_low": None, "alpha_high": None, "alpha_identity_gap": None, "alpha_identity_ok": None,
             "pair_cos": {}, "cosmin": None, "undefined": {}}
    nonfinite = [a for a in eps_of if not math.isfinite(absn[a])] + (["control"] if not math.isfinite(legn) else [])
    if nonfinite:
        empty["undefined"]["nonfinite"] = f"non-finite deviation norm in {nonfinite}"
        return empty
    out = empty
    ref = [a for a, e in eps_of.items() if e == 1.0][0]
    for a in eps_of:
        if legn > 0:
            out["n"][a] = absn[a] / legn
        else:
            out["undefined"][f"n:{a}"] = "control leg displacement is zero"
        if absn[ref] > 0:
            out["R"][a] = absn[a] / (eps_of[a] * absn[ref])
        else:
            out["undefined"][f"R:{a}"] = "eps = 1 deviation norm is zero"
        if absn[a] > 0:
            out["ln"][a] = math.log(absn[a])
        else:
            out["undefined"][f"ln:{a}"] = "deviation norm is zero"
        out["leg_cos"][a] = cosine(W[a] - W0, dC)
    out["alpha_low"] = decade_slope(out["ln"], "e1e-1", "e1e-2")
    out["alpha_high"] = decade_slope(out["ln"], "e1", "e1e-1")
    if all(out["ln"][a] is not None for a in eps_of):
        out["alpha_global"] = alpha_regression(eps_of, out["ln"])
        gap = abs(out["alpha_global"] - (out["alpha_low"] + out["alpha_high"]) / 2.0)
        out["alpha_identity_gap"] = gap
        out["alpha_identity_ok"] = bool(gap <= ALPHA_IDENTITY_TOL)
        if not out["alpha_identity_ok"]:
            out["undefined"]["alpha_identity"] = f"log-uniform identity violated: alpha_global {out['alpha_global']} v mean of local slopes (gap {gap})"
            if strict_identity:
                raise AssertionError(out["undefined"]["alpha_identity"])          # preflight: fail loudly before any long leg
    else:
        out["undefined"]["alpha_global"] = "a deviation norm is zero (fewer than three defined logarithms)"
        if out["alpha_low"] is None:
            out["undefined"]["alpha_low"] = "a decade endpoint is UNDEFINED"
        if out["alpha_high"] is None:
            out["undefined"]["alpha_high"] = "a decade endpoint is UNDEFINED"
    arms = list(eps_of)
    pairs = {}
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            pairs[f"{arms[i]}|{arms[j]}"] = cosine(dev[arms[i]], dev[arms[j]])
    out["pair_cos"] = pairs
    if all(v is not None for v in pairs.values()):
        out["cosmin"] = float(min(pairs.values()))
    else:
        out["undefined"]["cosmin"] = "a pair cosine is UNDEFINED (zero deviation)"
    return out


IDENTITY_CHECKS = ["R:e1", "cos:e1"]        # construction identities: R_1 = ||dW_1|| / (1 * ||dW_1||) and cos(dW_1, dW_1)


def preflight_law(m1, n_pred=N_PRED):
    """BAR 1 on the one-step metrics: the registered law unchanged. The receipt separates construction identities (R_1, cos(e1, e1))
    from the informative checks (R and cos for e1e-2 / e1e-1, the sealed n_1 endpoint). Returns (ok, detail)."""
    detail = {"n_1": m1["n"].get("e1"), "n_pred": n_pred, "R": m1["R"], "cos_to_e1": {}, "identities": list(IDENTITY_CHECKS), "informative": [], "fails": []}
    n1 = m1["n"].get("e1")
    detail["informative"].append("n_1")
    if not finite(n1) or abs(n1 - n_pred) > BAR1_TOL:
        detail["fails"].append(f"n_1(1) {n1} v n_pred {n_pred} (tol {BAR1_TOL})")
    for a in EPS_ARMS:
        r = m1["R"][a]
        if a == "e1":
            c = 1.0
        else:
            c = m1["pair_cos"].get(f"{a}|e1", m1["pair_cos"].get(f"e1|{a}"))
            detail["informative"] += [f"R:{a}", f"cos:{a}"]
        detail["cos_to_e1"][a] = c
        if not finite(r) or abs(r - 1.0) > R1_TOL:
            detail["fails"].append(f"R_{a}(1) {r} (tol {R1_TOL})")
        if not finite(c) or c < COS1_MIN:
            detail["fails"].append(f"cos(dW_{a}(1), dW_1(1)) {c} < {COS1_MIN}")
    return not detail["fails"], detail


def in_band(x):
    return x is not None and ALPHA_BAND[0] <= x <= ALPHA_BAND[1]


def regime(mH, preflight_ok):
    """BAR 2 at H, first match wins: FORGOTTEN / REGIME-UNRESOLVED / MAGNITUDE-SCALED PERSISTENT / NONLINEAR DIRECTION-SHARED /
    TRAJECTORY-SENSITIVE / INTERMEDIATE. Any magnitude-scaled claim needs BOTH local slopes in the band."""
    ns = [mH["n"][a] for a in mH["n"]]
    if all(finite(v) for v in ns) and all(v <= FORGOTTEN for v in ns):
        return "FORGOTTEN"
    keys = (mH.get("alpha_global"), mH.get("alpha_low"), mH.get("alpha_high"), mH.get("cosmin"), mH["n"].get("e1"))
    if not preflight_ok or any(not finite(v) for v in ns) or any(not finite(v) for v in keys) or mH.get("alpha_identity_ok") is False:
        return "REGIME-UNRESOLVED"                       # an identity violation after the legs is a recorded refusal, never a crash
    ag, al, ah, cm, n1 = keys
    if in_band(al) and in_band(ah) and cm >= COS_SHARED and n1 >= PERSISTENT:
        return "MAGNITUDE-SCALED PERSISTENT"
    if cm >= COS_SHARED and not (in_band(al) and in_band(ah)):
        return "NONLINEAR DIRECTION-SHARED"
    if ag <= ALPHA_FLAT and max(al, ah) <= LOCAL_FLAT and cm <= COS_DECOR:
        return "TRAJECTORY-SENSITIVE"
    return "INTERMEDIATE"


def locus(metrics_by_h):
    """BAR 3: h_lin = last grid h with both local slopes in the band and cosmin >= COS_SHARED; h_dec = first h with cosmin <= COS_DECOR;
    h_curve = first h with |alpha_low - alpha_high| > CURVE_GAP. UNDEFINED horizons excluded per readout. None = not reached."""
    h_lin = h_dec = h_curve = None
    for h in sorted(metrics_by_h):
        m = metrics_by_h[h]
        al, ah, cm = m.get("alpha_low"), m.get("alpha_high"), m.get("cosmin")
        if finite(cm):
            if in_band(al) and in_band(ah) and cm >= COS_SHARED:
                h_lin = h
            if h_dec is None and cm <= COS_DECOR:
                h_dec = h
        if h_curve is None and finite(al) and finite(ah) and abs(al - ah) > CURVE_GAP:
            h_curve = h
    return {"h_lin": h_lin, "h_dec": h_dec, "h_curve": h_curve}


def function_bar(dce_H):
    per = {}
    for a, v in dce_H.items():
        per[a] = "NEUTRAL" if abs(v) <= CE_ABS else ("HARMED" if v > 0 else "HELPED")
    return per, ("FUNCTION-NEUTRAL" if all(v == "NEUTRAL" for v in per.values()) else "FUNCTION-" + ",".join(f"{a}:{v}" for a, v in per.items() if v != "NEUTRAL"))


class WallLimit(SystemExit):
    """Raised in the main thread by the SIGALRM handler when the registered wall cap is reached."""


def install_wall_limit(limit_s=None, receipt=None, rec=None):
    """Arm the registered wall cap: SIGALRM at `limit_s` raises WallLimit (caught by main, which books NOT-RUN and exits 3); a
    second alarm WALL_GRACE_S later, reached only if the first could not unwind (a long C call), writes a minimal NOT-RUN receipt
    itself and hard-exits 4. Returns the limit armed."""
    limit_s = MAX_WALL_S if limit_s is None else limit_s
    state = {"fired": 0}

    def handler(signum, frame):
        state["fired"] += 1
        if state["fired"] == 1:
            signal.alarm(WALL_GRACE_S)
            raise WallLimit(3)
        if receipt is not None and rec is not None:
            rec.update({"status": "NOT-RUN", "regime": "NOT-RUN", "label": wall_label(), "wall_limit": wall_record(hard_exit=True),
                        "wall_s": round(time.time() - PHASE.get("t0", time.time()), 1), "ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
            receipt.write_text(json.dumps(rec, indent=1) + "\n")
        os._exit(4)

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(limit_s))
    return int(limit_s)


def set_phase(name):
    PHASE["name"] = name
    print(f"[fmel2] phase {name} at {time.time() - PHASE.get('t0', time.time()):.1f} s", flush=True)


def wall_label():
    return f"NOT-RUN (registered wall limit {MAX_WALL_S} s reached during {PHASE['name']}; stop law (a), PRE-REG L75924)"


def wall_record(hard_exit=False):
    return {"limit_s": MAX_WALL_S, "real_limit_s": MAX_WALL_S_REAL, "phase": PHASE["name"], "reason": "registered whole-run wall cap", "hard_exit": hard_exit,
            "elapsed_s": round(time.time() - PHASE.get("t0", time.time()), 1)}


def preflight_disposition(pf_ok, bypass, digests_match):
    """The registered early returns after the one-step preflight: (status, regime, label) or None to proceed to the long legs.
    A failed preflight books REGIME-UNRESOLVED (unless the SMOKE-only bypass is set); a preflight digest that differs from the
    locked FMEL1 preflight digest is a substrate change and books NOT-RUN. The bypass never turns a failed preflight into a regime."""
    if not pf_ok and not bypass:
        return "PREFLIGHT-FAILED", "REGIME-UNRESOLVED", label("REGIME-UNRESOLVED", "FUNCTION-NOT-MEASURED")
    if not digests_match:
        return "NOT-RUN", "NOT-RUN", "NOT-RUN (preflight digests differ from the locked FMEL1 preflight digests: substrate change)"
    return None


def label(reg, func):
    return f"{reg}+{func} [writer A only, one anchor, one seed lineage, CPU deterministic, eps >= 1e-2; gate descriptive]"


def reference_digests(stage0, fme1, fmel1, horizons=None):
    """{fresh arm: {h: locked state digest}} read from the LOCKED RECEIPTS only: Stage-0 C and FME1 Z / E at the qualification
    horizons, plus the FMEL1 preflight h = 1 digest for every arm of this ladder. No checkpoint is opened."""
    horizons = QUAL if horizons is None else horizons
    refs = {"C": {h: stage0["cells"][WRITER]["C"]["snapshots"][str(h)]["state_digest"] for h in horizons}}
    for arm, ref in (("e1", "Z"), ("e1e-2", "E")):
        refs[arm] = {h: fme1["arms"][ref]["snapshots"][str(h)]["state_digest"] for h in horizons}
    pf = fmel1["preflight"]["arms"]
    for arm in ARMS:
        d1 = pf[arm]["digest"]
        if arm in refs:
            assert refs[arm][1] == d1, f"FMEL1 preflight digest for {arm} disagrees with the Stage-0 / FME1 h = 1 digest"
        else:
            refs[arm] = {1: d1}
    return refs


def assert_fmel1_provenance(fmel1, lock, stage0, fme1):
    """The FMEL1 receipt this rung pins: sha equal to the source literal and the lock; bound to the same Stage-0 / FME1 receipts and
    anchor; status PREFLIGHT-FAILED (no long leg ran there); the five preflight digests present."""
    s = sha256_file(FMEL1)
    assert s == FMEL1_SHA == FME.locked_sha(lock, str(FMEL1)), "FMEL1 receipt v the source literal / the receipt lock"
    assert fmel1["pins"]["stage0_sha256"] == FME.STAGE0_SHA == sha256_file(STAGE0) and fmel1["pins"]["fme1_sha256"] == L1.FME1_SHA == sha256_file(FME1)
    assert fmel1["anchor"]["state_digest"] == stage0["cells"][WRITER]["C"]["bind"]["state_digest"] == fme1["anchor"]["state_digest"]
    assert fmel1["status"] == "PREFLIGHT-FAILED" and fmel1["smoke"] is False and fmel1["writer"] == WRITER and fmel1["n_pred_literal"] == N_PRED
    assert set(fmel1["preflight"]["arms"]) >= set(ARMS)
    return {"fmel1_sha256": s, "fmel1_commit": fmel1["commit"], "fmel1_status": fmel1["status"]}


# ---------------------------------------------------------------- the registered mode
def mode_ladder(tok, enc, starts, info, segs, d, held, rec, stream):
    assert OMA.CK_DIR == CK_DIR, (OMA.CK_DIR, CK_DIR)          # snapshots land under this rung's tree only
    set_phase("provenance")
    stage0 = json.loads(STAGE0.read_text()); desk = json.loads(DESK.read_text()); fme1 = json.loads(FME1.read_text()); fmel1 = json.loads(FMEL1.read_text())
    rec["pins"] = {"sources": dict(PINS), "leg_path_symbols": FME.LEG_PATH_SYMBOLS, "leg_path_sha_measured": FME.leg_path_sha(), "leg_path_sha": FME.LEG_PATH_SHA,
                   "stage0_receipt": str(STAGE0), "stage0_sha256": sha256_file(STAGE0), "desk_receipt": str(DESK), "desk_sha256": sha256_file(DESK),
                   "fme1_receipt": str(FME1), "fme1_sha256": sha256_file(FME1), "fmel1_receipt": str(FMEL1), "fmel1_sha256": sha256_file(FMEL1), "data_files": FME.data_file_shas()}
    if not SMOKE:
        lock = json.loads(LOCK.read_text())
        rec["pins"].update(FME.assert_provenance(stage0, desk, lock, sha256_file(OMA.WRITERS[WRITER]["anchor"]), torch.get_num_threads(), torch.__version__, np.__version__))
        rec["pins"].update(L1.assert_fme1_provenance(fme1, lock, stage0))
        rec["pins"].update(assert_fmel1_provenance(fmel1, lock, stage0, fme1))
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
    if not SMOKE:
        assert rec["leg_slices_digest_full"] == fmel1["leg_slices_digest_full"], "future stream v the FMEL1 full-leg digest"
    rec["leg"] = {"first_step": ANCHOR + 1, "last_step": ANCHOR + LEG_FULL, "n_steps": LEG_FULL, "first_slice": list(slices[0]), "last_slice": list(slices[-1]),
                  "epoch_position_first": OMA.epoch_position(ANCHOR + 1, info["n_enc"]), "epoch_position_last": OMA.epoch_position(ANCHOR + LEG_FULL, info["n_enc"])}
    ok_disk, free = disk_preflight()
    rec["disk_preflight"] = {"free_bytes": free, "min_free_bytes": MIN_FREE_BYTES, "ok": ok_disk}
    if not ok_disk:
        raise SystemExit(f"DISK PREFLIGHT REFUSED: {free / (1 << 30):.1f} GiB free < {MIN_FREE_BYTES / (1 << 30):.0f} GiB")
    try:
        refs = reference_digests(stage0, fme1, fmel1)
    except AssertionError as e:
        rec["status"] = "NOT-RUN"; rec["regime"] = "NOT-RUN"; rec["label"] = f"NOT-RUN (locked receipts disagree: {e})"
        return rec
    if SMOKE and os.environ.get("SMOKE_TAMPER_REF") == "1":
        refs["C"][QUAL[1]] = "0" * 64                            # abort smoke: a corrupted reference digest must book NOT-RUN
        rec["smoke_tampered_ref"] = f"C@{QUAL[1]}"
    if SMOKE and os.environ.get("SMOKE_TAMPER_PF") == "1":
        refs["e1e-1"][1] = "0" * 64                              # substrate-change smoke: a preflight digest mismatch must book NOT-RUN before any leg
        rec["smoke_tampered_ref"] = "e1e-1@1 (preflight)"
    rec["reference_digests"] = {a: {str(h): v for h, v in hs.items()} for a, hs in refs.items()}
    set_phase("anchor-and-substrate-load")
    anchor_sd = load_model_sd(OMA.WRITERS[WRITER]["anchor"])
    rec["anchor"] = {"path": OMA.WRITERS[WRITER]["anchor"], "file_sha256": sha256_file(OMA.WRITERS[WRITER]["anchor"]), "state_digest": state_digest(anchor_sd)}
    if not SMOKE:
        assert rec["anchor"]["state_digest"] == stage0["cells"][WRITER]["C"]["bind"]["state_digest"] == fme1["anchor"]["state_digest"] == fmel1["anchor"]["state_digest"]
    W0 = OMA.flat(anchor_sd, segs, d)
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
    set_phase("preflight")
    t0 = time.time()
    pf = {"arms": {}}
    W1 = {}
    for arm in ARM_ORDER:
        f, dg, binfo, touched, grp = one_step(arm, ARMS[arm], tok, enc, slices, segs, d)
        pf["arms"][arm] = {"digest": dg, "tensors_touched": touched, "group_next": grp, "fmel1_digest": refs[arm][1], "digest_match": dg == refs[arm][1]}
        assert touched == (0 if arm == "C" else len(UG.KEYS)), (arm, touched)
        W1[arm] = f
    m1 = horizon_metrics(W0, W1["C"], {a: W1[a] for a in EPS_ARMS}, strict_identity=True)
    for a in m1["abs"]:
        m1["group_share"][a] = group_shares(W1[a] - W1["C"], segs)
    pf["metrics"] = m1
    pf["ok"], pf["detail"] = preflight_law(m1, n_pred)
    pf["digests_match_fmel1"] = all(v["digest_match"] for v in pf["arms"].values())
    pf["wall_s"] = round(time.time() - t0, 1)
    rec["preflight"] = pf
    print(f"[fmel2] preflight {'PASS' if pf['ok'] else 'FAIL'} digests_match_fmel1 {pf['digests_match_fmel1']} n_1(1) {m1['n'].get('e1')} R {m1['R']} cos {pf['detail']['cos_to_e1']}", flush=True)
    del W1
    rec["smoke_preflight_bypassed"] = bool(PREFLIGHT_BYPASS and not pf["ok"])
    assert not (rec["smoke_preflight_bypassed"] and not SMOKE)
    early = preflight_disposition(pf["ok"], PREFLIGHT_BYPASS, pf["digests_match_fmel1"])
    if early is not None:
        rec["status"], rec["regime"], rec["label"] = early
        return rec
    # ---- the four long legs
    rec["arms"] = {}
    rec["status"] = "RUNNING"
    for arm in ARM_ORDER:
        cell = {}
        rec["arms"][arm] = cell
        set_phase(f"leg:{arm}")
        try:
            snaps = long_leg(arm, ARMS[arm], tok, enc, slices, segs, d, held, refs, pf["arms"][arm]["digest"], stream, cell, mid)
        except Abort:
            rec["status"] = "NOT-RUN"
            rec["regime"] = "NOT-RUN"
            rec["label"] = f"NOT-RUN ({cell.get('abort')})"
            print(f"[fmel2] ABORT {arm}: {cell.get('abort')}", flush=True)
            return rec
        del snaps
        print(f"[fmel2] {arm} leg done {cell['wall_s']} s ({cell['it_per_s']} it/s) loss_last {cell['loss_last']:.4f}", flush=True)
    # ---- readouts per grid horizon from the fresh snapshots
    set_phase("readouts")
    metrics, dce, ce = {}, {}, {}
    prev_dev = {}
    W_end_C = None
    for h in GRID:
        WC = OMA.flat(load_model_sd(rec["arms"]["C"]["snapshots"][str(h)]["path"]), segs, d)
        W = {a: OMA.flat(load_model_sd(rec["arms"][a]["snapshots"][str(h)]["path"]), segs, d) for a in EPS_ARMS}
        m = horizon_metrics(W0, WC, W)
        m["rotation"] = {}
        for a in W:
            dev = W[a] - WC
            m["group_share"][a] = group_shares(dev, segs)
            m["rotation"][a] = cosine(dev, prev_dev[a]) if a in prev_dev else None
            prev_dev[a] = dev
        m["sched"] = rec["arms"]["C"]["sched_at_grid"].get(str(h))
        ce[str(h)] = {a: rec["arms"][a]["ce_held"][str(h)] for a in ARMS}
        dce[str(h)] = {a: rec["arms"][a]["ce_held"][str(h)] - rec["arms"]["C"]["ce_held"][str(h)] for a in EPS_ARMS}
        metrics[h] = m
        if h == H_END:
            W_end_C = WC
    rec["metrics"] = {str(h): m for h, m in metrics.items()}
    rec["ce_held"] = ce
    rec["dce"] = dce
    W_end_booked = OMA.flat(end_sd, segs, d)
    d_end = W_end_C - W_end_booked
    end_abs, end_leg = float(np.linalg.norm(d_end)), float(np.linalg.norm(W_end_booked - W0))
    rec["substrate_end"] = {"step": ANCHOR + H_END, "booked": str(END_MODEL), "booked_digest": end_dg, "booked_sha256": sha256_file(END_MODEL),
                            "fresh_digest": rec["arms"]["C"]["snapshots"][str(H_END)]["state_digest"],
                            "rho": (end_abs / end_leg if (finite(end_abs) and finite(end_leg) and end_leg > 0) else None), "abs": (end_abs if finite(end_abs) else None),
                            "ce_fresh": rec["arms"]["C"]["ce_held"][str(H_END)], "ce_booked": end_ce}
    rec["substrate_mid"] = rec["arms"]["C"].get("substrate_mid")
    set_phase("gates")
    dev_gate = "mps" if torch.backends.mps.is_available() else "cpu"
    rec["gate_device"] = dev_gate
    gate = {}
    for name, sd in (("C", load_model_sd(rec["arms"]["C"]["snapshots"][str(H_END)]["path"])), ("e1", load_model_sd(rec["arms"]["e1"]["snapshots"][str(H_END)]["path"])), ("booked_end", end_sd)):
        gm = UG.build(tok, dev_gate); gm.load_state_dict(sd); gm.eval()
        solves, valid = gate_eval(gm, tok, dev_gate, n=(2 if SMOKE else None))
        gate[name] = {"solves": solves, "total": int(sum(solves.values())), "valid_pct": round(valid, 2)}
    rec["gate"] = gate
    # ---- adjudication
    set_phase("adjudication")
    mH = metrics[H_END]
    reg = regime(mH, pf["ok"])
    per, func = function_bar(dce[str(H_END)])
    tail = {a: (mH["abs"][a] / metrics[TAIL_H]["abs"][a] if (finite(mH["abs"][a]) and (metrics[TAIL_H]["abs"][a] or 0) > 0) else None) for a in mH["abs"]}
    over = [(a, str(h)) for h in GRID for a in dce[str(h)] if abs(dce[str(h)][a]) > CE_ABS]
    sign = {a: ("+" if dce[str(H_END)][a] > 0 else "-" if dce[str(H_END)][a] < 0 else "0") for a in EPS_ARMS}
    for a in QUALIFIED:
        assert set(rec["arms"][a]["qualification"]) == {str(h) for h in QUAL}, (a, list(rec["arms"][a]["qualification"]))   # BAR 0 never passes vacuously
    for a in ARMS:
        assert "1" in rec["arms"][a]["qualification"], a                                                                  # every arm was checked at h = 1
    rec["adjudication"] = {"bar0": {a: all(q["ok"] for q in rec["arms"][a]["qualification"].values()) for a in ARMS},
                           "bar1_preflight": pf["ok"], "bar2_regime": reg, "alpha_global_H": mH["alpha_global"], "alpha_low_H": mH["alpha_low"], "alpha_high_H": mH["alpha_high"],
                           "cosmin_H": mH["cosmin"], "n_H": mH["n"], "R_H": mH["R"], "bar3_locus": locus(metrics),
                           "bar4_function": {"per_arm": per, "label": func, "dce_H": dce[str(H_END)], "sign_pattern": sign, "over_threshold": over},
                           "bar5_tail": tail, "undefined_H": mH["undefined"]}
    rec["regime"] = reg
    rec["status"] = "DONE"
    rec["label"] = label(reg, func)
    print(f"[fmel2] alpha_global(H) {mH['alpha_global']} low {mH['alpha_low']} high {mH['alpha_high']} cosmin(H) {mH['cosmin']} n {mH['n']} -> {rec['label']}; "
          f"gate {{ {', '.join(f'{k} {v['total']}' for k, v in gate.items())} }}", flush=True)
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
    t0 = time.time()
    PHASE["t0"] = t0
    rec = {"status": "SETUP", "regime": "NOT-RUN", "prereg": "FIRST-MOMENT-ERASURE-LADDER-2"}          # replaced in place once the base record exists
    armed_s = install_wall_limit(MAX_WALL_S, RECEIPT, rec)
    try:
        _setup_and_run(rec, armed_s)
    except WallLimit:
        rec.update({"status": "NOT-RUN", "regime": "NOT-RUN", "label": wall_label()})
        rec["wall_limit"] = dict(rec.get("wall_limit", {}), armed_s=armed_s, fired=True, **wall_record())
        print(f"[fmel2] WALL LIMIT: {rec['label']}", flush=True)
    except SystemExit as e:                         # a registered refusal (disk preflight, refuse-if-exists inside the mode): keep its vocabulary
        if rec.get("status") in (None, "SETUP", "RUNNING"):
            rec.update({"status": "NOT-RUN", "regime": "NOT-RUN", "label": f"NOT-RUN (refused: {str(e)[:200]}; phase {PHASE['name']})"})
        _write_receipt(rec, t0)
        raise
    except BaseException as e:                      # any other failure: the partial receipt still says what happened, then re-raise
        if rec.get("status") in (None, "SETUP", "RUNNING"):
            rec.update({"status": "CRASHED", "regime": "NOT-RUN", "label": f"CRASHED ({type(e).__name__}: {str(e)[:200]}; phase {PHASE['name']})"})
        _write_receipt(rec, t0)
        raise
    _write_receipt(rec, t0)
    print(f"[fmel2] {rec.get('status')} in {rec['wall_s']} s -> {RECEIPT}", flush=True)
    if rec.get("status") != "DONE":
        raise SystemExit(3)


def _write_receipt(rec, t0):
    """Write the receipt, then disarm the cap (the write itself stays covered by the alarm)."""
    rec["wall_s"] = round(time.time() - t0, 1); rec["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    RECEIPT.write_text(json.dumps(rec, indent=1) + "\n")
    signal.alarm(0)


def _setup_and_run(rec, armed_s):
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
        if os.environ.get("SMOKE_ARENA_TAG"):
            assert FMEL1.exists() and FME1.exists() and STAGE0.exists(), "SMOKE_ARENA_TAG needs an existing smoke arena with an L1 preflight receipt"
            _l1 = json.loads(FMEL1.read_text())
            assert _l1["status"] == "PREFLIGHT-FAILED" and set(_l1["preflight"]["arms"]) >= set(ARMS), "reused smoke arena lacks the L1 preflight digests"
        else:
            # L1's smoke pipeline builds the arena (OMA smoke anchors + Stage 0 + desk + FME1 treat) and its own preflight receipt,
            # which REFUSES on that arena by design (exit 3) and thereby supplies this rung's h = 1 reference digests
            if FMEL1.exists():
                raise SystemExit(f"REFUSING: smoke FMEL1 arena {FMEL1} exists")
            try:
                L1.main()
                raise SystemExit("L1 smoke did not refuse on the synthetic arena (expected PREFLIGHT-FAILED)")
            except SystemExit as e:
                if isinstance(e, WallLimit) or e.code != 3:
                    raise
            assert FMEL1.exists() and json.loads(FMEL1.read_text())["status"] == "PREFLIGHT-FAILED"
        OMA.CK_DIR = CK_DIR
    base = OMA.base_record("fmel2-ladder", tok, info, "cpu", "mps" if torch.backends.mps.is_available() else "cpu")
    rec.clear(); rec.update(base)                       # the same dict object the wall handler holds
    rec["oma_source_sha256"] = rec.pop("source_sha256")
    rec.update({"prereg": "FIRST-MOMENT-ERASURE-LADDER-2", "kind": "first_moment_erasure_ladder2/ladder", "writer": WRITER, "n_pred_literal": N_PRED,
                "self_sha256": sha256_file(__file__), "l1_source_sha256": sha256_file(L1.__file__), "fme1_source_sha256": sha256_file(FME.__file__), "ck_dir": str(CK_DIR),
                "leg_steps": LEG_FULL, "target": ANCHOR + LEG_FULL, "horizons": GRID, "qualification_horizons": QUAL, "ladder": LADDER, "arms": {}, "arm_order": ARM_ORDER,
                "law": {"BAR1_TOL": BAR1_TOL, "R1_TOL": R1_TOL, "COS1_MIN": COS1_MIN, "FORGOTTEN": FORGOTTEN, "PERSISTENT": PERSISTENT, "ALPHA_BAND": list(ALPHA_BAND),
                        "COS_SHARED": COS_SHARED, "ALPHA_FLAT": ALPHA_FLAT, "LOCAL_FLAT": LOCAL_FLAT, "COS_DECOR": COS_DECOR, "CURVE_GAP": CURVE_GAP, "ALPHA_IDENTITY_TOL": ALPHA_IDENTITY_TOL,
                        "CE_ABS": CE_ABS, "TAIL_H": TAIL_H, "MIN_FREE_BYTES": MIN_FREE_BYTES, "preflight_identities": IDENTITY_CHECKS,
                        "regime_order": ["FORGOTTEN", "REGIME-UNRESOLVED", "MAGNITUDE-SCALED PERSISTENT", "NONLINEAR DIRECTION-SHARED", "TRAJECTORY-SENSITIVE", "INTERMEDIATE"],
                        "alpha": "alpha_global = independent least squares over the three eps; alpha_low = log10(||dW_1e-1|| / ||dW_1e-2||); alpha_high = log10(||dW_1|| / ||dW_1e-1||); "
                                 "identity alpha_global == (alpha_low + alpha_high) / 2 asserted whenever all three are defined",
                        "zero_norm": "n = 0, R = 0 (UNDEFINED if ||dW_1|| = 0), ln UNDEFINED, cosine UNDEFINED; non-finite -> every derived value UNDEFINED; alpha_global needs three ln, "
                                     "each local slope its two, cosmin three cosines; UNDEFINED -> null + reason, never NaN"}})
    rec.pop("eps_twin", None)
    rec["writers"] = {WRITER: rec["writers"][WRITER]}
    rec["law"]["MAX_WALL_S"] = MAX_WALL_S
    rec["law"]["MAX_WALL_S_REAL"] = MAX_WALL_S_REAL
    rec["wall_limit"] = {"armed_s": armed_s, "fired": False, "armed_at": "top of main (whole run)"}
    rec["status"] = "SETUP"

    def stream(row):
        with STREAM.open("a") as f:
            f.write(json.dumps(row) + "\n")

    out = mode_ladder(tok, enc, starts, info, segs, d, held, rec, stream)
    assert out is rec


if __name__ == "__main__":
    main()
