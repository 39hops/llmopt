"""RANDOM-DIRECTION-CONTROL-1 instrument (PRE-REG RESULTS L76553, AMENDMENT
-PRE-INSTRUMENT L76847): the family's direction control on writer A at the
validated A@7200 arena. FOUR fresh continuous CPU-deterministic legs
7201..15420: the native control C (asserted bit-exact against the locked
FMEL2 C at all 12 grid horizons) and three seeded random exp_avg
perturbations R1 / R2 / R3, compared against the booked FMEL2 eps = 1e-1
moment-axis arm M, whose 12 snapshots are LOCKED COMPARISON VECTORS (file
sha and state digest asserted against logs/fmel2/ladder.json before use;
never resumed as state). Writer A only; no writer-B code path.

Thin sibling of scratch/first_moment_erasure_ladder2.py (L2): the leg
mechanics come from L1 (one_step / long_leg copied here with the
intervention call replaced, see long_leg_r; the copy is guarded by a
source-identity test), the wall cap and the function bar from L2; this
module adds L2's own source and the locked FMEL2 receipt to the pins.

Construction (AMENDMENT L76847; exact first-step map delta_w = K delta_m,
K = -lr beta1 / (bc1 D), D = sqrt((beta2 v + (1 - beta2) c^2) / bc2) +
1e-8, c = clipped first-batch gradient, K independent of exp_avg): per
group BLOCK0..7 / OUTSIDE, with dm_M = -0.1 m and v_M = K dm_M,
  q ~ N(0, I) in float64 exp_avg coordinates (torch.Generator seed),
  q_perp = q - (<K q, v_M> / ||v_M||^2) dm_M,
  dm_R = (||v_M|| / ||K q_perp||) q_perp,
so K dm_R matches the moment-axis write's per-group norm (shares, total)
and is orthogonal to it group by group; nothing is divided by K. Applied
as exp_avg <- float32(float64(exp_avg) + dm_R); every other state untouched
(digests asserted). The h = 1 targets (norm, shares) are DERIVED from the
sha-pinned FMEL2 e1e-1 h = 1 snapshot against the fresh C.

Registered mode (MODE=control; the only one), in this order:
  1. pins, locked receipt shas (Stage-0, desk, FME1, FMEL1, FMEL2), anchor,
     leg digests, disk preflight (16 GiB), the 12 M snapshot file shas.
  2. CONSTRUCTION: one desk bind (never stepped) gives c, K, m and the
     untouched-state digest; the three draws are shaped.
  3. ONE-STEP PREFLIGHT (BAR 1): every arm bound fresh, one step; fresh C
     h = 1 digest == locked FMEL2 C h = 1; (a) magnitude to 1e-3, (b)
     |cos| to dW_M(1) <= 0.05, (c) group shares to 0.01, (d) pairwise
     |cos| <= 0.05, (e) untouched state. Failure -> CONSTRUCTION-UNRESOLVED,
     stop, exit 3. Analytic-v-realized residuals recorded.
  4. four long legs (C, R1, R2, R3), each a fresh bind; in-line BAR 0 (C at
     all 12 horizons; every arm's h = 1 == its preflight digest); snapshots
     at the 12 grid horizons, opt at H; HELD-32 CE; C's substrate readout.
  5. readouts per horizon (G_r, G_M, A_r, n, cos to M, pairwise, rotation,
     shares, dCE), substrate, descriptive gates, BAR 2 / 3 / 4 / 5.
                                                  -> logs/rdc1/control.json

SMOKE=1: L2's smoke pipeline builds the synthetic non-target arena (OMA
smoke anchors + Stage 0 + FME1 + FMEL1 + the FMEL2 smoke ladder, whose
e1e-1 snapshots are this rung's comparison vectors), then this instrument
runs on it with the smoke grid [1, 2, 3]. Receipts logs/rdc1/smoke<TAG>_*,
snapshots checkpoints/rdc1_smoke. SMOKE_ARENA_TAG reuses an arena;
SMOKE_TAMPER_REF / SMOKE_TAMPER_M / SMOKE_TAMPER_CONSTRUCTION /
SMOKE_PREFLIGHT_BYPASS / SMOKE_MAX_WALL_S are SMOKE-only knobs.

Usage: MODE=control .venv/bin/python scratch/random_direction_control.py
       SMOKE=1 SMOKE_TAG=rd1 .venv/bin/python scratch/random_direction_control.py
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
# Registered whole-run wall cap (PRE-REG L76553 stop law (a), carried verbatim from AMENDMENT L76267): fixed at 25200 s in real mode,
# armed at the top of main(); SMOKE may shorten it (SMOKE_MAX_WALL_S) solely to qualify the handler.
MAX_WALL_S_REAL = 25200
MAX_WALL_S = int(os.environ["SMOKE_MAX_WALL_S"]) if (SMOKE and os.environ.get("SMOKE_MAX_WALL_S")) else MAX_WALL_S_REAL
assert SMOKE or MAX_WALL_S == MAX_WALL_S_REAL
WALL_GRACE_S = 120
PHASE = {"name": "setup"}

import first_moment_erasure_ladder2 as L2  # noqa: E402  (asserts the FME1 / OMA1 / L1 pins + leg-path sha at import)
import numpy as np  # noqa: E402
import torch  # noqa: E402

L1 = L2.L1
FME = L2.FME
OMA = L2.OMA
UG = L2.UG
TM = L2.TM
OG = OMA.OG
state_digest = L2.state_digest
gate_eval = L2.gate_eval
sha256_file = L2.sha256_file
cosine = L2.cosine
group_shares = L2.group_shares
disk_preflight = L2.disk_preflight
load_model_sd = L2.load_model_sd
qualify = L2.qualify
Abort = L2.Abort
finite = L2.finite
function_bar = L2.function_bar

# ---------------------------------------------------------------- pins (L2's set + L2 itself + the locked FMEL2 receipt)
L2_SOURCE_SHA = "8c2749908bdece423864128140de61ef301fa689419095fe7ab5a6582681dd9d"      # scratch/first_moment_erasure_ladder2.py at the seal
FMEL2_SHA = "0faffc61bd65c96d9581aaddd8b47c2379e27ba0cd5400153e9b6bb61015c917"         # logs/fmel2/ladder.json (locked)
PINS = dict(L2.PINS, **{"scratch/first_moment_erasure_ladder2.py": L2_SOURCE_SHA})
_bad = FME.check_pins(PINS)
if _bad:
    raise SystemExit("PIN MISMATCH: " + json.dumps({k: {"expected": v[0][:16], "actual": (v[1] or "MISSING")[:16]} for k, v in _bad.items()}))

WRITER = L2.WRITER
N_PRED = L2.N_PRED
ANCHOR = L2.ANCHOR
LEG_FULL = L2.LEG_FULL
H_END = L2.H_END
GRID = L2.GRID
QUAL = L2.QUAL
SEEDS = {"R1": 2026091501, "R2": 2026091502, "R3": 2026091503}          # frozen (PRE-REG L76553; unchanged by L76847)
R_ARMS = ["R1", "R2", "R3"]
ARM_ORDER = ["C", "R1", "R2", "R3"]
M_ARM, M_EPS = "e1e-1", 0.1                                             # the locked FMEL2 moment-axis comparison arm
ARENA = {"anchor": 7200, "leg": 8220, "grid": [1, 5, 20, 100, 300, 900, 1800, 3080, 4500, 6000, 7200, 8220], "threads": 8}
# law (literal; PRE-REG L76553 + AMENDMENT L76847)
MAG_TOL, COS_M_MAX, SHARE_TOL, COS_PAIR_MAX = 1e-3, 0.05, 0.01, 0.05   # BAR 1 (a) (b) (c) (d)
GENERIC_BAND, SPECIFIC_MAX, DOMINANT_MIN = (0.5, 2.0), 0.25, 4.0      # BAR 2 on A_r(H)
SHARED_LATE, INDEP_LATE, H_AMP_G = 0.50, 0.25, 10.0                    # BAR 3
CE_ABS = L2.CE_ABS                                                     # BAR 4 absolute
TAIL_H = L2.TAIL_H
MIN_FREE_BYTES = L2.MIN_FREE_BYTES
ADAM_EPS = 1e-8
END_MODEL, END_DIGEST, END_SHA = L2.END_MODEL, L2.END_DIGEST, L2.END_SHA
MID_STEP, MID_MODEL, MID_DIGEST, MID_SHA = L2.MID_STEP, L2.MID_MODEL, L2.MID_DIGEST, L2.MID_SHA
assert CE_ABS == 0.005

STAGE0 = L2.STAGE0
DESK = L2.DESK
FME1 = L2.FME1
FMEL1 = L2.FMEL1
FMEL2 = Path(f"logs/fmel2/smoke{FME.ARENA_TAG}_ladder.json" if SMOKE else "logs/fmel2/ladder.json")
M_TREE = "checkpoints/fmel2_smoke/" if SMOKE else "checkpoints/fmel2/A/e1e-1/"
LOCK = L2.LOCK
OUT_DIR = Path("logs/rdc1")
CK_DIR = Path("checkpoints/rdc1_smoke" if SMOKE else "checkpoints/rdc1")
RECEIPT = OUT_DIR / (f"smoke{SMOKE_TAG}_control.json" if SMOKE else "control.json")
STREAM = OUT_DIR / (f"smoke{SMOKE_TAG}_control.jsonl" if SMOKE else "control.jsonl")
if not SMOKE:
    assert (ANCHOR, LEG_FULL, GRID, OMA.THREADS) == (ARENA["anchor"], ARENA["leg"], ARENA["grid"], ARENA["threads"]), "arena literals"
    assert ANCHOR + LEG_FULL == OMA.TOTAL == 15_420
    assert str(FMEL2) == "logs/fmel2/ladder.json" and str(L2.FMEL1) == "logs/fmel1/ladder.json" and str(L1.FME1) == "logs/fme1/treat.json"
assert GRID[-1] == H_END and TAIL_H in GRID and 1 in GRID
OMA.CK_DIR = CK_DIR                            # this instrument's snapshots never land under checkpoints/oma1, fme1, fmel1 or fmel2


# ---------------------------------------------------------------- the construction (pure numpy on flats; no division by K anywhere)
def draw(seed, d):
    """q ~ N(0, I) in float64 exp_avg coordinates, flat order, from a CPU torch.Generator seeded with the frozen seed."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(int(d), generator=g, dtype=torch.float64).numpy()


def flat_digest(x):
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()


def k_map(opt, model, c, segs, d, step_next, grp, eps=ADAM_EPS):
    """K = -lr beta1 / (bc1 D) elementwise (float64 flat), D = sqrt((beta2 v + (1 - beta2) c^2) / bc2) + eps, from the bound
    optimizer's exp_avg_sq and the clipped first-batch gradient c; plus m = exp_avg (float64 flat). Independent of exp_avg."""
    names = [n for n, _ in model.named_parameters()]
    pidx = {n: i for i, n in enumerate(names)}
    params = opt.param_groups[0]["params"]
    lr, b1, b2 = grp["lr"], grp["beta1"], grp["beta2"]
    bc1, bc2 = 1.0 - b1 ** step_next, 1.0 - b2 ** step_next
    K = np.empty(d, dtype=np.float64); m = np.empty(d, dtype=np.float64); zero_v = np.zeros(d, dtype=bool)
    for k, a, b, _ in segs:
        st = opt.state[params[pidx[k]]]
        v = st["exp_avg_sq"].detach().to("cpu", torch.float64).reshape(-1).numpy()
        cc = c[a:b]
        D = np.sqrt((b2 * v + (1.0 - b2) * cc * cc) / bc2) + eps
        K[a:b] = -lr * b1 / (bc1 * D)
        m[a:b] = st["exp_avg"].detach().to("cpu", torch.float64).reshape(-1).numpy()
        zero_v[a:b] = v == 0.0
    return K, m, zero_v


def k_stats(K, zero_v, segs):
    """Descriptive: per-group write-metric effective dimension n_eff = (sum K^2)^2 / sum K^4, zero-variance coordinate count, |K| range."""
    out = {}
    for grp in sorted(set(s[3] for s in segs)):
        idx = np.concatenate([np.arange(a, b) for k, a, b, g in segs if g == grp])
        k2 = K[idx] ** 2
        s2, s4 = float(k2.sum()), float((k2 * k2).sum())
        out[grp] = {"n": int(idx.size), "n_eff": (s2 * s2 / s4 if s4 > 0 else None), "zero_v": int(zero_v[idx].sum()),
                    "abs_k_max": float(np.abs(K[idx]).max()), "abs_k_min": float(np.abs(K[idx]).min())}
    return out


def construct(q, K, m, segs, m_eps=M_EPS):
    """AMENDMENT L76847 per group: dm_M = -m_eps m, v_M = K dm_M; q_perp = q - (<K q, v_M> / ||v_M||^2) dm_M;
    dm_R = (||v_M|| / ||K q_perp||) q_perp. Returns (dm_R float64 flat, per-group record) or (None, record with the UNDEFINED
    reason) when a group's v_M or K q_perp is zero (matching impossible). No coordinate is ever divided by K."""
    dm_M = -m_eps * m
    v_M = K * dm_M
    dm_R = np.zeros_like(q)
    rec = {"groups": {}, "undefined": {}}
    for grp in sorted(set(s[3] for s in segs)):
        idx = np.concatenate([np.arange(a, b) for k, a, b, g in segs if g == grp])
        qg, Kg, vg, dmg = q[idx], K[idx], v_M[idx], dm_M[idx]
        nv2 = float(vg @ vg)
        if nv2 == 0.0:
            rec["undefined"][grp] = "moment-axis write is zero in this group"
            continue
        Kq = Kg * qg
        coef = float(Kq @ vg) / nv2
        qp = qg - coef * dmg
        Kqp = Kg * qp
        nkq = float(np.linalg.norm(Kqp))
        if nkq == 0.0 or not math.isfinite(nkq):
            rec["undefined"][grp] = "projected random write is zero or non-finite in this group"
            continue
        s = math.sqrt(nv2) / nkq
        dm_R[idx] = s * qp
        rec["groups"][grp] = {"vM_norm": math.sqrt(nv2), "proj_coef": coef, "kq_perp_norm": nkq, "scale": s, "dmR_norm": float(np.linalg.norm(dm_R[idx])),
                              "dmM_norm": float(np.linalg.norm(dmg)), "write_cos_to_M": (cosine(Kg * dm_R[idx], vg))}
    if rec["undefined"]:
        return None, rec
    rec["dmM_norm_total"] = float(np.linalg.norm(dm_M)); rec["vM_norm_total"] = float(np.linalg.norm(v_M))
    rec["dmR_norm_total"] = float(np.linalg.norm(dm_R)); rec["dmR_over_dmM_total"] = rec["dmR_norm_total"] / rec["dmM_norm_total"] if rec["dmM_norm_total"] > 0 else None
    rec["dmR_over_dmM_group"] = {g: (x["dmR_norm"] / x["dmM_norm"] if x["dmM_norm"] > 0 else None) for g, x in rec["groups"].items()}
    rec["analytic_write_cos_to_M_global"] = cosine(K * dm_R, v_M)
    rec["analytic_write_norm"] = float(np.linalg.norm(K * dm_R))
    return dm_R, rec


def dm_quantiles(dm):
    a = np.abs(dm)
    qs = np.quantile(a, [0.5, 0.9, 0.99, 0.999])
    return {"max": float(a.max()), "q50": float(qs[0]), "q90": float(qs[1]), "q99": float(qs[2]), "q999": float(qs[3]), "norm": float(np.linalg.norm(dm))}


def untouched_digest(opt, model, segs):
    """sha256 over exp_avg_sq bytes and the Adam step counters in flatten order plus the group record: the state the intervention
    must leave untouched (exp_avg is excluded on purpose)."""
    names = [n for n, _ in model.named_parameters()]
    pidx = {n: i for i, n in enumerate(names)}
    params = opt.param_groups[0]["params"]
    h = hashlib.sha256()
    for k, a, b, _ in segs:
        st = opt.state[params[pidx[k]]]
        h.update(st["exp_avg_sq"].detach().to("cpu").contiguous().numpy().tobytes())
        h.update(str(float(st["step"])).encode())
    h.update(json.dumps(OMA.group_record(opt), sort_keys=True).encode())
    return h.hexdigest()


def apply_dm(opt, model, dm, segs, d):
    """exp_avg <- float32(float64(exp_avg) + dm) on every tensor (dm None = control, nothing touched). Returns (tensors touched,
    realized float64 delta flat = new - old)."""
    if dm is None:
        return 0, None
    names = [n for n, _ in model.named_parameters()]
    pidx = {n: i for i, n in enumerate(names)}
    params = opt.param_groups[0]["params"]
    delta = np.empty(d, dtype=np.float64)
    n = 0
    with torch.no_grad():
        for k, a, b, _ in segs:
            st = opt.state[params[pidx[k]]]
            old = st["exp_avg"]
            new = (old.to(torch.float64) + torch.from_numpy(dm[a:b]).reshape(old.shape)).to(old.dtype)
            delta[a:b] = (new.to(torch.float64) - old.to(torch.float64)).reshape(-1).numpy()
            old.copy_(new)
            n += 1
    return n, delta


# ---------------------------------------------------------------- pure laws
def preflight_law(dev, devM, shares, sharesM):
    """BAR 1 on the realized one-step deviations: dev = {arm: flat}, devM the moment-axis flat, shares / sharesM the group shares.
    (a) magnitude to MAG_TOL, (b) |cos| to M <= COS_M_MAX, (c) max share gap <= SHARE_TOL, (d) pairwise |cos| <= COS_PAIR_MAX.
    Returns (ok, detail); UNDEFINED values fail."""
    nM = float(np.linalg.norm(devM))
    detail = {"target_norm": nM, "target_shares": sharesM, "magnitude": {}, "cos_to_M": {}, "share_gap": {}, "pair_cos": {}, "fails": []}
    if not (finite(nM) and nM > 0) or sharesM is None:
        detail["fails"].append("moment-axis h = 1 deviation is zero or non-finite")
        return False, detail
    arms = list(dev)
    for a in arms:
        n = float(np.linalg.norm(dev[a]))
        ratio = n / nM if finite(n) else None
        detail["magnitude"][a] = ratio
        if ratio is None or abs(ratio - 1.0) > MAG_TOL:
            detail["fails"].append(f"magnitude {a}: ||dW_r(1)|| / ||dW_M(1)|| = {ratio} (tol {MAG_TOL})")
        c = cosine(dev[a], devM)
        detail["cos_to_M"][a] = c
        if c is None or abs(c) > COS_M_MAX:
            detail["fails"].append(f"orthogonality {a}: cos to M {c} (|cos| <= {COS_M_MAX})")
        gap = None if shares.get(a) is None else max(abs(shares[a][g] - sharesM[g]) for g in sharesM)
        detail["share_gap"][a] = gap
        if gap is None or gap > SHARE_TOL:
            detail["fails"].append(f"locus {a}: max group share gap {gap} (tol {SHARE_TOL})")
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            c = cosine(dev[arms[i]], dev[arms[j]])
            detail["pair_cos"][f"{arms[i]}|{arms[j]}"] = c
            if c is None or abs(c) > COS_PAIR_MAX:
                detail["fails"].append(f"independence {arms[i]}|{arms[j]}: cos {c} (|cos| <= {COS_PAIR_MAX})")
    return not detail["fails"], detail


def horizon_metrics(W0, WC, W, WM):
    """Per-horizon readouts from flats: W0 anchor, WC fresh control, W = {R arm: flat}, WM the locked moment-axis snapshot.
    Zero-norm / non-finite law: UNDEFINED = None + reason, never NaN. G and A need the h = 1 norms and are filled by the caller."""
    arms = list(W)
    dC = WC - W0
    legn = float(np.linalg.norm(dC))
    dev = {a: W[a] - WC for a in arms}
    devM = WM - WC
    absn = {a: float(np.linalg.norm(dev[a])) for a in arms}
    absM = float(np.linalg.norm(devM))
    out = {"leg_abs": (legn if math.isfinite(legn) else None), "abs": {a: (absn[a] if math.isfinite(absn[a]) else None) for a in arms}, "abs_M": (absM if math.isfinite(absM) else None),
           "n": {a: None for a in arms}, "n_M": None, "cos_M": {a: None for a in arms}, "pair_cos": {}, "group_share": {a: None for a in arms}, "group_share_M": None,
           "leg_cos": {a: None for a in arms}, "G": {a: None for a in arms}, "G_M": None, "A": {a: None for a in arms}, "undefined": {}}
    nonfinite = [a for a in arms if not math.isfinite(absn[a])] + (["control"] if not math.isfinite(legn) else []) + (["M"] if not math.isfinite(absM) else [])
    if nonfinite:
        out["undefined"]["nonfinite"] = f"non-finite deviation norm in {nonfinite}"
        return out
    for a in arms:
        if legn > 0:
            out["n"][a] = absn[a] / legn
        else:
            out["undefined"][f"n:{a}"] = "control leg displacement is zero"
        out["cos_M"][a] = cosine(dev[a], devM)
        if out["cos_M"][a] is None:
            out["undefined"][f"cos_M:{a}"] = "a deviation is zero"
        out["leg_cos"][a] = cosine(W[a] - W0, dC)
    out["n_M"] = absM / legn if legn > 0 else None
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            out["pair_cos"][f"{arms[i]}|{arms[j]}"] = cosine(dev[arms[i]], dev[arms[j]])
    return out


def growth(metrics_by_h):
    """Fill G_r(h) = ||dW_r(h)|| / ||dW_r(1)||, G_M(h) and A_r(h) = G_r / G_M in place; UNDEFINED when a h = 1 norm is zero / missing."""
    m1 = metrics_by_h[1]
    for h, m in metrics_by_h.items():
        if finite(m["abs_M"]) and finite(m1["abs_M"]) and m1["abs_M"] > 0:
            m["G_M"] = m["abs_M"] / m1["abs_M"]
        else:
            m["G_M"] = None; m["undefined"]["G_M"] = "moment-axis h = 1 deviation norm is zero or UNDEFINED"
        for a in m["abs"]:
            if finite(m["abs"][a]) and finite(m1["abs"][a]) and m1["abs"][a] > 0:
                m["G"][a] = m["abs"][a] / m1["abs"][a]
            else:
                m["G"][a] = None; m["undefined"][f"G:{a}"] = "h = 1 deviation norm is zero or UNDEFINED"
            if m["G"][a] is not None and m["G_M"] is not None and m["G_M"] > 0:
                m["A"][a] = m["G"][a] / m["G_M"]
            else:
                m["A"][a] = None; m["undefined"][f"A:{a}"] = "G_r or G_M UNDEFINED (or G_M zero)"
    return metrics_by_h


def regime(A_H, preflight_ok):
    """BAR 2 at H, first match wins: REGIME-UNRESOLVED / DIRECTION-GENERIC / MOMENT-SPECIFIC / RANDOM-DOMINANT / MIXED."""
    vals = [A_H.get(a) for a in R_ARMS]
    if not preflight_ok or any(not finite(v) for v in vals):
        return "REGIME-UNRESOLVED"
    if all(GENERIC_BAND[0] <= v <= GENERIC_BAND[1] for v in vals):
        return "DIRECTION-GENERIC"
    if all(v <= SPECIFIC_MAX for v in vals):
        return "MOMENT-SPECIFIC"
    if all(v >= DOMINANT_MIN for v in vals):
        return "RANDOM-DOMINANT"
    return "MIXED"


def direction(cosM_H):
    """BAR 3 (reported): DIRECTION-SHARED-LATE / DIRECTION-INDEPENDENT-LATE / MIXED-DIRECTION; UNDEFINED -> DIRECTION-UNRESOLVED."""
    vals = [cosM_H.get(a) for a in R_ARMS]
    if any(not finite(v) for v in vals):
        return "DIRECTION-UNRESOLVED"
    if all(abs(v) >= SHARED_LATE for v in vals):
        return "DIRECTION-SHARED-LATE"
    if all(abs(v) <= INDEP_LATE for v in vals):
        return "DIRECTION-INDEPENDENT-LATE"
    return "MIXED-DIRECTION"


def h_amp(G_by_h):
    """First grid h with G >= H_AMP_G over {h: G} (UNDEFINED horizons skipped); None = not reached."""
    for h in sorted(G_by_h):
        g = G_by_h[h]
        if finite(g) and g >= H_AMP_G:
            return h
    return None


def label(reg, dirlab, func):
    return f"{reg}+{dirlab}+{func} [writer A only, one anchor, one seed lineage, CPU deterministic, write norm = FMEL2 eps 1e-1, three random directions; gate descriptive]"


def preflight_disposition(pf_ok, bypass, c_digest_match):
    """Early returns after the one-step preflight: a fresh C h = 1 digest differing from the locked FMEL2 C digest is a substrate
    change (NOT-RUN, judged first: the targets are void without it); a failed construction preflight books CONSTRUCTION-UNRESOLVED
    (unless the SMOKE-only bypass is set); None proceeds to the long legs."""
    if not c_digest_match:
        return "NOT-RUN", "NOT-RUN", "NOT-RUN (fresh C h = 1 digest differs from the locked FMEL2 C h = 1 digest: substrate change)"
    if not pf_ok and not bypass:
        return "PREFLIGHT-FAILED", "CONSTRUCTION-UNRESOLVED", label("CONSTRUCTION-UNRESOLVED", "DIRECTION-NOT-MEASURED", "FUNCTION-NOT-MEASURED")
    return None


def reference_digests(fmel2, horizons=None):
    """BAR 0 references from the LOCKED FMEL2 receipt only: {"C": {h: state digest}} at every grid horizon, and the M snapshot
    records {h: {path, sha256, state_digest}}; every path must lie under the comparison-vector tree. No checkpoint is opened."""
    horizons = GRID if horizons is None else horizons
    C = fmel2["arms"]["C"]["snapshots"]; M = fmel2["arms"][M_ARM]["snapshots"]
    refs = {"C": {h: C[str(h)]["state_digest"] for h in horizons}}
    mrefs = {}
    for h in horizons:
        e = M[str(h)]
        assert e["path"].startswith(M_TREE) and e["path"].endswith(f"h{h:04d}.pt"), e["path"]
        assert len(e["sha256"]) == 64 and len(e["state_digest"]) == 64
        mrefs[h] = {"path": e["path"], "sha256": e["sha256"], "state_digest": e["state_digest"]}
    assert refs["C"][1] == fmel2["preflight"]["arms"]["C"]["digest"], "FMEL2 C h = 1 leg digest v its preflight digest"
    assert mrefs[1]["state_digest"] == fmel2["preflight"]["arms"][M_ARM]["digest"], "FMEL2 e1e-1 h = 1 leg digest v its preflight digest"
    return refs, mrefs


def verify_m_files(mrefs):
    """File sha of every comparison-vector snapshot against the locked receipt (a pure read; digests are asserted when loaded)."""
    out = {}
    for h, e in mrefs.items():
        p = Path(e["path"])
        if not p.exists():
            out[str(h)] = {"path": e["path"], "status": "MISSING"}
            continue
        s = sha256_file(p)
        out[str(h)] = {"path": e["path"], "file_sha256": s, "receipt_sha256": e["sha256"], "status": "OK" if s == e["sha256"] else "DRIFTED"}
    return out


def load_m(mrefs, h):
    sd = load_model_sd(mrefs[h]["path"], step=ANCHOR + h)
    dg = state_digest(sd)
    if dg != mrefs[h]["state_digest"]:
        raise Abort(f"comparison vector M h = {h}: state digest {dg[:16]} v locked {mrefs[h]['state_digest'][:16]}")
    return sd


def assert_fmel2_provenance(fmel2, lock, stage0, fme1, fmel1, horizons=None):
    """The FMEL2 receipt this rung pins: sha equal to the source literal and the lock; bound to the same Stage-0 / FME1 / FMEL1
    receipts and anchor; status DONE, real mode, writer A; C and e1e-1 snapshots at every grid horizon with sha and digest."""
    horizons = GRID if horizons is None else horizons
    s = sha256_file(FMEL2)
    assert s == FMEL2_SHA == FME.locked_sha(lock, str(FMEL2)), "FMEL2 receipt v the source literal / the receipt lock"
    assert fmel2["pins"]["stage0_sha256"] == FME.STAGE0_SHA == sha256_file(STAGE0) and fmel2["pins"]["fme1_sha256"] == L1.FME1_SHA == sha256_file(FME1)
    assert fmel2["pins"]["fmel1_sha256"] == L2.FMEL1_SHA == sha256_file(FMEL1)
    assert fmel2["anchor"]["state_digest"] == stage0["cells"][WRITER]["C"]["bind"]["state_digest"] == fme1["anchor"]["state_digest"] == fmel1["anchor"]["state_digest"]
    assert fmel2["status"] == "DONE" and fmel2["smoke"] is False and fmel2["writer"] == WRITER and fmel2["n_pred_literal"] == N_PRED
    assert fmel2["preflight"]["ok"] is True and fmel2["adjudication"]["bar1_preflight"] is True
    for arm in ("C", M_ARM):
        assert all(str(h) in fmel2["arms"][arm]["snapshots"] for h in horizons), arm
    assert fmel2["arms"][M_ARM]["eps"] == M_EPS and fmel2["arms"]["C"]["eps"] == 0.0
    return {"fmel2_sha256": s, "fmel2_commit": fmel2["commit"], "fmel2_regime": fmel2["regime"], "fmel2_label": fmel2["label"]}


# ---------------------------------------------------------------- wall cap (L2's mechanism, this rung's names)
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
    print(f"[rdc1] phase {name} at {time.time() - PHASE.get('t0', time.time()):.1f} s", flush=True)


def wall_label():
    return f"NOT-RUN (registered wall limit {MAX_WALL_S} s reached during {PHASE['name']}; stop law (a), PRE-REG L76553)"


def wall_record(hard_exit=False):
    return {"limit_s": MAX_WALL_S, "real_limit_s": MAX_WALL_S_REAL, "phase": PHASE["name"], "reason": "registered whole-run wall cap", "hard_exit": hard_exit,
            "elapsed_s": round(time.time() - PHASE.get("t0", time.time()), 1)}


# ---------------------------------------------------------------- legs (L1's mechanics with the intervention call replaced)
def desk_bind(tok, enc, slices, segs, d):
    """One bind that is never stepped: the clipped first-batch gradient c, the K map, m, the untouched-state digest and the group
    for step ANCHOR + 1. The optimizer state is asserted unchanged afterwards."""
    model, opt, _ck, binfo = OMA.bind(WRITER, tok, "cpu")
    _sched, grp = OMA.resume_sched(OMA.WRITERS[WRITER]["kind"], opt, ANCHOR, binfo["serialized"])
    a, b = slices[0]
    c, gnorm, loss = OMA.clipped_grad(model, tok, enc[a:b], "cpu", segs, d)
    K, m, zero_v = k_map(opt, model, c, segs, d, ANCHOR + 1, grp)
    law = OMA.bar1_law(model, opt, c, segs, ANCHOR + 1, grp)
    assert OG.opt_state_digest(opt) == binfo["opt_state_digest_bound"], "optimizer state changed during the desk bind (no step may occur)"
    untouched = untouched_digest(opt, model, segs)
    return {"c": c, "K": K, "m": m, "zero_v": zero_v, "grp": grp, "grad_norm": gnorm, "batch_loss": loss, "bind": binfo, "untouched_digest": untouched, "bar1_law": law,
            "W0": OMA.flat(OMA.sd_cpu(model), segs, d)}


def one_step_r(dm, tok, enc, slices, segs, d, untouched_ref):
    """Preflight: fresh bind, intervention, exactly one step on the first future slice. Returns (flat, digest, binfo, touched, group,
    realized delta, untouched-state digest match)."""
    model, opt, _ck, binfo = OMA.bind(WRITER, tok, "cpu")
    _sched, grp = OMA.resume_sched(OMA.WRITERS[WRITER]["kind"], opt, ANCHOR, binfo["serialized"])
    touched, delta = apply_dm(opt, model, dm, segs, d)
    untouched_ok = untouched_digest(opt, model, segs) == untouched_ref
    snaps, _losses, _ = OMA.run_leg(model, opt, _sched, tok, enc, slices[:1], "cpu", [1])
    sd = snaps[1]
    return OMA.flat(sd, segs, d), state_digest(sd), binfo, touched, grp, delta, untouched_ok


def long_leg_r(arm, dm, tok, enc, slices, segs, d, held, refs, preflight_digest, stream, cell, mid):
    """L1.long_leg with apply_eps(opt, eps) replaced by apply_dm(opt, model, dm, segs, d) and the C qualification at every grid
    horizon (source identity against L1 guarded by the test). One fresh continuous leg with in-line BAR 0 and the preflight-digest
    assertion; snapshots at GRID (opt at H); HELD-32 CE per grid horizon; for C the in-line descriptive substrate readout at
    MID_STEP. Raises Abort (after filling `cell`) on any digest mismatch."""
    model, opt, _ck, binfo = OMA.bind(WRITER, tok, "cpu")
    sched, grp = OMA.resume_sched(OMA.WRITERS[WRITER]["kind"], opt, ANCHOR, binfo["serialized"])
    touched, delta = apply_dm(opt, model, dm, segs, d)
    assert touched == (0 if dm is None else len(UG.KEYS)), (arm, touched)
    cell.update({"arm": arm, "seed": SEEDS.get(arm), "delta_digest": (None if delta is None else flat_digest(delta)), "bind": binfo, "group_next": grp, "tensors_touched": touched, "qualification": {}, "sched_at_grid": {}})
    check_at = set(GRID) | {1} | ({mid["h"]} if arm == "C" and mid else set())
    grid_set = set(GRID)

    def on_step(i, loss):
        stream({"writer": WRITER, "arm": arm, "seed": SEEDS.get(arm), "device": "cpu", "step": ANCHOR + i, "loss": loss})
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
                dmid = Wm - mid["W"]
                cell["substrate_mid"] = {"step": ANCHOR + i, "booked": str(MID_MODEL), "booked_digest": mid["digest"], "booked_sha256": mid["sha256"],
                                         "fresh_digest": dg, "rho": float(np.linalg.norm(dmid) / max(np.linalg.norm(mid["W"] - mid["W0"]), 1e-300)),
                                         "abs": float(np.linalg.norm(dmid)), "ce_fresh": OMA.held_ce(model, tok, held, "cpu")[0], "ce_booked": mid["ce"]}

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
def mode_control(tok, enc, starts, info, segs, d, held, rec, stream):
    assert OMA.CK_DIR == CK_DIR, (OMA.CK_DIR, CK_DIR)
    set_phase("provenance")
    stage0 = json.loads(STAGE0.read_text()); desk = json.loads(DESK.read_text()); fme1 = json.loads(FME1.read_text())
    fmel1 = json.loads(FMEL1.read_text()); fmel2 = json.loads(FMEL2.read_text())
    rec["pins"] = {"sources": dict(PINS), "leg_path_symbols": FME.LEG_PATH_SYMBOLS, "leg_path_sha_measured": FME.leg_path_sha(), "leg_path_sha": FME.LEG_PATH_SHA,
                   "stage0_receipt": str(STAGE0), "stage0_sha256": sha256_file(STAGE0), "desk_receipt": str(DESK), "desk_sha256": sha256_file(DESK),
                   "fme1_receipt": str(FME1), "fme1_sha256": sha256_file(FME1), "fmel1_receipt": str(FMEL1), "fmel1_sha256": sha256_file(FMEL1),
                   "fmel2_receipt": str(FMEL2), "fmel2_sha256": sha256_file(FMEL2), "data_files": FME.data_file_shas()}
    if not SMOKE:
        lock = json.loads(LOCK.read_text())
        rec["pins"].update(FME.assert_provenance(stage0, desk, lock, sha256_file(OMA.WRITERS[WRITER]["anchor"]), torch.get_num_threads(), torch.__version__, np.__version__))
        rec["pins"].update(L1.assert_fme1_provenance(fme1, lock, stage0))
        rec["pins"].update(L2.assert_fmel1_provenance(fmel1, lock, stage0, fme1))
        rec["pins"].update(assert_fmel2_provenance(fmel2, lock, stage0, fme1, fmel1))
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
    assert rec["leg_slices_digest_full"] == fmel2["leg_slices_digest_full"], "future stream v the FMEL2 full-leg digest"
    if not SMOKE:
        assert rec["leg_slices_digest_full"] == fmel1["leg_slices_digest_full"], "future stream v the FMEL1 full-leg digest"
    rec["leg"] = {"first_step": ANCHOR + 1, "last_step": ANCHOR + LEG_FULL, "n_steps": LEG_FULL, "first_slice": list(slices[0]), "last_slice": list(slices[-1]),
                  "epoch_position_first": OMA.epoch_position(ANCHOR + 1, info["n_enc"]), "epoch_position_last": OMA.epoch_position(ANCHOR + LEG_FULL, info["n_enc"])}
    ok_disk, free = disk_preflight()
    rec["disk_preflight"] = {"free_bytes": free, "min_free_bytes": MIN_FREE_BYTES, "ok": ok_disk}
    if not ok_disk:
        raise SystemExit(f"DISK PREFLIGHT REFUSED: {free / (1 << 30):.1f} GiB free < {MIN_FREE_BYTES / (1 << 30):.0f} GiB")
    try:
        refs, mrefs = reference_digests(fmel2)
    except AssertionError as e:
        rec["status"] = "NOT-RUN"; rec["regime"] = "NOT-RUN"; rec["label"] = f"NOT-RUN (locked receipt disagrees: {e})"
        return rec
    if SMOKE and os.environ.get("SMOKE_TAMPER_REF") == "1":
        refs["C"][GRID[1]] = "0" * 64                            # abort smoke: a corrupted C reference digest must book NOT-RUN in the C leg
        rec["smoke_tampered_ref"] = f"C@{GRID[1]}"
    if SMOKE and os.environ.get("SMOKE_TAMPER_M") == "1":
        mrefs[1]["sha256"] = "0" * 64                            # comparison-vector smoke: a drifted M file must refuse before any state exists
        rec["smoke_tampered_ref"] = "M@1 (file sha)"
    rec["reference_digests"] = {a: {str(h): v for h, v in hs.items()} for a, hs in refs.items()}
    rec["comparison_vectors"] = {str(h): dict(v) for h, v in mrefs.items()}
    rec["comparison_vector_files"] = verify_m_files(mrefs)
    if any(v["status"] != "OK" for v in rec["comparison_vector_files"].values()):
        rec["status"] = "NOT-RUN"; rec["regime"] = "NOT-RUN"
        rec["label"] = "NOT-RUN (comparison-vector snapshot missing or drifted: " + json.dumps({h: v["status"] for h, v in rec["comparison_vector_files"].items() if v["status"] != "OK"}) + ")"
        return rec
    rec["locked_M"] = {"abs": {str(h): fmel2["metrics"][str(h)]["abs"][M_ARM] for h in GRID}, "n": {str(h): fmel2["metrics"][str(h)]["n"][M_ARM] for h in GRID},
                       "rotation": {str(h): fmel2["metrics"][str(h)]["rotation"][M_ARM] for h in GRID}, "group_share_1": fmel2["metrics"]["1"]["group_share"][M_ARM],
                       "ce_held": {str(h): fmel2["arms"][M_ARM]["ce_held"][str(h)] for h in GRID}, "dce": {str(h): fmel2["dce"][str(h)][M_ARM] for h in GRID}}
    set_phase("anchor-and-substrate-load")
    anchor_sd = load_model_sd(OMA.WRITERS[WRITER]["anchor"])
    rec["anchor"] = {"path": OMA.WRITERS[WRITER]["anchor"], "file_sha256": sha256_file(OMA.WRITERS[WRITER]["anchor"]), "state_digest": state_digest(anchor_sd)}
    if not SMOKE:
        assert rec["anchor"]["state_digest"] == stage0["cells"][WRITER]["C"]["bind"]["state_digest"] == fmel2["anchor"]["state_digest"]
    W0 = OMA.flat(anchor_sd, segs, d)
    end_sd = load_model_sd(END_MODEL); mid_sd = load_model_sd(MID_MODEL, step=MID_STEP)
    end_dg, mid_dg = state_digest(end_sd), state_digest(mid_sd)
    if END_DIGEST:
        assert end_dg.startswith(END_DIGEST) and mid_dg.startswith(MID_DIGEST), (end_dg[:16], mid_dg[:16])
        assert sha256_file(END_MODEL).startswith(END_SHA) and sha256_file(MID_MODEL).startswith(MID_SHA), "booked substrate files v the pre-reg shas"
    mC = UG.build(tok, "cpu")
    mC.load_state_dict(mid_sd); mid_ce = OMA.held_ce(mC, tok, held, "cpu")[0]
    mC.load_state_dict(end_sd); end_ce = OMA.held_ce(mC, tok, held, "cpu")[0]
    del mC
    mid = {"h": MID_STEP - ANCHOR, "W": OMA.flat(mid_sd, segs, d), "W0": W0, "digest": mid_dg, "sha256": sha256_file(MID_MODEL), "ce": mid_ce}
    assert 1 <= mid["h"] <= LEG_FULL
    # ---- CONSTRUCTION (one desk bind, never stepped; the three shaped draws)
    set_phase("construction")
    t0 = time.time()
    dk = desk_bind(tok, enc, slices, segs, d)
    assert np.array_equal(dk["W0"], W0), "desk bind weights v the anchor flat"
    K, m = dk["K"], dk["m"]
    assert np.all(np.isfinite(K)) and np.all(K != 0.0) and np.all(np.isfinite(m)), "K must be finite and non-zero everywhere (D >= eps_adam)"
    con = {"group_next": dk["grp"], "step_next": ANCHOR + 1, "grad_norm": dk["grad_norm"], "clip_coef": OG.clip_coef(dk["grad_norm"]), "batch_loss": dk["batch_loss"],
           "untouched_digest": dk["untouched_digest"], "k_stats": k_stats(K, dk["zero_v"], segs), "zero_v_total": int(dk["zero_v"].sum()),
           "m_norm": float(np.linalg.norm(m)), "K_digest": flat_digest(K), "m_digest": flat_digest(m), "arms": {}}
    # v_M = K (-0.1 m) = -0.1 carry: the desk receipt's carry norm is an independent check of the K map on this anchor
    vM_norm = float(np.linalg.norm(K * (-M_EPS * m)))
    carry = float(desk["cells"][WRITER]["law"]["norms"]["carry"])
    con["vM_norm_analytic"] = vM_norm; con["desk_carry_norm"] = carry; con["vM_v_desk_carry_rel"] = abs(vM_norm - M_EPS * carry) / (M_EPS * carry)
    assert con["vM_v_desk_carry_rel"] <= 1e-6, ("K map v the locked desk carry norm", con["vM_v_desk_carry_rel"])
    assert abs(dk["bar1_law"]["n_pred"] - n_pred) <= 1e-9, ("desk bind n_pred v the sealed value", dk["bar1_law"]["n_pred"], n_pred)
    dms = {}
    for arm in R_ARMS:
        q = draw(SEEDS[arm], d)
        dm, crec = construct(q, K, m, segs)
        if SMOKE and dm is not None and os.environ.get("SMOKE_TAMPER_CONSTRUCTION") == "1":
            dm = dm + 0.3 * (-M_EPS * m)                     # SMOKE only: 0.3 dm_M added AFTER the projection, so BAR 1 (a) and (b) must refuse before any leg
            crec["smoke_tampered"] = "0.3 dm_M added after the projection"
            rec["smoke_tampered_construction"] = "0.3 dm_M added after the projection on every random arm"
        if dm is None:
            rec["status"] = "PREFLIGHT-FAILED"; rec["regime"] = "CONSTRUCTION-UNRESOLVED"
            rec["label"] = label("CONSTRUCTION-UNRESOLVED", "DIRECTION-NOT-MEASURED", "FUNCTION-NOT-MEASURED")
            rec["construction"] = dict(con, failure={arm: crec["undefined"]}, wall_s=round(time.time() - t0, 1))
            return rec
        dms[arm] = dm
        con["arms"][arm] = {"seed": SEEDS[arm], "q_digest": flat_digest(q), "dm_digest": flat_digest(dm), "construction": crec, "dm_abs": dm_quantiles(dm)}
        del q
    con["wall_s"] = round(time.time() - t0, 1)
    rec["construction"] = con
    print(f"[rdc1] construction: ||v_M|| {vM_norm:.6e} (desk carry x 0.1 rel {con['vM_v_desk_carry_rel']:.1e}); ||dm_R|| / ||dm_M|| "
          f"{ {a: round(con['arms'][a]['construction']['dmR_over_dmM_total'], 4) for a in R_ARMS} }; zero-v coords {con['zero_v_total']}", flush=True)
    # ---- ONE-STEP PREFLIGHT (no long leg before it passes)
    set_phase("preflight")
    t0 = time.time()
    pf = {"arms": {}}
    W1, dev1, realized = {}, {}, {}
    for arm in ARM_ORDER:
        f, dg, binfo, touched, grp, delta, unt_ok = one_step_r(dms.get(arm), tok, enc, slices, segs, d, dk["untouched_digest"])
        pf["arms"][arm] = {"digest": dg, "tensors_touched": touched, "group_next": grp, "untouched_state_match": unt_ok, "seed": SEEDS.get(arm),
                           "delta_digest": (None if delta is None else flat_digest(delta))}
        assert touched == (0 if arm == "C" else len(UG.KEYS)), (arm, touched)
        assert grp == dk["grp"], (arm, grp, dk["grp"])
        W1[arm] = f
        if delta is not None:
            realized[arm] = delta
    pf["arms"]["C"]["fmel2_digest"] = refs["C"][1]
    pf["arms"]["C"]["digest_match"] = pf["arms"]["C"]["digest"] == refs["C"][1]
    c_match = pf["arms"]["C"]["digest_match"]
    if c_match:
        WM1 = OMA.flat(load_m(mrefs, 1), segs, d)
        devM = WM1 - W1["C"]
        vM = K * (-M_EPS * m)
        for arm in R_ARMS:
            dev1[arm] = W1[arm] - W1["C"]
        sharesM = group_shares(devM, segs)
        shares = {a: group_shares(dev1[a], segs) for a in R_ARMS}
        pf["ok"], pf["detail"] = preflight_law(dev1, devM, shares, sharesM)
        pf["detail"]["untouched_state"] = {a: pf["arms"][a]["untouched_state_match"] for a in ARM_ORDER}
        if not all(pf["detail"]["untouched_state"].values()):
            pf["ok"] = False; pf["detail"]["fails"].append("untouched state: " + json.dumps(pf["detail"]["untouched_state"]))
        pf["targets"] = {"norm_M_realized": float(np.linalg.norm(devM)), "norm_M_display_literal": 0.012408559997086078,   # logs/fmel2/ladder.json metrics.1.abs.e1e-1, recorded for display only (never compared) "shares_M_realized": sharesM,
                         "norm_M_analytic": float(np.linalg.norm(vM)), "M_analytic_v_realized_rel": float(np.linalg.norm(devM - vM) / max(np.linalg.norm(vM), 1e-300)),
                         "M_realized_v_locked_abs1_rel": abs(float(np.linalg.norm(devM)) - rec["locked_M"]["abs"]["1"]) / rec["locked_M"]["abs"]["1"]}
        pf["analytic_v_realized"] = {}
        for arm in R_ARMS:
            kd_int = K * dms[arm]; kd_real = K * realized[arm]
            pf["analytic_v_realized"][arm] = {"rel_intended": float(np.linalg.norm(dev1[arm] - kd_int) / max(np.linalg.norm(kd_int), 1e-300)),
                                              "rel_realized_delta": float(np.linalg.norm(dev1[arm] - kd_real) / max(np.linalg.norm(kd_real), 1e-300)),
                                              "delta_v_intended_rel": float(np.linalg.norm(realized[arm] - dms[arm]) / max(np.linalg.norm(dms[arm]), 1e-300)),
                                              "realized_norm": float(np.linalg.norm(dev1[arm])), "analytic_norm_intended": float(np.linalg.norm(kd_int)),
                                              "cos_analytic_realized": cosine(dev1[arm], kd_int)}
        pf["shares"] = shares
        pf["metrics"] = {"n_1": {a: (float(np.linalg.norm(dev1[a])) / float(np.linalg.norm(W1["C"] - W0))) for a in R_ARMS}, "n_1_M": float(np.linalg.norm(devM)) / float(np.linalg.norm(W1["C"] - W0)),
                         "leg_abs_1": float(np.linalg.norm(W1["C"] - W0))}
        del WM1, devM, vM
        avr = ", ".join(f"{a} {v['rel_realized_delta']:.1e}" for a, v in pf["analytic_v_realized"].items())
        print(f"[rdc1] preflight {'PASS' if pf['ok'] else 'FAIL'}: magnitude {pf['detail']['magnitude']} cos_M {pf['detail']['cos_to_M']} share_gap {pf['detail']['share_gap']} "
              f"pair {pf['detail']['pair_cos']}; analytic v realized {{ {avr} }}", flush=True)
    else:
        pf["ok"], pf["detail"] = False, {"fails": ["fresh C h = 1 digest differs from the locked FMEL2 C h = 1 digest"]}
        print(f"[rdc1] preflight: fresh C h = 1 digest {pf['arms']['C']['digest'][:16]} v locked {refs['C'][1][:16]} (substrate change)", flush=True)
    pf["wall_s"] = round(time.time() - t0, 1)
    rec["preflight"] = pf
    del W1, dev1, realized, dk, K, m                 # the desk flats (about 0.6 GB) are not needed by the legs; dms carries the interventions
    rec["smoke_preflight_bypassed"] = bool(PREFLIGHT_BYPASS and not pf["ok"] and c_match)
    assert not (rec["smoke_preflight_bypassed"] and not SMOKE)
    early = preflight_disposition(pf["ok"], PREFLIGHT_BYPASS, c_match)
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
            snaps = long_leg_r(arm, dms.get(arm), tok, enc, slices, segs, d, held, refs, pf["arms"][arm]["digest"], stream, cell, mid)
        except Abort:
            rec["status"] = "NOT-RUN"; rec["regime"] = "NOT-RUN"; rec["label"] = f"NOT-RUN ({cell.get('abort')})"
            print(f"[rdc1] ABORT {arm}: {cell.get('abort')}", flush=True)
            return rec
        if arm != "C":
            assert cell["delta_digest"] == pf["arms"][arm]["delta_digest"], (arm, "realized exp_avg delta differs between preflight and leg")
        del snaps
        print(f"[rdc1] {arm} leg done {cell['wall_s']} s ({cell['it_per_s']} it/s) loss_last {cell['loss_last']:.4f}", flush=True)
    del dms
    # ---- readouts per grid horizon from the fresh snapshots and the locked comparison vectors
    set_phase("readouts")
    metrics, dce, ce = {}, {}, {}
    prev_dev, prev_devM = {}, None
    W_end_C = None
    try:
        for h in GRID:
            WC = OMA.flat(load_model_sd(rec["arms"]["C"]["snapshots"][str(h)]["path"]), segs, d)
            W = {a: OMA.flat(load_model_sd(rec["arms"][a]["snapshots"][str(h)]["path"]), segs, d) for a in R_ARMS}
            WM = OMA.flat(load_m(mrefs, h), segs, d)
            mtr = horizon_metrics(W0, WC, W, WM)
            mtr["rotation"] = {}
            for a in W:
                dev = W[a] - WC
                mtr["group_share"][a] = group_shares(dev, segs)
                mtr["rotation"][a] = cosine(dev, prev_dev[a]) if a in prev_dev else None
                prev_dev[a] = dev
            devM = WM - WC
            mtr["group_share_M"] = group_shares(devM, segs)
            mtr["rotation_M"] = cosine(devM, prev_devM) if prev_devM is not None else None
            mtr["rotation_M_locked"] = rec["locked_M"]["rotation"][str(h)]
            prev_devM = devM
            mtr["abs_M_v_locked_rel"] = (abs(mtr["abs_M"] - rec["locked_M"]["abs"][str(h)]) / rec["locked_M"]["abs"][str(h)]) if finite(mtr["abs_M"]) else None
            mtr["sched"] = rec["arms"]["C"]["sched_at_grid"].get(str(h))
            ce[str(h)] = {a: rec["arms"][a]["ce_held"][str(h)] for a in ARM_ORDER}
            ce[str(h)]["M_locked"] = rec["locked_M"]["ce_held"][str(h)]
            dce[str(h)] = {a: rec["arms"][a]["ce_held"][str(h)] - rec["arms"]["C"]["ce_held"][str(h)] for a in R_ARMS}
            metrics[h] = mtr
            if h == H_END:
                W_end_C = WC
    except Abort as e:
        rec["status"] = "NOT-RUN"; rec["regime"] = "NOT-RUN"; rec["label"] = f"NOT-RUN ({e})"
        return rec
    growth(metrics)
    rec["metrics"] = {str(h): mm for h, mm in metrics.items()}
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
    for name in ARM_ORDER:
        sd = load_model_sd(rec["arms"][name]["snapshots"][str(H_END)]["path"])
        gm = UG.build(tok, dev_gate); gm.load_state_dict(sd); gm.eval()
        solves, valid = gate_eval(gm, tok, dev_gate, n=(2 if SMOKE else None))
        gate[name] = {"solves": solves, "total": int(sum(solves.values())), "valid_pct": round(valid, 2)}
    rec["gate"] = gate
    # ---- adjudication
    set_phase("adjudication")
    mH = metrics[H_END]
    reg = regime(mH["A"], pf["ok"])
    dirlab = direction(mH["cos_M"])
    per, func = function_bar(dce[str(H_END)])
    tail = {a: (mH["abs"][a] / metrics[TAIL_H]["abs"][a] if (finite(mH["abs"][a]) and (metrics[TAIL_H]["abs"][a] or 0) > 0) else None) for a in R_ARMS}
    tail["M"] = (mH["abs_M"] / metrics[TAIL_H]["abs_M"] if (finite(mH["abs_M"]) and (metrics[TAIL_H]["abs_M"] or 0) > 0) else None)
    over = [(a, str(h)) for h in GRID for a in dce[str(h)] if abs(dce[str(h)][a]) > CE_ABS]
    assert set(rec["arms"]["C"]["qualification"]) == {str(h) for h in GRID}, list(rec["arms"]["C"]["qualification"])      # BAR 0 never passes vacuously
    for a in ARM_ORDER:
        assert rec["arms"][a].get("preflight_digest_match") is True, a                                                    # every arm was checked at h = 1
    rec["adjudication"] = {"bar0": {"C_all_horizons": all(q["ok"] for q in rec["arms"]["C"]["qualification"].values()), "preflight_digest_match": {a: rec["arms"][a]["preflight_digest_match"] for a in ARM_ORDER},
                                    "comparison_vectors_ok": all(v["status"] == "OK" for v in rec["comparison_vector_files"].values())},
                           "bar1_preflight": pf["ok"], "bar2_regime": reg, "A_H": mH["A"], "G_H": mH["G"], "G_M_H": mH["G_M"], "n_H": mH["n"], "n_M_H": mH["n_M"],
                           "bar3_direction": {"label": dirlab, "cos_M_H": mH["cos_M"], "pair_cos_H": mH["pair_cos"], "h_amp": {a: h_amp({h: metrics[h]["G"][a] for h in GRID}) for a in R_ARMS},
                                              "h_amp_M": h_amp({h: metrics[h]["G_M"] for h in GRID}), "rotation_H": mH["rotation"], "rotation_M_H": mH["rotation_M"]},
                           "bar4_function": {"per_arm": per, "label": func, "dce_H": dce[str(H_END)], "over_threshold": over},
                           "bar5_tail": tail, "undefined_H": mH["undefined"]}
    rec["regime"] = reg
    rec["status"] = "DONE"
    rec["label"] = label(reg, dirlab, func)
    gates = ", ".join(f"{k} {v['total']}" for k, v in gate.items())
    print(f"[rdc1] A(H) {mH['A']} G_M(H) {mH['G_M']} cos_M(H) {mH['cos_M']} n(H) {mH['n']} -> {rec['label']}; gate {{ {gates} }}", flush=True)
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
    assert MODE == ("smoke" if SMOKE else "control"), (MODE, SMOKE)          # MODE=smoke without SMOKE=1 must never reach the real paths
    t0 = time.time()
    PHASE["t0"] = t0
    rec = {"status": "SETUP", "regime": "NOT-RUN", "prereg": "RANDOM-DIRECTION-CONTROL-1"}
    armed_s = install_wall_limit(MAX_WALL_S, RECEIPT, rec)
    try:
        _setup_and_run(rec, armed_s)
    except WallLimit:
        rec.update({"status": "NOT-RUN", "regime": "NOT-RUN", "label": wall_label()})
        rec["wall_limit"] = dict(rec.get("wall_limit", {}), armed_s=armed_s, fired=True, **wall_record())
        print(f"[rdc1] WALL LIMIT: {rec['label']}", flush=True)
    except SystemExit as e:
        if rec.get("status") in (None, "SETUP", "RUNNING"):
            rec.update({"status": "NOT-RUN", "regime": "NOT-RUN", "label": f"NOT-RUN (refused: {str(e)[:200]}; phase {PHASE['name']})"})
        _write_receipt(rec, t0)
        raise
    except BaseException as e:
        if rec.get("status") in (None, "SETUP", "RUNNING"):
            rec.update({"status": "CRASHED", "regime": "NOT-RUN", "label": f"CRASHED ({type(e).__name__}: {str(e)[:200]}; phase {PHASE['name']})"})
        _write_receipt(rec, t0)
        raise
    _write_receipt(rec, t0)
    print(f"[rdc1] {rec.get('status')} in {rec['wall_s']} s -> {RECEIPT}", flush=True)
    if rec.get("status") != "DONE":
        raise SystemExit(3)


def _write_receipt(rec, t0):
    rec["wall_s"] = round(time.time() - t0, 1); rec["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    RECEIPT.write_text(json.dumps(rec, indent=1) + "\n")
    signal.alarm(0)


def _setup_and_run(rec, armed_s):
    tok = TM.MathTokenizer()
    assert len(tok.vocab) == 40 and not os.environ.get("VOCAB_EXTRA") and not os.environ.get("SEQ_CAP") and not os.environ.get("BIRTH_BS") and TM.BS == OMA.BS
    OMA.OA.assert_verbatim()
    segs, d, flat_dg = UG.flatten_law(UG.build(tok, "cpu"))
    assert d == 18_911_616
    enc, starts, info = OMA.future_stream(tok)
    ugc0 = json.loads(OMA.UGC0.read_text())
    assert info["n_enc"] == 164_490 and info["steps_per_epoch"] == 5_140 and len(starts) == 5_140
    assert info["probe64_digest"] == ugc0["probe"]["digest"]
    info["flatten_law_digest"] = flat_dg
    batches, probe = UG.probe_batches(tok)
    held = [batches[i] for i in probe["held"]]
    info["held_panel"] = {"n": len(held), "probe_digest": probe["digest"], "ugc0_digest": ugc0["probe"]["digest"]}
    if not SMOKE:
        assert probe["digest"] == ugc0["probe"]["digest"]
    rearmed = None
    if SMOKE:
        if os.environ.get("SMOKE_ARENA_TAG"):
            assert FMEL2.exists() and FMEL1.exists() and FME1.exists() and STAGE0.exists(), "SMOKE_ARENA_TAG needs an existing smoke arena with an FMEL2 ladder receipt"
            _l2 = json.loads(FMEL2.read_text())
            assert _l2["status"] == "DONE" and all(str(h) in _l2["arms"][M_ARM]["snapshots"] for h in GRID), "reused smoke arena lacks a DONE FMEL2 ladder"
        else:
            # L2's smoke pipeline builds the arena (OMA smoke anchors + Stage 0 + desk + FME1 + the L1 refusal preflight) and then its own
            # DONE ladder on it, whose e1e-1 snapshots are this rung's comparison vectors. L2.main disarms the alarm when it writes its
            # receipt, so this rung's cap is re-armed with the remaining budget right after.
            if FMEL2.exists():
                raise SystemExit(f"REFUSING: smoke FMEL2 arena {FMEL2} exists")
            L2.main()
            assert FMEL2.exists() and json.loads(FMEL2.read_text())["status"] == "DONE"
            remaining = max(1, int(MAX_WALL_S - (time.time() - PHASE["t0"])))
            rearmed = install_wall_limit(remaining, RECEIPT, rec)
        OMA.CK_DIR = CK_DIR
    base = OMA.base_record("rdc1-control", tok, info, "cpu", "mps" if torch.backends.mps.is_available() else "cpu")
    rec.clear(); rec.update(base)
    rec["oma_source_sha256"] = rec.pop("source_sha256")
    rec.update({"prereg": "RANDOM-DIRECTION-CONTROL-1", "kind": "random_direction_control/control", "writer": WRITER, "n_pred_literal": N_PRED,
                "self_sha256": sha256_file(__file__), "l2_source_sha256": sha256_file(L2.__file__), "l1_source_sha256": sha256_file(L1.__file__), "fme1_source_sha256": sha256_file(FME.__file__),
                "ck_dir": str(CK_DIR), "leg_steps": LEG_FULL, "target": ANCHOR + LEG_FULL, "horizons": GRID, "seeds": SEEDS, "arms": {}, "arm_order": ARM_ORDER, "comparison_arm": M_ARM, "m_eps": M_EPS,
                "law": {"MAG_TOL": MAG_TOL, "COS_M_MAX": COS_M_MAX, "SHARE_TOL": SHARE_TOL, "COS_PAIR_MAX": COS_PAIR_MAX, "GENERIC_BAND": list(GENERIC_BAND), "SPECIFIC_MAX": SPECIFIC_MAX,
                        "DOMINANT_MIN": DOMINANT_MIN, "SHARED_LATE": SHARED_LATE, "INDEP_LATE": INDEP_LATE, "H_AMP_G": H_AMP_G, "CE_ABS": CE_ABS, "TAIL_H": TAIL_H, "MIN_FREE_BYTES": MIN_FREE_BYTES,
                        "ADAM_EPS": ADAM_EPS, "MAX_WALL_S": MAX_WALL_S, "MAX_WALL_S_REAL": MAX_WALL_S_REAL,
                        "regime_order": ["REGIME-UNRESOLVED", "DIRECTION-GENERIC", "MOMENT-SPECIFIC", "RANDOM-DOMINANT", "MIXED"],
                        "construction": "per group: dm_M = -0.1 m, v_M = K dm_M; q ~ N(0, I) float64 in exp_avg coordinates; q_perp = q - (<K q, v_M> / ||v_M||^2) dm_M; "
                                        "dm_R = (||v_M|| / ||K q_perp||) q_perp; exp_avg <- float32(float64(exp_avg) + dm_R); no division by K (AMENDMENT L76847)",
                        "targets": "derived: dW_M(1) = W_M(1) - W_C(1) from the sha-pinned FMEL2 e1e-1 h = 1 snapshot and the fresh C (digest == locked FMEL2 C h = 1)",
                        "zero_norm": "UNDEFINED = null + reason, never NaN; G needs the h = 1 norm, A needs G_r and G_M; an UNDEFINED A_r(H) -> REGIME-UNRESOLVED"}})
    rec.pop("eps_twin", None)
    rec["writers"] = {WRITER: rec["writers"][WRITER]}
    rec["wall_limit"] = {"armed_s": armed_s, "fired": False, "armed_at": "top of main (whole run)", "rearmed_after_smoke_arena_s": rearmed}
    rec["status"] = "SETUP"

    def stream(row):
        with STREAM.open("a") as f:
            f.write(json.dumps(row) + "\n")

    out = mode_control(tok, enc, starts, info, segs, d, held, rec, stream)
    assert out is rec


if __name__ == "__main__":
    main()
