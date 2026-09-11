"""SG-FAILURE-DESK-0: zero-main-model-training desk on the retained
SYNTHETIC-GRADIENT-WRITER-1 qualification snapshots (PRE-REG
SG-FAILURE-DESK-0). Nothing trains; no gate is read; no birth.

Distinguishes, per SG cell, snapshot and SG block l in 4..7:
 (1) INSTANTANEOUS REPRESENTABILITY: a deterministic float64 closed-form
     oracle from the registered predictor inputs [h_l, e, 1] to the
     normalized target Y = delta^BP_l / s_l, fitted on the FIT half of the
     frozen probe (chunks 0..3) and scored on the disjoint HELDOUT half
     (chunks 4..7): (a) LINEAR least squares (lambda 0, pinv; primary:
     n_fit of about 8k tokens against d_in 425), (b) LINEAR ridge with
     lambda chosen by closed-form GCV on the FIT set over the relative grid
     GCV_GRID_REL (conditioning check), (c) a RICHER frozen-state oracle:
     ridge on [X, relu(Z Omega + b)] with Z the FIT-standardized [h, e],
     RF_WIDTH random features, Omega ~ N(0, 1 / sqrt(d_in)), seed RF_SEED
     (the richer family contains the linear one), lambda by the same GCV
     law (a fixed 1e-6 relative ridge overfit 2049 features on
     the smoke's 2k fit tokens: held ratio 14 to 26). Reported: heldout
     normalized MSE / zero-
     baseline ratio (sum ||Y - Yhat||^2 / sum ||Y||^2), heldout pooled
     cosine, and the fit-set ratio (under / over-fit readout).
 (2) ONLINE PREDICTOR TRACKING: the saved phi_t (pred_step_t.pt) scored on
     the same HELDOUT tokens at its matching W_t and at the neighbouring
     desk snapshots W_prev / W_next (inputs and targets of those states),
     ratio and cosine each; target nonstationarity cos(Y_t, Y_next) and
     RMS(Y_next) / RMS(Y_t) per block over HELDOUT.
 (3) DYNAMICS, all five cells (control included) per snapshot and block:
     delta^BP element RMS (HELDOUT, eligible positions), hat_delta RMS
     (SG cells), their ratio, hidden-error cosine cos(hat_delta, delta^BP),
     the ACTUAL block-parameter-gradient cosine cos(J^T hat_delta, true
     grad) per block (gradients accumulated over the HELDOUT chunks; the
     true gradient is the frozen-top backprop gradient of the same state),
     residual effective rank of x_{l+1} for all eight blocks (the P3 / ACT
     law, dfa_act.effective_rank, HELDOUT eligible tokens), output-error
     RMS, head.weight Frobenius norm, norm.g L2 norm, the probe CE.
Snapshots: DESK_STEPS per cell (step 0 once: the shared W_0, asserted
equal across the five cells). Every state digest is asserted against
logs/sgwriter1/qual.jsonl before it is read. Writes logs/sgfail0/desk.json
(refuses to overwrite) + a summary row to logs/sgfail0/desk.jsonl.
SMOKE=1: the five sgwriter1_smoke_seal2 cells at steps 0 and 300, FIT
chunks 0..1, HELDOUT 2..3, logs/sgfail0/smoke.jsonl only.
Usage: .venv/bin/python scratch/sg_failure_desk.py
"""
import datetime
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
os.environ.setdefault("ARM", "off")
os.environ.setdefault("BIRTH_SEED", "0")

import torch  # noqa: E402

import train_mathnative as TM  # noqa: E402
from atomtraj_pins import state_digest  # noqa: E402
from dfa_act import effective_rank  # noqa: E402
from dfa_credit import block_params, causal_mask, ce_loss, freeze_lower, hybrid_objective  # noqa: E402
from dfa_probe import probe_rows, probe_tensors  # noqa: E402
from sg_credit import build_predictors, teacher_targets, writer_forward  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"
BIRTHS = Path("logs/sgwriter1/smoke_seal2.jsonl" if SMOKE else "logs/sgwriter1/qual.jsonl")
OUT_DIR = Path("logs/sgfail0")
OUT = OUT_DIR / ("smoke.jsonl" if SMOKE else "desk.json")
CHUNK = 32
FIT_CHUNKS = [0, 1] if SMOKE else [0, 1, 2, 3]
HELD_CHUNKS = [2, 3] if SMOKE else [4, 5, 6, 7]
DESK_STEPS = [0, 300] if SMOKE else [0, 463, 1028, 2056, 3084, 5140, 7196, 10280, 12336, 15420]
SG_BLOCKS = [4, 5, 6, 7]
RIDGE_REL = 1e-6
RF_WIDTH = 2048
RF_SEED = 777
SEED = 11 if SMOKE else 27


def rms(v):
    return math.sqrt(float((v * v).mean())) if v.numel() else float("nan")


def cos_pooled(a, b):
    na, nb = float(a.norm()), float(b.norm())
    return float((a * b).sum() / (na * nb)) if na > 0 and nb > 0 else None


def ratio(y, yhat):
    d = float((y * y).sum())
    return float(((y - yhat) ** 2).sum() / d) if d > 0 else None


def ridge_fit(X, Y, lam):
    """W = (X^T X + lam I)^-1 X^T Y in float64; lam 0 -> least squares by pinv."""
    if lam > 0:
        A = X.t() @ X + lam * torch.eye(X.shape[1], dtype=X.dtype)
        return torch.linalg.solve(A, X.t() @ Y)
    return torch.linalg.pinv(X) @ Y


def rel_lambda(X):
    return RIDGE_REL * float(torch.trace(X.t() @ X)) / X.shape[1]


GCV_GRID_REL = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def gcv_ridge(X, Y):
    """Ridge with lambda chosen by generalized cross-validation on the FIT
    set alone (closed form: GCV(lam) = (||Y - X W||^2 / n) / (1 - tr(H) / n)^2,
    tr(H) = sum d_i / (d_i + lam) over the eigenvalues d_i of X^T X), over
    the relative grid GCV_GRID_REL x tr(X^T X) / d_in. Deterministic; no
    held-out token is touched. Returns (W, lam, gcv_curve)."""
    n, d = X.shape
    XtX = X.t() @ X
    XtY = X.t() @ Y
    dvals, V = torch.linalg.eigh(0.5 * (XtX + XtX.t()))
    dvals = dvals.clamp_min(0)
    scale = float(torch.trace(XtX)) / d
    best = None
    curve = {}
    for rel in GCV_GRID_REL:
        lam = rel * scale
        W = V @ ((V.t() @ XtY) / (dvals + lam).unsqueeze(1))
        rss = float(((Y - X @ W) ** 2).sum()) / n
        tr_h = float((dvals / (dvals + lam)).sum())
        gcv = rss / (1 - tr_h / n) ** 2
        curve[f"{rel:g}"] = gcv
        if best is None or gcv < best[0]:
            best = (gcv, lam, W, rel)
    return best[2], best[1], {"curve": curve, "chosen_rel": best[3]}


def rf_features(HE, gen_seed=RF_SEED, width=RF_WIDTH):
    d_in = HE.shape[1]
    g = torch.Generator().manual_seed(gen_seed)
    Om = torch.randn(d_in, width, generator=g, dtype=torch.float64) / math.sqrt(d_in)
    b = torch.randn(width, generator=g, dtype=torch.float64)
    return torch.relu(HE @ Om + b)


def rf_design(X_lin, mu, sd):
    """The richer oracle's design: [X_lin (incl. bias), relu(Z Omega + b)] with
    Z = (X_lin[:, :-1] - mu) / sd standardized by FIT statistics, so the
    richer family CONTAINS the linear one."""
    Z = (X_lin[:, :-1] - mu) / sd
    return torch.cat([X_lin, rf_features(Z)], 1)


def state_arrays(model, tok, rows, chunk_ids, consts):
    """Per state: for the chunks given, the eligible-position arrays
    h_l (x_{l+1}), e, Y_l = delta^BP_l / s_l for l in SG_BLOCKS, plus the
    probe CE, e RMS and the rank accumulators for all eight blocks."""
    H = {l: [] for l in SG_BLOCKS}
    Y = {l: [] for l in SG_BLOCKS}
    E = []
    n = [0] * 8
    sx = [torch.zeros(384, dtype=torch.float64) for _ in range(8)]
    sxx = [torch.zeros(384, 384, dtype=torch.float64) for _ in range(8)]
    losses = []
    for ci in chunk_ids:
        ch = rows[ci * CHUNK:(ci + 1) * CHUNK]
        ids, mask, labels = probe_tensors(ch, tok)
        elig = labels != -100
        logits_T, outs_T = teacher_targets(model, ids, mask)
        loss = ce_loss(logits_T, labels)
        grads = torch.autograd.grad(loss, [outs_T[l] for l in SG_BLOCKS] + [logits_T])
        for g, l in zip(grads[:-1], SG_BLOCKS):
            H[l].append(outs_T[l].detach()[elig])
            Y[l].append(g.detach()[elig] / float(consts[str(l)]))
        E.append(grads[-1].detach()[elig])
        with torch.no_grad():
            for l in range(8):
                xs = outs_T[l].detach()[elig]
                n[l] += xs.shape[0]
                sx[l] += xs.sum(0)
                sxx[l] += xs.t() @ xs
        losses.append(float(loss.detach()))
    H = {l: torch.cat(v) for l, v in H.items()}
    Y = {l: torch.cat(v) for l, v in Y.items()}
    E = torch.cat(E)
    ranks = {}
    for l in range(8):
        r, why = effective_rank(n[l], sx[l], sxx[l])
        ranks[str(l)] = r if r is not None else why
    return {"H": H, "Y": Y, "E": E, "loss": sum(losses) / len(losses), "ranks": ranks, "n_tokens": int(E.shape[0])}


def param_grad_cosines(model, tok, rows, chunk_ids, preds, consts):
    """cos(J^T hat_delta, true frozen-top BP gradient) per SG block, gradients
    accumulated over the chunks; the model is the fp64 arena copy."""
    acc_sg = {l: [torch.zeros_like(p) for p in block_params(model, l)] for l in SG_BLOCKS}
    acc_bp = {l: [torch.zeros_like(p) for p in block_params(model, l)] for l in SG_BLOCKS}
    for ci in chunk_ids:
        ch = rows[ci * CHUNK:(ci + 1) * CHUNK]
        ids, mask, labels = probe_tensors(ch, tok)
        elig = (labels != -100).unsqueeze(-1).to(torch.float64)
        model.zero_grad(set_to_none=True)
        hybrid_objective(model, [], ids, mask, labels, 8)["loss"].backward()
        for l in SG_BLOCKS:
            for a, p in zip(acc_bp[l], block_params(model, l)):
                a += p.grad.detach()
        model.zero_grad(set_to_none=True)
        outs, logits = writer_forward(model, ids, mask, SG_BLOCKS)
        loss = ce_loss(logits, labels)
        e = torch.autograd.grad(loss, logits, retain_graph=True)[0].detach()
        S = 0.0
        with torch.no_grad():
            hats = {l: float(consts[str(l)]) * preds[str(l)](outs[l].detach(), e) * elig for l in SG_BLOCKS}
        for l in SG_BLOCKS:
            S = S + (hats[l] * outs[l]).sum()
        S.backward()
        for l in SG_BLOCKS:
            for a, p in zip(acc_sg[l], block_params(model, l)):
                a += p.grad.detach()
    model.zero_grad(set_to_none=True)
    out = {}
    for l in SG_BLOCKS:
        a = torch.cat([x.reshape(-1) for x in acc_sg[l]])
        b = torch.cat([x.reshape(-1) for x in acc_bp[l]])
        out[str(l)] = {"cos": cos_pooled(a, b), "norm_sg": float(a.norm()), "norm_bp": float(b.norm())}
    return out


def load_state(b, step, tok):
    p = Path(b["outdir"]) / f"step_{step:05d}.pt"
    sd = torch.load(p, map_location="cpu")
    assert state_digest(sd) == b["snapshots"][str(step)]["state_digest"], f"{p}: digest v receipt"
    model = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
    model.load_state_dict(sd)
    model = model.double().train()
    freeze_lower(model, 4)
    return model, sd, str(p)


def load_preds(b, step):
    p = Path(b["outdir"]) / f"pred_step_{step:05d}.pt"
    sd = torch.load(p, map_location="cpu")
    assert state_digest(sd) == b["pred_snapshots"][str(step)]["state_digest"], f"{p}: digest v receipt"
    preds = build_predictors(b["family"], SG_BLOCKS, seed=b["pred_seed"])
    preds.load_state_dict(sd)
    return preds.double().eval()


def predictor_eval(preds, arrays, consts):
    """phi on [h, e] of `arrays` v the targets of `arrays`, per block."""
    out = {}
    with torch.no_grad():
        for l in SG_BLOCKS:
            G = preds[str(l)](arrays["H"][l], arrays["E"])
            out[str(l)] = {"ratio": ratio(arrays["Y"][l], G), "cos": cos_pooled(G, arrays["Y"][l]),
                           "rms_hat": rms(float(consts[str(l)]) * G), "rms_delta": rms(float(consts[str(l)]) * arrays["Y"][l])}
    return out


def oracle_fits(fit, held):
    out = {}
    for l in SG_BLOCKS:
        Xf = torch.cat([fit["H"][l], fit["E"], torch.ones(fit["H"][l].shape[0], 1, dtype=torch.float64)], 1)
        Xh = torch.cat([held["H"][l], held["E"], torch.ones(held["H"][l].shape[0], 1, dtype=torch.float64)], 1)
        Yf, Yh = fit["Y"][l], held["Y"][l]
        res = {}
        Wl = ridge_fit(Xf, Yf, 0.0)
        res["linear_lstsq"] = {"held_ratio": ratio(Yh, Xh @ Wl), "held_cos": cos_pooled(Xh @ Wl, Yh), "fit_ratio": ratio(Yf, Xf @ Wl)}
        Wg, lam_g, info_g = gcv_ridge(Xf, Yf)
        res["linear_ridge_gcv"] = {"held_ratio": ratio(Yh, Xh @ Wg), "held_cos": cos_pooled(Xh @ Wg, Yh), "fit_ratio": ratio(Yf, Xf @ Wg), "lambda": lam_g, "chosen_rel": info_g["chosen_rel"]}
        mu = Xf[:, :-1].mean(0, keepdim=True)
        sd = Xf[:, :-1].std(0, keepdim=True).clamp_min(1e-12)
        Pf = rf_design(Xf, mu, sd)
        Ph = rf_design(Xh, mu, sd)
        Wr, lam_r, info_r = gcv_ridge(Pf, Yf)
        res["rf_ridge_gcv"] = {"held_ratio": ratio(Yh, Ph @ Wr), "held_cos": cos_pooled(Ph @ Wr, Yh), "fit_ratio": ratio(Yf, Pf @ Wr), "lambda": lam_r, "chosen_rel": info_r["chosen_rel"]}
        res["n_fit"] = int(Xf.shape[0])
        res["n_held"] = int(Xh.shape[0])
        out[str(l)] = res
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SMOKE and OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    if not SMOKE and dirty:
        raise SystemExit("REFUSING: registered desk on a dirty tree")
    tok = TM.MathTokenizer()
    _, rows, probe_digest, _ = probe_rows(tok)
    births = [json.loads(l) for l in BIRTHS.open() if '"kind": "birth"' in l]
    births = [b for b in births if b["seed"] == SEED and b["mode"] in ("zero", "sg") and b.get("final")]
    cells = {b["cell"]: b for b in births}
    assert len(cells) == 5, sorted(cells)
    inits = {b["init_state_digest"] for b in births}
    assert len(inits) == 1, "W_0 differs across cells"
    consts = next(b["constants"] for b in births if b["mode"] == "sg")
    rec = {"prereg": "SG-FAILURE-DESK-0", "kind": "sg_failure_desk", "smoke": SMOKE, "commit": commit, "tree_dirty": dirty,
           "probe_token_digest": probe_digest, "fit_chunks": FIT_CHUNKS, "held_chunks": HELD_CHUNKS, "desk_steps": DESK_STEPS,
           "ridge_rel": RIDGE_REL, "rf_width": RF_WIDTH, "rf_seed": RF_SEED, "constants": consts, "dtype": "float64", "device": "cpu",
           "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "cells": {}}
    t0 = time.time()
    for cell, b in sorted(cells.items(), key=lambda kv: (kv[1]["mode"] != "zero", kv[0])):
        is_sg = b["mode"] == "sg"
        steps = [s for s in DESK_STEPS if (s != 0 or b["mode"] == "zero")]
        crec = {"mode": b["mode"], "family": b.get("family"), "plr": b.get("plr"), "outdir": b["outdir"], "states": {}}
        cache = {}
        for step in steps:
            model, sd, path = load_state(b, step, tok)
            fit = state_arrays(model, tok, rows, FIT_CHUNKS, consts)
            held = state_arrays(model, tok, rows, HELD_CHUNKS, consts)
            srec = {"path": path, "state_digest": b["snapshots"][str(step)]["state_digest"], "probe_ce_fit": fit["loss"], "probe_ce_held": held["loss"],
                    "n_fit": fit["n_tokens"], "n_held": held["n_tokens"], "rank_held": held["ranks"],
                    "e_rms_held": rms(held["E"]), "head_weight_fro": float(sd["head.weight"].double().norm()), "norm_g_l2": float(sd["norm.g"].double().norm()),
                    "delta_rms_held": {str(l): rms(float(consts[str(l)]) * held["Y"][l]) for l in SG_BLOCKS},
                    "baseline_mse_held": {str(l): float((held["Y"][l] ** 2).mean()) for l in SG_BLOCKS},
                    "oracle": oracle_fits(fit, held)}
            if is_sg:
                preds = load_preds(b, step)
                srec["pred_path"] = str(Path(b["outdir"]) / f"pred_step_{step:05d}.pt")
                srec["online_match"] = predictor_eval(preds, held, consts)
                srec["hat_over_delta_rms"] = {str(l): (srec["online_match"][str(l)]["rms_hat"] / srec["online_match"][str(l)]["rms_delta"]
                                                       if srec["online_match"][str(l)]["rms_delta"] > 0 else None) for l in SG_BLOCKS}
                srec["param_grad_cos"] = param_grad_cosines(model, tok, rows, HELD_CHUNKS, preds, consts)
                cache[step] = {"held": held, "preds": preds}
            crec["states"][str(step)] = srec
            print(f"[sgfail] {cell} step {step}: ce {held['loss']:.3f} rank5..8 {[round(held['ranks'][str(l)], 2) if isinstance(held['ranks'][str(l)], float) else held['ranks'][str(l)] for l in SG_BLOCKS]} "
                  f"lstsq held ratio {[round(srec['oracle'][str(l)]['linear_lstsq']['held_ratio'], 3) for l in SG_BLOCKS]} rf {[round(srec['oracle'][str(l)]['rf_ridge_gcv']['held_ratio'], 3) for l in SG_BLOCKS]} "
                  + (f"online {[round(srec['online_match'][str(l)]['ratio'], 3) for l in SG_BLOCKS]} pgcos {[round(srec['param_grad_cos'][str(l)]['cos'], 3) if srec['param_grad_cos'][str(l)]['cos'] is not None else None for l in SG_BLOCKS]}" if is_sg else "")
                  + f" ({time.time() - t0:.0f}s)", flush=True)
        if is_sg:
            order = [s for s in steps if s in cache]
            for i, step in enumerate(order):
                srec = crec["states"][str(step)]
                srec["online_prev"] = predictor_eval(cache[step]["preds"], cache[order[i - 1]]["held"], consts) if i > 0 else None
                srec["online_next"] = predictor_eval(cache[step]["preds"], cache[order[i + 1]]["held"], consts) if i + 1 < len(order) else None
                if i + 1 < len(order):
                    Yt, Yn = cache[step]["held"]["Y"], cache[order[i + 1]]["held"]["Y"]
                    srec["target_nonstationarity_next"] = {str(l): {"cos": cos_pooled(Yt[l], Yn[l]), "rms_ratio_next_over_t": (rms(Yn[l]) / rms(Yt[l]) if rms(Yt[l]) > 0 else None)} for l in SG_BLOCKS}
                    srec["next_step"] = order[i + 1]
                if i > 0:
                    srec["prev_step"] = order[i - 1]
            del cache
        rec["cells"][cell] = crec
    rec["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rec["wall_s"] = round(time.time() - t0, 1)
    if SMOKE:
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print("[sgfail] smoke row appended")
    else:
        OUT.write_text(json.dumps(rec, indent=1))
        with (OUT_DIR / "desk.jsonl").open("a") as f:
            f.write(json.dumps({k: v for k, v in rec.items() if k != "cells"} | {"n_cells": len(rec["cells"]), "n_states": sum(len(c["states"]) for c in rec["cells"].values())}) + "\n")
        print(f"[sgfail] written {OUT}")


if __name__ == "__main__":
    main()
