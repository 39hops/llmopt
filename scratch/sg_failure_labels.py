"""SG-FAILURE-DESK-0 descriptive labels, computed from the desk receipt
(logs/sgfail0/desk.json) by the sealed thresholds (PRE-REG L70972,
AMENDMENT -AUDIT L71096); pure functions over the receipt, no checkpoint
is read. Per cell, medians over desk steps >= 463 and blocks 4..7:
  linear   = median HELDOUT linear_lstsq held_ratio      GOOD <= 0.5, POOR >= 0.9
  richer   = median min(linear_lstsq, rf_ridge_gcv)     GOOD <= 0.5 (while linear > 0.5)
  online   = median online_match ratio                   BAD >= 0.9
  collapse TARGET: baseline_mse_held < 1e-4 on some block, some step >= 463
           RANK:   rank_held[4..7] < 3 on some block, some step >= 463
           READOUT: head.weight Frobenius or norm.g L2 outside [0.5x, 2x] of
                    the control's at the same step
           SG-ONLY: a collapse readout on an SG cell not on the control at that step
  branches A: linear GOOD and online BAD;  B: linear not GOOD and richer GOOD;
           C: linear POOR and richer POOR;  D: any SG-only collapse.
Also the per-lag online prev / next medians, the nonstationarity medians,
the block-parameter-gradient cosine range, the MLP coincidence readout (at
the first step with baseline_mse_held < 1e-4: rank keys 4..7 and head /
norm norms v the control), and the registered-prior inputs. Writes
logs/sgfail0/labels.json (refuses to overwrite).
Usage: .venv/bin/python scratch/sg_failure_labels.py [desk.json] [labels.json]
"""
import json
import statistics
import sys
from pathlib import Path

SG_BLOCKS = ["4", "5", "6", "7"]
GOOD, POOR, ONLINE_BAD = 0.5, 0.9, 0.9
TARGET_TOL, RANK_TOL, READOUT_LO, READOUT_HI = 1e-4, 3.0, 0.5, 2.0


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def steps_ge(states, lo=463):
    return [s for s in states if int(s) >= lo]


def labels(rec):
    cells = rec["cells"]
    control = next(c for c in cells.values() if c["mode"] == "zero")
    out = {"thresholds": {"good": GOOD, "poor": POOR, "online_bad": ONLINE_BAD, "target": TARGET_TOL, "rank": RANK_TOL, "readout": [READOUT_LO, READOUT_HI]}, "cells": {}}
    for name, c in cells.items():
        st = c["states"]
        steps = steps_ge(st)
        lin = [st[s]["oracle"][l]["linear_lstsq"]["held_ratio"] for s in steps for l in SG_BLOCKS]
        rich = [min(st[s]["oracle"][l]["linear_lstsq"]["held_ratio"], st[s]["oracle"][l]["rf_ridge_gcv"]["held_ratio"]) for s in steps for l in SG_BLOCKS]
        rf_floor = sum(1 for s in steps for l in SG_BLOCKS if st[s]["oracle"][l]["rf_ridge_gcv"]["chosen_rel"] <= 1e-10)
        r = {"mode": c["mode"], "family": c.get("family"), "plr": c.get("plr"), "n_steps": len(steps),
             "probe_ce_held_per_step": {s: st[s]["probe_ce_held"] for s in steps},
             "rank_4_7_per_step": {s: {l: st[s]["rank_held"][l] for l in SG_BLOCKS} for s in steps},
             "delta_rms_per_step": {s: {l: st[s]["delta_rms_held"][l] for l in SG_BLOCKS} for s in steps},
             "head_fro_per_step": {s: st[s]["head_weight_fro"] for s in steps}, "norm_g_per_step": {s: st[s]["norm_g_l2"] for s in steps},
             "linear_median": med(lin), "richer_median": med(rich), "rf_gcv_at_grid_floor": rf_floor,
             "linear_per_step": {s: med([st[s]["oracle"][l]["linear_lstsq"]["held_ratio"] for l in SG_BLOCKS]) for s in steps},
             "richer_per_step": {s: med([min(st[s]["oracle"][l]["linear_lstsq"]["held_ratio"], st[s]["oracle"][l]["rf_ridge_gcv"]["held_ratio"]) for l in SG_BLOCKS]) for s in steps}}
        r["linear_good"] = r["linear_median"] is not None and r["linear_median"] <= GOOD
        r["linear_poor"] = r["linear_median"] is not None and r["linear_median"] >= POOR
        r["richer_good"] = (r["richer_median"] is not None and r["richer_median"] <= GOOD and not r["linear_good"])
        r["richer_poor"] = r["richer_median"] is not None and r["richer_median"] >= POOR
        # collapse readouts
        tgt = [(s, l) for s in steps for l in SG_BLOCKS if st[s]["baseline_mse_held"][l] < TARGET_TOL]
        rnk = [(s, l) for s in steps for l in SG_BLOCKS if isinstance(st[s]["rank_held"][l], float) and st[s]["rank_held"][l] < RANK_TOL]
        # a non-float rank is the ACT law's NOT-RESOLVABLE readout (degenerate covariance: a negative eigenvalue beyond tolerance);
        # it is reported separately and never counted as a rank below the threshold
        r["rank_not_resolvable"] = [(s, l, st[s]["rank_held"][l]) for s in steps for l in SG_BLOCKS if not isinstance(st[s]["rank_held"][l], float)]
        rdo = []
        for s in steps:
            cs = control["states"][s]
            for k in ("head_weight_fro", "norm_g_l2"):
                q = st[s][k] / cs[k]
                if q < READOUT_LO or q > READOUT_HI:
                    rdo.append((s, k, q))
        r["collapse_target"] = tgt
        r["collapse_rank"] = rnk
        r["collapse_readout"] = rdo
        if c["mode"] == "sg":
            on = [st[s]["online_match"][l]["ratio"] for s in steps for l in SG_BLOCKS]
            r["online_median"] = med(on)
            r["online_bad"] = r["online_median"] is not None and r["online_median"] >= ONLINE_BAD
            r["online_per_step"] = {s: med([st[s]["online_match"][l]["ratio"] for l in SG_BLOCKS]) for s in steps}
            r["online_prev_per_step"] = {s: {"lag": int(s) - st[s]["prev_step"], "ratio": med([st[s]["online_prev"][l]["ratio"] for l in SG_BLOCKS])} for s in steps if st[s].get("online_prev")}
            r["online_next_per_step"] = {s: {"lag": st[s]["next_step"] - int(s), "ratio": med([st[s]["online_next"][l]["ratio"] for l in SG_BLOCKS])} for s in steps if st[s].get("online_next")}
            r["nonstat_cos_per_step"] = {s: med([st[s]["target_nonstationarity_next"][l]["cos"] for l in SG_BLOCKS]) for s in steps if st[s].get("target_nonstationarity_next")}
            r["hidden_cos_range"] = [min(st[s]["online_match"][l]["cos"] for s in steps for l in SG_BLOCKS if st[s]["online_match"][l]["cos"] is not None),
                                     max(st[s]["online_match"][l]["cos"] for s in steps for l in SG_BLOCKS if st[s]["online_match"][l]["cos"] is not None)]
            pg = [st[s]["param_grad_cos"][l]["cos"] for s in steps for l in SG_BLOCKS if st[s]["param_grad_cos"][l]["cos"] is not None]
            r["param_grad_cos_range"] = [min(pg), max(pg)] if pg else None
            pg1028 = [st[s]["param_grad_cos"][l]["cos"] for s in steps_ge(st, 1028) for l in SG_BLOCKS if st[s]["param_grad_cos"][l]["cos"] is not None]
            r["param_grad_cos_within_0.2_from_1028"] = all(-0.2 <= x <= 0.2 for x in pg1028) if pg1028 else None
            r["hat_over_delta_rms_per_step"] = {s: med([st[s]["hat_over_delta_rms"][l] for l in SG_BLOCKS]) for s in steps}
            sg_only_t = [x for x in tgt if control["states"][x[0]]["baseline_mse_held"][x[1]] >= TARGET_TOL]
            sg_only_r = [x for x in rnk if not (isinstance(control["states"][x[0]]["rank_held"][x[1]], float) and control["states"][x[0]]["rank_held"][x[1]] < RANK_TOL)]
            r["sg_only_collapse"] = {"target": sg_only_t, "rank": sg_only_r, "readout": rdo}
            r["branch_A"] = bool(r["linear_good"] and r["online_bad"])
            r["branch_B"] = bool((not r["linear_good"]) and r["richer_good"])
            r["branch_C"] = bool(r["linear_poor"] and r["richer_poor"])
            r["branch_D"] = bool(sg_only_t or sg_only_r or rdo)
            if tgt:
                s0 = min(tgt, key=lambda x: int(x[0]))[0]
                r["coincidence_at_first_target_fall"] = {"step": s0, "rank_4_7": {l: st[s0]["rank_held"][l] for l in SG_BLOCKS},
                                                         "control_rank_4_7": {l: control["states"][s0]["rank_held"][l] for l in SG_BLOCKS},
                                                         "head_fro_v_control": st[s0]["head_weight_fro"] / control["states"][s0]["head_weight_fro"],
                                                         "norm_g_v_control": st[s0]["norm_g_l2"] / control["states"][s0]["norm_g_l2"],
                                                         "probe_ce_held": st[s0]["probe_ce_held"], "control_probe_ce_held": control["states"][s0]["probe_ce_held"]}
        else:
            r["control_collapse_none"] = not (tgt or rnk)
        out["cells"][name] = r
    return out


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/sgfail0/desk.json")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "logs/sgfail0/labels.json")
    if dst.exists():
        raise SystemExit(f"REFUSING: {dst} exists")
    rec = json.loads(src.read_text())
    out = labels(rec)
    out["source"] = str(src)
    out["desk_commit"] = rec.get("commit")
    dst.write_text(json.dumps(out, indent=1))
    for name, r in out["cells"].items():
        print(name, {k: r.get(k) for k in ("linear_median", "richer_median", "online_median", "branch_A", "branch_B", "branch_C", "branch_D", "control_collapse_none") if k in r})
    print(f"[labels] written {dst}")


if __name__ == "__main__":
    main()
