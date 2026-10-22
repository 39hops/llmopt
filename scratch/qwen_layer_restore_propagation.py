"""Counterfactual perturbation-propagation analysis for the SELECTED PAIR
(layers 22 and 23) of VERDICT QWEN-LAYER-RESTORE-0, on the captures that
verdict already booked. No vendor bytes are read, no model runs, nothing
is fitted or selected on any outcome: every statistic below is a
descriptive summary of the booked CONTROL and GROUP arrays.

What this is: the distribution (not only the aggregate) of the hidden-
state perturbation that restoring layers 22 and 23 to the vendor BF16
weights injects at the pair's output (out23) and carries through the
downstream tower (out27, out31, out39, out47, out55, out63, the final
norm), together with how that perturbation lines up with the teacher-
margin bins, the per-position KL change, the teacher-token log-
probability change and the argmax flips.

What this is NOT: a 64-layer quantization-error census (one pair was
intervened on; the other 62 layers were never restored); an independent
validation set (E0 is the set the disagreement was characterised on, and
the corpus and prefix records are the same 454 positions the verdict
used); a proof that the compression error is distributed across layers
(no statement about any other layer follows from one pair's
perturbation).

Inputs (read-only, the file digests re-checked against the booked
receipts before anything is read): logs/qwen_layer_restore/
score_control.json and score_group.json with their capture dumps and
score arrays; the vendor tokenizer for rendering flipped tokens. Receipt
logs/qwen_layer_restore_propagation/propagation.json plus flips.jsonl
(refuse-if-exists). Row order of the captures: E0 (4,722) then corpus
(356) then prefix (98), the order the score stage wrote them.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from llmopt.lab import hash as lh  # noqa: E402
from llmopt.lab import layer_restore as lr  # noqa: E402

SRC = "logs/qwen_layer_restore"
OUT = os.environ.get("OUT", "logs/qwen_layer_restore_propagation")
VENDOR_DIR = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
BOUNDARIES = ["in22", "out23", "out27", "out31", "out39", "out47", "out55", "out63", "final_norm"]
SETS = [("e0", 4722), ("corpus", 356), ("prefix", 98)]
T_MARGIN_EDGES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf]
C_MARGIN_EDGES = [0.0, 0.05, 0.2, 0.5, 1.0, 2.0, np.inf]
REL_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.5]
STEP_EDGES = [0, 1, 2, 8, 32, 128, 10 ** 9]
TOP = 20
QS = [0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def q(x, qs=QS) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    d = {f"p{int(round(p * 100))}": float(v) for p, v in zip(qs, np.quantile(x, qs))}
    d.update({"n": int(x.size), "mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())})
    return d


def rank(x):
    x = np.asarray(x, dtype=np.float64)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, np.float64)
    r[o] = np.arange(1, x.size + 1)
    # average ranks on exact ties
    xs = x[o]
    i = 0
    while i < xs.size:
        j = i
        while j + 1 < xs.size and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return r


def spearman(a, b) -> float:
    a, b = rank(a), rank(b)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def bin_index(x, edges):
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)


def main() -> None:
    t_all = time.time()
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    recs = {a: json.load(open(f"{SRC}/score_{a}.json")) for a in ("control", "group")}
    for a, r in recs.items():
        if r["smoke"] or r["arm"] != a or not r.get("capture"):
            raise SystemExit(f"REFUSING: {a} receipt is not the booked capture arm")
    # ---- re-check every input digest (FILE digests, the kind the receipts hold) before reading
    checks = {}
    paths = {}
    for a, r in recs.items():
        for key, d in r["dumps"].items():
            p = os.path.join(r["dump_dir"], d["file"])
            c = lh.pinned_file_check(p, d["sha256"])
            checks[f"{a}:{key}"] = {k: c[k] for k in ("path", "kind", "file_sha256_ok", "ok")}
            if not c["ok"]:
                raise SystemExit(f"REFUSING: {a}:{key} does not match its booked file digest")
            paths[(a, key)] = p
    print(f"[prop] {len(checks)} input files match their booked file digests", flush=True)
    npz = {a: np.load(paths[(a, "score_npz")]) for a in recs}
    caps = {(a, b): np.load(paths[(a, b)], mmap_mode="r") for a in recs for b in BOUNDARIES}
    n_rows = caps[("control", "in22")].shape[0]
    if n_rows != sum(n for _, n in SETS) or any(c.shape != (n_rows, 5120) for c in caps.values()):
        raise SystemExit("REFUSING: capture shapes differ from the booked sets")
    spans, s0 = {}, 0
    for name, n in SETS:
        spans[name] = (s0, s0 + n)
        s0 += n
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(VENDOR_DIR)
        dec = lambda i: tok.decode([int(i)])  # noqa: E731
    except Exception as ex:  # the rendering is cosmetic; ids are the record
        print(f"[prop] tokenizer unavailable ({str(ex)[:80]}); ids only", flush=True)
        dec = lambda i: None  # noqa: E731

    # ---- per-row perturbation at every boundary (row relative L2 and cosine)
    t0 = time.time()
    row_rel = {}
    row_cos = {}
    for b in BOUNDARIES:
        A = np.asarray(caps[("control", b)], dtype=np.float64)
        B = np.asarray(caps[("group", b)], dtype=np.float64)
        d = A - B
        an = np.sqrt((A * A).sum(1))
        bn = np.sqrt((B * B).sum(1))
        row_rel[b] = np.sqrt((d * d).sum(1)) / np.maximum(an, np.finfo(np.float64).tiny)
        row_cos[b] = (A * B).sum(1) / np.maximum(an * bn, np.finfo(np.float64).tiny)
        del A, B, d
        print(f"[prop] {b}: row rel median {np.median(row_rel[b]):.4f} max {row_rel[b].max():.4f}", flush=True)
    print(f"[prop] boundaries in {time.time()-t0:.0f} s", flush=True)
    in22_identical = bool(row_rel["in22"].max() == 0.0)

    out = {"analysis_kind": "counterfactual perturbation-propagation analysis for the selected pair (layers 22 and 23), "
                            "on the booked CONTROL and GROUP captures of VERDICT QWEN-LAYER-RESTORE-0",
           "is_not": ["a 64-layer quantization-error census", "an independent validation set",
                      "a proof of distributed compression error", "an intervention selection or a fit on E0"],
           "code_commit": head, "tree_dirty": dirty, "driver": "scratch/qwen_layer_restore_propagation.py",
           "inputs": {"score_receipts": [f"{SRC}/score_control.json", f"{SRC}/score_group.json"],
                      "digest_checks": checks, "digest_kind": "file digests (lab.hash.pinned_file_check)",
                      "capture_row_order": [f"{n}:{a}-{b}" for n, (a, b) in spans.items()]},
           "in22_identical_by_construction": in22_identical, "sets": {}}

    for sname, (a0, a1) in spans.items():
        nz = {a: {k.split("/", 1)[1]: npz[a][k] for k in npz[a].files if k.startswith(sname + "/")} for a in recs}
        c, g = nz["control"], nz["group"]
        n = a1 - a0
        t_tok = c["t_token"]
        t_margin = c["t_margin"]
        c_arg, g_arg = c["argmax"], g["argmax"]
        c_ok, g_ok = c_arg == t_tok, g_arg == t_tok
        corrected = (~c_ok) & g_ok
        broken = c_ok & (~g_ok)
        changed = c_arg != g_arg
        both_wrong_changed = changed & (~c_ok) & (~g_ok)
        dkl = g["kl"] - c["kl"]
        dlp = (g["z_teacher_token"] - g["lse"]) - (c["z_teacher_token"] - c["lse"])
        rr = {b: row_rel[b][a0:a1] for b in BOUNDARIES}
        rc = {b: row_cos[b][a0:a1] for b in BOUNDARIES}
        fin, loc = rr["final_norm"], rr["out23"]
        S = {"n": int(n), "agree_control": int(c_ok.sum()), "agree_group": int(g_ok.sum()),
             "corrected": int(corrected.sum()), "broken": int(broken.sum()), "argmax_changed": int(changed.sum()),
             "both_wrong_changed": int(both_wrong_changed.sum()),
             "mean_kl_control": float(c["kl"].mean()), "mean_kl_group": float(g["kl"].mean())}
        # distribution of the perturbation at every boundary
        S["boundary_row_rel_l2"] = {b: dict(q(rr[b]), **{f"frac_above_{t}": float((rr[b] > t).mean()) for t in REL_THRESHOLDS})
                                    for b in BOUNDARIES}
        S["boundary_row_cosine"] = {b: q(rc[b], [0.001, 0.01, 0.05, 0.5]) for b in BOUNDARIES}
        S["boundary_pooled_rel_l2"] = {b: lr.boundary_stats(np.asarray(caps[("control", b)][a0:a1]),
                                                             np.asarray(caps[("group", b)][a0:a1]))["rel_l2"] for b in BOUNDARIES}
        # propagation: downstream over local, per row
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = {b: rr[b] / loc for b in BOUNDARIES if b not in ("in22", "out23")}
        S["row_ratio_to_out23"] = {b: q(r[np.isfinite(r)]) for b, r in ratio.items()}
        S["spearman_row_rel_out23_vs_downstream"] = {b: spearman(loc, rr[b]) for b in BOUNDARIES if b not in ("in22", "out23")}
        # KL and teacher-token log-probability changes
        S["delta_kl"] = dict(q(dkl), frac_improved=float((dkl < 0).mean()), frac_worsened=float((dkl > 0).mean()),
                             sum=float(dkl.sum()), rel_change_of_mean=float(g["kl"].mean() / c["kl"].mean() - 1.0))
        S["delta_logprob_teacher_token"] = dict(q(dlp), frac_improved=float((dlp > 0).mean()))
        S["kl_control"] = q(c["kl"])
        S["kl_group"] = q(g["kl"])
        # perturbation against outcome
        S["spearman"] = {"final_rel_vs_abs_delta_kl": spearman(fin, np.abs(dkl)),
                         "final_rel_vs_delta_kl": spearman(fin, dkl),
                         "final_rel_vs_kl_control": spearman(fin, c["kl"]),
                         "out23_rel_vs_abs_delta_kl": spearman(loc, np.abs(dkl)),
                         "final_rel_vs_teacher_margin": spearman(fin, t_margin),
                         "final_rel_vs_control_margin": spearman(fin, c["margin"])}
        S["final_rel_median_by_outcome"] = {"unchanged": float(np.median(fin[~changed])) if (~changed).any() else None,
                                            "changed": float(np.median(fin[changed])) if changed.any() else None,
                                            "corrected": float(np.median(fin[corrected])) if corrected.any() else None,
                                            "broken": float(np.median(fin[broken])) if broken.any() else None}
        # quintiles of the final-norm perturbation: flip rate and KL change per quintile
        edges = np.quantile(fin, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        qi = np.clip(np.searchsorted(edges, fin, side="right") - 1, 0, 4)
        S["by_final_rel_quintile"] = [{"quintile": k + 1, "rel_range": [float(edges[k]), float(edges[k + 1])], "n": int((qi == k).sum()),
                                       "argmax_changed": int(changed[qi == k].sum()), "corrected": int(corrected[qi == k].sum()),
                                       "broken": int(broken[qi == k].sum()), "mean_delta_kl": float(dkl[qi == k].mean()),
                                       "mean_abs_delta_kl": float(np.abs(dkl[qi == k]).mean())} for k in range(5)]
        # teacher-margin bins and control-margin bins
        def bins(values, edges, label):
            bi = bin_index(values, edges)
            rows = []
            for k in range(len(edges) - 1):
                m = bi == k
                rows.append({label: [float(edges[k]), (None if np.isinf(edges[k + 1]) else float(edges[k + 1]))], "n": int(m.sum()),
                             "agree_control": int(c_ok[m].sum()), "agree_group": int(g_ok[m].sum()),
                             "corrected": int(corrected[m].sum()), "broken": int(broken[m].sum()),
                             "argmax_changed": int(changed[m].sum()),
                             "mean_kl_control": float(c["kl"][m].mean()) if m.any() else None,
                             "mean_kl_group": float(g["kl"][m].mean()) if m.any() else None,
                             "mean_delta_kl": float(dkl[m].mean()) if m.any() else None,
                             "median_final_rel": float(np.median(fin[m])) if m.any() else None,
                             "median_out23_rel": float(np.median(loc[m])) if m.any() else None})
            return rows
        S["by_teacher_margin_bin"] = bins(t_margin, T_MARGIN_EDGES, "teacher_margin_nats")
        S["by_control_margin_bin"] = bins(c["margin"], C_MARGIN_EDGES, "control_top2_margin_nats")
        if sname == "e0":
            step = c["step"]
            S["by_rollout_step_bin"] = []
            bi = bin_index(step, STEP_EDGES)
            for k in range(len(STEP_EDGES) - 1):
                m = bi == k
                S["by_rollout_step_bin"].append({"step_range": [int(STEP_EDGES[k]), None if STEP_EDGES[k + 1] >= 10 ** 9 else int(STEP_EDGES[k + 1])],
                                                 "n": int(m.sum()), "median_final_rel": float(np.median(fin[m])) if m.any() else None,
                                                 "median_out23_rel": float(np.median(loc[m])) if m.any() else None,
                                                 "argmax_changed": int(changed[m].sum()), "corrected": int(corrected[m].sum()),
                                                 "broken": int(broken[m].sum()), "mean_delta_kl": float(dkl[m].mean()) if m.any() else None})
            blk = c["blocks"]
            S["by_prompt"] = [{"block": int(b), "n": int((blk == b).sum()), "median_final_rel": float(np.median(fin[blk == b])),
                               "corrected": int(corrected[blk == b].sum()), "broken": int(broken[blk == b].sum()),
                               "mean_delta_kl": float(dkl[blk == b].mean())} for b in np.unique(blk)]

        def row_record(i):
            return {"row": int(i), "block": int(c["blocks"][i]), "step": int(c["step"][i]) if "step" in c else None,
                    "teacher_token": int(t_tok[i]), "teacher_text": dec(t_tok[i]),
                    "control_argmax": int(c_arg[i]), "control_text": dec(c_arg[i]),
                    "group_argmax": int(g_arg[i]), "group_text": dec(g_arg[i]),
                    "teacher_margin": float(t_margin[i]), "control_margin": float(c["margin"][i]), "group_margin": float(g["margin"][i]),
                    "kl_control": float(c["kl"][i]), "kl_group": float(g["kl"][i]), "delta_kl": float(dkl[i]),
                    "delta_logprob_teacher": float(dlp[i]),
                    "rel_out23": float(loc[i]), "rel_final_norm": float(fin[i]), "cos_final_norm": float(rc["final_norm"][i]),
                    "class": ("corrected" if corrected[i] else "broken" if broken[i] else
                              "both_wrong_changed" if both_wrong_changed[i] else "unchanged")}
        S["extreme_rows"] = {"top_final_rel": [row_record(i) for i in np.argsort(-fin)[:TOP]],
                             "top_out23_rel": [row_record(i) for i in np.argsort(-loc)[:TOP]],
                             "largest_kl_improvements": [row_record(i) for i in np.argsort(dkl)[:TOP]],
                             "largest_kl_worsenings": [row_record(i) for i in np.argsort(-dkl)[:TOP]]}
        S["flips"] = [row_record(i) for i in np.flatnonzero(changed)]
        out["sets"][sname] = S
        print(f"[prop] {sname}: n {n} corrected {S['corrected']} broken {S['broken']} changed {S['argmax_changed']}; "
              f"final rel median {np.median(fin):.4f} (changed {S['final_rel_median_by_outcome']['changed']}); "
              f"dKL mean {dkl.mean():+.5f} frac improved {S['delta_kl']['frac_improved']:.3f}", flush=True)
    # pooled validation (corpus + prefix) KL and agreement, the verdict's set
    v0, v1 = spans["corpus"][0], spans["prefix"][1]
    out["validation_pooled"] = {"n": v1 - v0, "final_rel": q(row_rel["final_norm"][v0:v1]), "out23_rel": q(row_rel["out23"][v0:v1])}
    out["wall_s"] = time.time() - t_all
    os.makedirs(OUT)
    with open(os.path.join(OUT, "flips.jsonl"), "w") as f:
        for sname in out["sets"]:
            for r in out["sets"][sname]["flips"]:
                f.write(json.dumps(dict(set=sname, **r)) + "\n")
    json.dump(out, open(os.path.join(OUT, "propagation.json"), "w"), indent=1)
    print(f"[prop] wrote {OUT} in {out['wall_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
