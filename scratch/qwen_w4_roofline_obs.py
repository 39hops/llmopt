"""Observations adapter for PRE-REG QWEN-W4-PREFILL-ROOFLINE-0: derives every
registered measurement from the two receipts of scratch/qwen_w4_roofline.py
(logs/qwen_w4_roofline/decomp.json and micro.json) in committed code and
writes the observations document scripts/adjudicate.py reads. Validity
floors: no smoke receipt, no dirty tree, one commit, the promoted default
(DEFAULT_RECURRENCE "kernel" in code, the engine on the kernel on every
DeltaNet layer, the kernel the only recurrence executed in every counted
block), the pinned model (revision from the teacher manifest, the chain
sha of the artifact on disk), every W4 matrix on the GEMM path, the
development prompt at 1,024 tokens, T = 256 present in the microbenchmark.
A smoke receipt still yields measurements (for pipeline inspection) but
measurement_valid is False, so every bar books UNRESOLVED.

Usage: .venv/bin/python scratch/qwen_w4_roofline_obs.py logs/qwen_w4_roofline docs/preregs/qwen-w4-prefill-roofline-0.json > obs.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llmopt.lab.prereg import load, registration_digest  # noqa: E402

HF_REVISION = "1d4bf0f2"          # prefix of the pinned Qwen3.8-27B revision (VERDICT QWEN-GDN-PROMOTE-0); the full 40-hex value is compared by prefix
CHAIN = "logs/qwenwhole/artifact_digest_C.txt"       # the arm C digest chain the engine loads (the receipt records it under model_pin.artifact.chain)
N_DNET = 48
MATRICES = ("layers.0.mlp.gate_proj", "layers.0.mlp.down_proj")


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(rdir, prereg_path):
    prereg = load(prereg_path)
    R, reasons = {}, []
    for st in ("decomp", "micro"):
        p = os.path.join(rdir, f"{st}.json")
        if not os.path.exists(p):
            reasons.append(f"missing receipt {p}")
            continue
        R[st] = json.load(open(p))
        if R[st].get("smoke"):
            reasons.append(f"{st}: smoke receipt")
        if R[st].get("tree_dirty"):
            reasons.append(f"{st}: dirty tree")
        e = R[st].get("engine", {})
        if e.get("default_recurrence_in_code") != "kernel" or e.get("recurrence") != "kernel" or e.get("recurrence_layers") != N_DNET:
            reasons.append(f"{st}: not the promoted default ({e.get('default_recurrence_in_code')}, {e.get('recurrence')}, {e.get('recurrence_layers')})")
        if e.get("chunk") != 256 or e.get("matmul") != "gemm" or e.get("kernel_version") != 5:
            reasons.append(f"{st}: not the serving configuration (chunk {e.get('chunk')}, matmul {e.get('matmul')}, generation {e.get('kernel_version')})")
        pin = R[st].get("model_pin", {})
        sw = pin.get("source_weights", {})
        if not str(sw.get("hf_revision", "")).startswith(HF_REVISION) or sw.get("revision_source") != "logs/qwenteacher_v2/teacher_manifest.json":
            reasons.append(f"{st}: the source-weights revision is not the pinned one ({sw.get('hf_revision')} from {sw.get('revision_source')})")
        if not os.path.exists(CHAIN):
            reasons.append(f"{st}: the artifact chain {CHAIN} is absent")
        elif pin.get("artifact", {}).get("chain") != CHAIN or pin.get("artifact", {}).get("chain_sha256") != fsha(CHAIN):
            reasons.append(f"{st}: the artifact chain in the receipt is not {CHAIN} at its current sha256")
        cfg = pin.get("config", {})
        if cfg.get("model_type") != "qwen3_5" or cfg.get("num_hidden_layers") != 64 or cfg.get("layer_types", {}).get("linear_attention") != N_DNET:
            reasons.append(f"{st}: the configuration summary is not the pinned model's")
        c = R[st].get("census", {})
        if c.get("w4", {}).get("matrices") != 192 or any(not r.get("gemm_ok_all") for r in c.get("by_shape", {}).values() if r.get("codec") == "w4"):
            reasons.append(f"{st}: the W4 census is not 192 GEMM-path matrices")
    if len(R) == 2 and R["decomp"]["commit"] != R["micro"]["commit"]:
        reasons.append(f"receipts from two commits {R['decomp']['commit']} {R['micro']['commit']}")
    meas, diag = {}, {}
    if "decomp" in R:
        D = R["decomp"]
        if (D.get("prompt", {}).get("tokens", 0) < 1000 or D.get("prompt", {}).get("seed") != "prefill-wall-dev-1-1024") and not D.get("smoke"):
            reasons.append(f"decomp: the prompt is not the 1,024-token development prompt ({D.get('prompt')})")
        for key in ("dispatch_marginal", "dispatch_additive"):
            d = D.get(key, {})
            ex = d.get("executed", {})
            if d.get("update_calls", 0) == 0 or ex.get("ops", 0) != 0 or ex.get("kernel", 0) == 0 or not d.get("kernel_only"):
                reasons.append(f"decomp: {key} did not execute the kernel alone ({ex}, kernel_only {d.get('kernel_only')})")
        dm = D.get("dispatch_marginal", {})
        if any(not r.get("kernel_only") for r in dm.get("runs", [])):
            reasons.append("decomp: a counted marginal run was not kernel-only")
        want = dm.get("standin_calls_expected_per_run")
        if not dm.get("standin_runs") or any(r.get("standin_calls") != want for r in dm.get("standin_runs", [])):
            reasons.append(f"decomp: the recurrence stand-in did not run {want} times in every no_gdn_recurrence run ({dm.get('standin_runs')})")
        M = D.get("marginal", {})
        if M:
            pop = "dev-1024:marginal:4v4"
            meas["1"] = {"value": M["no_w4_mlp"]["marginal_pct_of_full"], "metric": "w4_mlp_marginal_pct_of_full", "population": pop, "aggregation": "median_difference_pct",
                         "provenance": json.dumps({"full_median_s": M["full"]["median_s"], "no_w4_mlp_median_s": M["no_w4_mlp"]["median_s"], "runs": {k: M[k]["runs"] for k in ("full", "no_w4_mlp")}})}
            S = D["w4_split"]
            meas["2"] = {"value": S["decode_share_of_w4"], "metric": "w4_decode_marginal_over_w4_marginal", "population": pop, "aggregation": "median_difference_ratio",
                         "provenance": json.dumps({k: S[k] for k in ("w4_marginal_s", "decode_marginal_s", "matmul_marginal_s", "split_sum_over_w4")})}
            meas["8"] = {"value": S["split_sum_over_w4"], "metric": "w4_split_sum_over_w4_marginal", "population": pop, "aggregation": "median_difference_ratio",
                         "provenance": json.dumps({"matmul_share_of_w4": S["matmul_share_of_w4"]})}
            meas["8:low"] = dict(meas["8"])
        I = D.get("instrumented", {})
        if I:
            meas["7"] = {"value": I["gpu_side_share"], "metric": "gpu_side_share_of_instrumented_wall", "population": "dev-1024:instrumented:1", "aggregation": "single_run_ratio",
                         "provenance": json.dumps({"wall_s": I["wall_s"], "issue_s_total": I["issue_s_total"], "gpu_side_s_total": I["gpu_side_s_total"]})}
        diag["decomp"] = {"marginal_pct": {k: round(v["marginal_pct_of_full"], 2) for k, v in M.items()}, "additive": {k: round(v, 3) for k, v in D.get("additive_synchronous", {}).get("exclusive_s", {}).items()},
                          "full_median_s": M.get("full", {}).get("median_s")}
    if "micro" in R:
        U = R["micro"]
        rates = U.get("rates", {})
        rows = {m: U.get("matrices", {}).get(m, {}) for m in MATRICES}
        T = "256"
        if any(T not in r.get("rows", {}) for r in rows.values()) and not U.get("smoke"):
            reasons.append("micro: T = 256 absent for a registered matrix")
        if U.get("smoke"):
            T = next(iter(rows[MATRICES[0]].get("rows", {"256": None})))
        if all(T in r.get("rows", {}) for r in rows.values()):
            pop = "layer0-gate-and-down:T256:7-repeats"
            if not all(rows[m]["rows"][T].get("production_equals_matmul_on_slab") for m in MATRICES):
                reasons.append("micro: the production decode-then-matmul is not bit-equal to the matmul on the pre-decoded slab")
            eff = {m: rows[m]["rows"][T]["matmul_f32_share_of_peak"] for m in MATRICES}
            meas["3"] = {"value": min(eff.values()), "metric": "matmul_f32_tflops_over_square_peak", "population": pop, "aggregation": "min_over_shapes_of_median_ratio",
                         "provenance": json.dumps({"by_shape": eff, "peak_tflops": rates["matmul_f32_square"]["tflops"], "tflops": {m: rows[m]["rows"][T]["matmul_f32_tflops"] for m in MATRICES}})}
            bwr = {m: rows[m]["decode"]["effective_gb_per_s_single"] / rates["stream_add_1gib"]["gb_per_s"] for m in MATRICES}
            meas["4"] = {"value": min(bwr.values()), "metric": "decode_effective_bandwidth_over_stream_add", "population": "layer0-gate-and-down:7-repeats", "aggregation": "min_over_shapes_of_median_ratio",
                         "provenance": json.dumps({"by_shape": bwr, "stream_gb_per_s": rates["stream_add_1gib"]["gb_per_s"], "fill_gb_per_s": rates["fill_1gib"]["gb_per_s"],
                                                   "decode_gb_per_s": {m: rows[m]["decode"]["effective_gb_per_s_single"] for m in MATRICES}, "index_and_codebook_over_fill": {m: rows[m]["decode"]["index_and_codebook_over_fill"] for m in MATRICES}})}
            lib = {m: rows[m]["rows"][T]["library_qmm_f32_over_production"] for m in MATRICES}
            meas["5"] = {"value": min(lib.values()), "metric": "library_qmm_f32_over_production_decode_then_matmul", "population": pop, "aggregation": "min_over_shapes_of_median_ratio",
                         "provenance": json.dumps({"by_shape_f32": lib, "by_shape_f16_descriptive": {m: rows[m]["rows"][T]["library_qmm_f16_over_production"] for m in MATRICES},
                                                   "qmm_f32_over_matmul_f32": {m: rows[m]["rows"][T]["library_qmm_f32_over_matmul_f32"] for m in MATRICES},
                                                   "count_weighted_128_64_qmm_f32_over_production": (128 * rows[MATRICES[0]]["rows"][T]["library_qmm_affine4_f32_s"] + 64 * rows[MATRICES[1]]["rows"][T]["library_qmm_affine4_f32_s"]) / (128 * rows[MATRICES[0]]["rows"][T]["production_decode_then_matmul_s"] + 64 * rows[MATRICES[1]]["rows"][T]["production_decode_then_matmul_s"]),
                                                   "ideal_fused_over_production": {m: rows[m]["rows"][T]["ideal_fused_over_production"] for m in MATRICES}})}
            alloc = {m: rows[m]["rows"][T]["cleared_over_warm_pool"] for m in MATRICES}
            meas["6"] = {"value": max(alloc.values()), "metric": "production_cleared_pool_over_warm_pool", "population": pop, "aggregation": "max_over_shapes_of_median_ratio",
                         "provenance": json.dumps({"by_shape": alloc, "decode_pipelined4_over_single": {m: rows[m]["decode"].get("pipelined4_over_single") for m in MATRICES},
                                                   "decode_cleared_over_warm": {m: rows[m]["decode"].get("cleared_over_warm_decode") for m in MATRICES}})}
            diag["micro"] = {"rates": rates, "T": T, "rows": {m: {k: v for k, v in rows[m]["rows"][T].items() if k != "note"} for m in MATRICES},
                             "decode": {m: rows[m]["decode"] for m in MATRICES}, "fp16_weights_exact": {m: rows[m].get("fp16_weights_exact") for m in MATRICES},
                             "roofline": U.get("roofline")}
    valid = not reasons
    obs = {"registration_sha256": registration_digest(prereg), "receipts_dir": rdir, "commit": next(iter({r["commit"] for r in R.values()})) if len({r["commit"] for r in R.values()}) == 1 else None,
           "measurement_valid": valid, **({} if valid else {"measurement_reason": "; ".join(reasons)}),
           "arms": {"PROMOTED": {"admissible": valid, "reason": "the promoted default serving path (kernel recurrence selected by name per layer, chunk 256, GEMM scope); dispatch proven by the executed counts" if valid else "; ".join(reasons)}},
           "measurements": meas, "diagnostics": diag}
    json.dump(obs, sys.stdout, indent=1)
    print()
    return 0 if valid else 2


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(main(*sys.argv[1:]))
