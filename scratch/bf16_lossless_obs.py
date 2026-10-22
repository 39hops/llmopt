"""Observations adapter for PRE-REG LOSSLESS-BF16-FEASIBILITY-0: derives
the registered measurements from the census receipt
(scratch/bf16_entropy_census.py) and the benchmark receipt
(scratch/bf16_lossless_bench.py) in committed code, with the text-model
parameter budget read from the pinned arm C manifest (the digest chain
is checked), and writes the observations document the adjudicator reads.

  bars 1, 2  extrapolated_text_model_gb_class_weighted: sum over classes
             of params(class) x best exact bits per weight(class) / 8,
             in GB (1e9 bytes); the best exact compressor of a class is
             the method with the fewest compressed bytes summed over
             the class's benchmarked tensors, among methods exact on
             every one of them
  bar 3      entropy_floor_text_model_gb_class_weighted: the same sum
             with the class's parameter-weighted mean H16 (no
             dictionary)
Class mapping: mlp, deltanet_proj, deltanet_gate_proj, attention_proj
and lm_head take their own sampled figures; the embedding (not sampled,
the same shape as lm_head) takes lm_head's; the small tensors (norms,
conv, A_log, dt_bias; 0.005 GB) take 16 bits. Ranges reported alongside:
the best sampled class figure applied everywhere, the worst applied
everywhere, and each single method applied with its own per-class
figures. Measurement is valid only when both pins verified, every
benchmark row is exact, both receipts are clean-tree non-smoke runs on
the same tensors and every class in the budget has a sampled figure or
a declared stand-in.
Usage: bf16_lossless_obs.py <entropy.json> <bench.json> <prereg.json> > observations.json
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llmopt.lab.prereg import load, registration_digest  # noqa: E402

ART = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
VDIR = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
STAND_IN = {"embed": "lm_head"}
SMALL_BITS = 16.0


def budget() -> dict:
    from llmopt.lab import qartifact
    q = qartifact.qualify_artifact(ART, os.path.join(VDIR, "model.safetensors.index.json"), "logs/qwenwhole/artifact_digest_C.txt")
    tot: dict = {}
    for k, e in q["manifest"].items():
        if not isinstance(e, dict) or "shape" not in e:
            continue
        if k.startswith("model.visual") or k.startswith("mtp."):
            continue
        if k == "lm_head.weight":
            cls = "lm_head"
        elif k.endswith("embed_tokens.weight"):
            cls = "embed"
        elif re.search(r"self_attn\.(q|k|v|o)_proj", k):
            cls = "attention_proj"
        elif re.search(r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)", k):
            cls = "deltanet_proj"
        elif re.search(r"linear_attn\.(in_proj_a|in_proj_b)", k):
            cls = "deltanet_gate_proj"
        elif re.search(r"mlp\.(gate|up|down)_proj", k):
            cls = "mlp"
        else:
            cls = "small"
        n = 1
        for d in e["shape"]:
            n *= int(d)
        tot[cls] = tot.get(cls, 0) + n
    return {"params_by_class": tot, "total_params": sum(tot.values()), "chain_rung0": q["report"]["rung0"]}


def main(entropy_path: str, bench_path: str, prereg_path: str) -> None:
    ent = json.load(open(entropy_path))
    ben = json.load(open(bench_path))
    prereg = load(prereg_path)
    reasons = []
    if ent.get("smoke") or ben.get("smoke"):
        reasons.append("a receipt is a smoke run")
    if ent.get("tree_dirty") or ben.get("tree_dirty"):
        reasons.append("a receipt was written on a dirty tree")
    for rc in (ent, ben):
        for sh, v in rc["shards"].items():
            if not v["pin_check"]["ok"]:
                reasons.append(f"pin check failed for {sh}")
    bad = [(t["name"], m) for t in ben["tensors"] for m, r in t["methods"].items() if not r["exact"]]
    if bad:
        reasons.append(f"inexact rows: {bad[:5]}")
    ent_names = {t["name"] for t in ent["tensors"] if not t["name"].startswith("mtp.")}
    ben_names = {t["name"] for t in ben["tensors"]}
    if not ben_names.issubset(ent_names):
        reasons.append("benchmark tensors are not all in the census")
    bud = budget()
    # per class: best exact method by summed compressed bytes
    classes: dict = {}
    for t in ben["tensors"]:
        c = classes.setdefault(t["class"], {"n": 0, "bytes_by_method": {}, "tensors": 0})
        c["n"] += t["n"]
        c["tensors"] += 1
        for m, r in t["methods"].items():
            c["bytes_by_method"][m] = c["bytes_by_method"].get(m, 0) + r["compressed_bytes"]
    per_class = {}
    for cls, c in classes.items():
        bpw = {m: 8.0 * b / c["n"] for m, b in c["bytes_by_method"].items()}
        best = min(bpw, key=lambda m: (bpw[m], m))
        per_class[cls] = {"sampled_params": c["n"], "tensors": c["tensors"], "bits_per_weight_by_method": bpw,
                          "best_method": best, "best_bits_per_weight": bpw[best]}
    h16 = {cls: v["H16_bits_weighted_mean"] for cls, v in ent["classes"].items()}
    methods = sorted({m for c in classes.values() for m in c["bytes_by_method"]})

    def figure(cls, key):
        src = STAND_IN.get(cls, cls)
        if cls == "small":
            return SMALL_BITS, "16 bits (small tensors, not compressed)"
        if src not in per_class:
            return None, f"no sampled figure for {cls}"
        if key == "best":
            return per_class[src]["best_bits_per_weight"], f"{src}:{per_class[src]['best_method']}"
        if key == "H16":
            return h16[src], f"{src}:H16"
        return per_class[src]["bits_per_weight_by_method"].get(key), f"{src}:{key}"

    def extrap(key):
        total_bits = 0.0
        detail = {}
        for cls, n in bud["params_by_class"].items():
            b, src = figure(cls, key)
            if b is None:
                reasons.append(src)
                return None, detail
            total_bits += n * b
            detail[cls] = {"params": n, "bits_per_weight": b, "source": src}
        return total_bits / 8 / 1e9, detail

    best_gb, best_detail = extrap("best")
    h16_gb, h16_detail = extrap("H16")
    per_method_gb = {m: extrap(m)[0] for m in methods}
    sampled_best = [v["best_bits_per_weight"] for v in per_class.values()]
    rng = {"best_sampled_class_everywhere_gb": bud["total_params"] * min(sampled_best) / 8 / 1e9 if sampled_best else None,
           "worst_sampled_class_everywhere_gb": bud["total_params"] * max(sampled_best) / 8 / 1e9 if sampled_best else None,
           "per_method_class_weighted_gb": per_method_gb}
    valid = not reasons
    meas = {}
    if valid:
        meas["1"] = {"value": best_gb, "metric": "extrapolated_text_model_gb_class_weighted",
                     "population": "text-model:26895998464-params",
                     "aggregation": "sum_over_classes_of_params_times_best_exact_bits_per_weight",
                     "provenance": json.dumps(best_detail)}
        meas["2"] = dict(meas["1"])
        meas["3"] = {"value": h16_gb, "metric": "entropy_floor_text_model_gb_class_weighted",
                     "population": "text-model:26895998464-params",
                     "aggregation": "sum_over_classes_of_params_times_H16_bits_per_weight",
                     "provenance": json.dumps(h16_detail)}
        meas["extrapolated_text_model_gb_class_weighted"] = {"value": best_gb, "metric": "extrapolated_text_model_gb_class_weighted",
                                                             "population": "text-model:26895998464-params",
                                                             "aggregation": "sum_over_classes_of_params_times_best_exact_bits_per_weight"}
    obs = {"note": f"Mechanical projection of {entropy_path} (census at {ent['commit']}) and {bench_path} (benchmark at {ben['commit']}); "
                   f"budget from the pinned arm C manifest ({bud['total_params']} text-model parameters, chain rung0 {bud['chain_rung0']}); "
                   f"adapter scratch/bf16_lossless_obs.py",
           "registration_sha256": registration_digest(prereg),
           "measurement_valid": valid, **({} if valid else {"measurement_reason": "; ".join(reasons)}),
           "arms": {"RAW": {"admissible": True, "reason": "the original bytes"},
                    "BEST_EXACT": {"admissible": valid, "reason": "every benchmark row exact and both pins verified" if valid else "; ".join(reasons)}},
           "measurements": meas,
           "per_class": per_class, "h16_by_class": h16, "budget": bud, "ranges": rng}
    json.dump(obs, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
