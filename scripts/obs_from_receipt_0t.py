"""QWEN-WHOLE-0T compile receipt -> typed observations.

    .venv/bin/python scripts/obs_from_receipt_0t.py \
        logs/qwenwhole/compile.jsonl > obs.json

Same closed-path contract as obs_from_receipt_0s.py: locked receipt
-> typed observations -> adjudicator, no hand transcription. The
adapter never reads the pre-reg's bar values; BAR 3's probe
baselines come from the PROBE RECEIPTS themselves
(logs/qwenprobe/L*.jsonl depth census for ffn,
logs/qwenprobe/family.jsonl for every non-FFN family), so the
ratio's denominator has measured provenance.

Derived measurements (ids match docs/preregs/qwen-whole-0t.json):
  1                    conservation_violations (already max over arms)
  2, 2:A, 2:C          realized artifact bytes per arm (B primary)
  3                    max over coded family:codec groups of
                       measured_op / probe_op
  3:ffn, 3:linear_attn_w4, 3:full_attn_w4, 3:io_s16, 3:attn_s16
                       absolute pooled per-family op errors
  refuted:artifact_bytes   arm B realized bytes (refutation clause)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FFN_CENSUS = [f"logs/qwenprobe/L{n}.jsonl" for n in (8, 32, 48, 56)]
FAMILY_PROBE = "logs/qwenprobe/family.jsonl"

# measured family:codec group -> (receipt families, probe arm label)
GROUPS = {
    "ffn:w4": (("ffn",), "W4"),
    "linear_attn:w4": (("linear_attn",), "W4@2"),
    "full_attn:w4": (("full_attn",), "W4@2"),
    "embeddings:w4": (("embeddings",), "W4@2"),
    "lm_head:w4": (("lm_head",), "W4@2"),
    "linear_attn:s16": (("linear_attn",), "S16@4"),
    "full_attn:s16": (("full_attn",), "S16@4"),
    "embeddings:s16": (("embeddings",), "S16@4"),
    "lm_head:s16": (("lm_head",), "S16@4"),
}


def probe_baselines() -> dict:
    """Worst-case pooled op per family:codec from the probe receipts."""
    base = {}
    ffn = []
    for p in FFN_CENSUS:
        row = json.loads(Path(p).read_text().splitlines()[0])
        if row.get("smoke"):
            raise SystemExit(f"REFUSING: smoke probe receipt {p}")
        ffn.append(row["operator_layer"]["W4"])
    base["ffn:w4"] = max(ffn)
    fam = json.loads(Path(FAMILY_PROBE).read_text().splitlines()[0])
    if fam.get("smoke"):
        raise SystemExit("REFUSING: smoke family-probe receipt")
    for key, (fams, arm) in GROUPS.items():
        if key == "ffn:w4":
            continue
        vals = [t["arms"][arm]["op"] for t in fam["tensors"]
                if t["family"] in fams]
        base[key] = max(vals)
    return base


def observations(summary: dict) -> dict:
    if summary.get("kind") != "summary":
        raise SystemExit("REFUSING: not a summary row")
    if summary.get("smoke"):
        raise SystemExit("REFUSING: smoke receipt can never be evidence")

    fe = summary["family_errors"]
    ab = summary["arm_bytes"]
    base = probe_baselines()

    ratios = {}
    for key in GROUPS:
        if key in fe:
            ratios[key] = fe[key]["op"] / base[key]
    missing = [k for k in ("ffn:w4", "linear_attn:w4", "full_attn:w4",
                           "embeddings:s16", "lm_head:s16") if k not in fe]
    if missing:
        return {"measurement_valid": False,
                "measurement_reason": f"family groups absent: {missing}"}

    def grp(*keys):
        return max(fe[k]["op"] for k in keys)

    mb = {"metric": "artifact_bytes", "population": "artifact:total",
          "aggregation": "sum"}
    # conjuncts share the bar's registered metric triple (prereg.py
    # enforces identity); values are absolute per-family pooled ops,
    # stated in provenance
    mf = {"metric": "op_error_family_max_ratio",
          "population": "families:coded",
          "aggregation": "max_over_families"}
    meas = {
        "1": {"metric": "conservation_violations",
              "population": "tensors:all_source_keys",
              "aggregation": "max_over_arms",
              "value": summary["conservation_violations"],
              "provenance": "summary.conservation_violations "
                            "(max over arms, global key check)"},
        "2": dict(mb, value=ab["B"], provenance="arm_bytes.B realized"),
        "2:A": dict(mb, value=ab["A"], provenance="arm_bytes.A realized"),
        "2:C": dict(mb, value=ab["C"], provenance="arm_bytes.C realized"),
        "3": {"metric": "op_error_family_max_ratio",
              "population": "families:coded",
              "aggregation": "max_over_families",
              "value": max(ratios.values()),
              "provenance": "max over " + ", ".join(
                  f"{k} {fe[k]['op']:.4f}/{base[k]:.4f}="
                  f"{r:.3f}" for k, r in sorted(ratios.items()))},
        "3:ffn": dict(mf, value=fe["ffn:w4"]["op"],
                      provenance="family_errors ffn:w4 pooled op"),
        "3:linear_attn_w4": dict(mf, value=fe["linear_attn:w4"]["op"],
                                 provenance="family_errors linear_attn:w4"),
        "3:full_attn_w4": dict(mf, value=fe["full_attn:w4"]["op"],
                               provenance="family_errors full_attn:w4"),
        "3:io_s16": dict(mf, value=grp("embeddings:s16", "lm_head:s16"),
                         provenance="max(embeddings:s16, lm_head:s16)"),
        "3:attn_s16": dict(mf, value=grp("linear_attn:s16",
                                         "full_attn:s16"),
                           provenance="max(linear_attn:s16, "
                                      "full_attn:s16)"),
        "refuted:artifact_bytes": dict(
            mb, value=ab["B"],
            provenance="arm_bytes.B realized (refutation clause)"),
    }
    arms = {a: {"admissible": True} for a in ("A", "B", "C")}
    contrasts = {b: {"admissible": True,
                     "reason": "one streaming pass, shared encoders, "
                               "same source shards"}
                 for b in ("1", "2", "3")}
    return {"measurement_valid": True, "arms": arms,
            "contrasts": contrasts, "measurements": meas,
            "receipt": {"code_commit": summary.get("code_commit"),
                        "revision": summary.get("revision"),
                        "wall_s": summary.get("wall_s")}}


def main() -> int:
    rows = [json.loads(l) for l in
            Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    summaries = [r for r in rows if r.get("kind") == "summary"]
    if len(summaries) != 1:
        raise SystemExit(f"REFUSING: {len(summaries)} summary rows")
    print(json.dumps(observations(summaries[0]), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
