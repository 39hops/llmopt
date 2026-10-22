"""Observations adapter for PRE-REG QWEN-METAL-S16-V5-DEFAULT-0: derives
every registered measurement from the serving-path A/B receipt
(scratch/bench_qmetal_serve_v5.py, MODE=ab) in committed code, refusing
partial or smoke inputs, and writes the observations document the
adjudicator reads.

  bar 1  improvement_percent_ctx1024      100 (control - candidate) /
                                          control of the per-version
                                          medians of the round medians
                                          at context 1,024, serving path
  bar 2  improvement_percent_ctx2048      the same at context 2,048
  1:worst, 2:worst                        the worst pairing: the slowest
                                          candidate round against the
                                          fastest control round
  bar 3  decode_peak_active_ratio_ctx2048 the candidate's largest
                                          decode-phase peak active bytes
                                          over the control's, context
                                          2,048
  3:footprint                             the same ratio for the device
                                          footprint after decode
  refutation key min_worst_pairing_percent  the smaller of the two
                                          worst pairings (a negative
                                          value is a reproducible
                                          regression at that context)
Arm admissibility: CONTROL is generation 4 as shipped, every resident at
version 4 in its rounds; CANDIDATE is admissible only when every
resident is at version 5 in its rounds, the first rounds of the two
versions agree bit for bit on the final logits, the cache after prefill
and every decode step's logits at both contexts, and every later round
of each version reproduces its first round bit for bit.
Usage: qmetal_serve_v5_obs.py <ab.json> <prereg.json> > observations.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llmopt.lab.prereg import load, registration_digest  # noqa: E402


def main(ab_path: str, prereg_path: str) -> None:
    ab = json.load(open(ab_path))
    prereg = load(prereg_path)
    vc, vk = int(ab["control_version"]), int(ab["candidate_version"])
    if vc != 4 or vk != 5:
        raise SystemExit(f"REFUSING: A/B versions {vc}/{vk} are not the registered 4 against 5")
    if ab["provenance"].get("smoke") or ab["provenance"]["mode"] != "ab":
        raise SystemExit("REFUSING: the receipt is a smoke run or not the ab mode")
    if ab["provenance"].get("tree_dirty"):
        raise SystemExit("REFUSING: the receipt was written on a dirty tree")
    if sorted(ab["contexts"]) != [1024, 2048] or ab["serving_applied"].get("head") != "final":
        raise SystemExit(f"REFUSING: contexts {ab['contexts']} or head {ab['serving_applied'].get('head')} are not the registered ones")
    if ab["serving_applied"]["cache_limit_bytes"] != 2 * 1024 ** 3 or ab["serving_applied"]["chunk"] != 256 \
            or ab["serving_applied"]["matmul"] != "gemm":
        raise SystemExit(f"REFUSING: serving configuration {ab['serving_applied']} is not the registered one")
    rows = ab["rounds"]
    cross = ab["cross_version_first_rounds"]
    meas = {}
    reasons = []
    valid = True
    for ctx, bar in ((1024, "1"), (2048, "2")):
        c = [r for r in rows if r["context"] == ctx and r["version"] == vc]
        k = [r for r in rows if r["context"] == ctx and r["version"] == vk]
        if len(c) < 4 or len(k) < 4:
            valid = False
            reasons.append(f"context {ctx}: {len(c)} control and {len(k)} candidate rounds, fewer than the registered 4")
            continue
        if any(r["steps"] < 48 for r in c + k):
            valid = False
            reasons.append(f"context {ctx}: a round has fewer than 48 steps")
            continue
        cm = [r["median_step_s"] for r in c]
        km = [r["median_step_s"] for r in k]
        mc, mk = float(np.median(cm)), float(np.median(km))
        imp = 100.0 * (mc - mk) / mc
        worst = 100.0 * (min(cm) - max(km)) / min(cm)
        pop = f"steps:ctx{ctx}-{ab['steps']}x{len(k)}rounds-serving"
        meas[bar] = {"value": imp, "metric": f"improvement_percent_ctx{ctx}", "population": pop,
                     "aggregation": "percent_change_of_median_of_round_medians",
                     "provenance": f"control round medians {[round(x*1e3, 2) for x in cm]} ms, candidate {[round(x*1e3, 2) for x in km]} ms; "
                                   f"medians {mc*1e3:.2f} and {mk*1e3:.2f} ms; p90 by round control "
                                   f"{[round(r['p90_step_s']*1e3, 1) for r in c]} candidate {[round(r['p90_step_s']*1e3, 1) for r in k]}"}
        meas[f"{bar}:worst"] = {"value": worst, "metric": f"improvement_percent_worst_pairing_ctx{ctx}",
                                "population": pop, "aggregation": "percent_change_worst_pairing"}
        if ctx == 2048:
            pc = max(r["decode_peak_active_bytes"] for r in c)
            pk = max(r["decode_peak_active_bytes"] for r in k)
            fc = max(r["memory_after_decode"]["footprint_bytes"] for r in c)
            fk = max(r["memory_after_decode"]["footprint_bytes"] for r in k)
            meas["3"] = {"value": pk / pc, "metric": "decode_peak_active_ratio_ctx2048", "population": pop,
                         "aggregation": "ratio_of_max_over_rounds",
                         "provenance": f"decode-phase peak active bytes control {pc} candidate {pk}; footprint after decode control {fc} candidate {fk}"}
            meas["3:footprint"] = {"value": fk / fc, "metric": "footprint_after_decode_ratio_ctx2048", "population": pop,
                                   "aggregation": "ratio_of_max_over_rounds"}
    if valid:
        meas["min_worst_pairing_percent"] = {"value": min(meas["1:worst"]["value"], meas["2:worst"]["value"]),
                                             "metric": "min_worst_pairing_percent", "population": "steps:ctx1024+ctx2048-serving",
                                             "aggregation": "min_over_contexts"}
    ctrl_versions = all(r["resident_versions"] == [vc] for r in rows if r["version"] == vc)
    cand_versions = all(r["resident_versions"] == [vk] for r in rows if r["version"] == vk)
    bit = set(cross) == {"1024", "2048"} and all(
        cross[c]["final_logits"]["bit_identical"] and cross[c]["cache_after_prefill"]["bit_identical"]
        and cross[c]["decode_logits"]["bit_identical"] for c in cross)
    later = [r for r in rows if "vs_first_round_of_this_version" in r]
    # every round after a version's first carries the field: 3 per version per context
    repro = len(later) == 2 * 2 * 3 and all(
        r["vs_first_round_of_this_version"]["final_logits_max_abs"] == 0.0
        and r["vs_first_round_of_this_version"]["cache_bit_identical"]
        and r["vs_first_round_of_this_version"]["decode_logits_max_abs"] == 0.0 for r in later)
    obs = {"note": f"Mechanical projection of {ab_path} (driver scratch/bench_qmetal_serve_v5.py at {ab['provenance']['commit']}, "
                   f"serving configuration {ab['serving_applied']}); adapter scratch/qmetal_serve_v5_obs.py",
           "registration_sha256": registration_digest(prereg),
           "measurement_valid": valid, **({} if valid else {"measurement_reason": "; ".join(reasons)}),
           "arms": {"CONTROL": {"admissible": bool(ctrl_versions),
                                "reason": f"generation 4 as shipped; every resident at version 4 in every control round {ctrl_versions}"},
                    "CANDIDATE": {"admissible": bool(cand_versions and bit and repro),
                                  "reason": f"every resident at version 5 in every candidate round {cand_versions}; first rounds bit-identical "
                                            f"to the control on final logits, cache and decode logits at both contexts {bit}; "
                                            f"later rounds reproduce their first round bit for bit {repro}"}},
           "measurements": meas}
    json.dump(obs, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
