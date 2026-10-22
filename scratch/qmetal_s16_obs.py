"""Observations adapter for PRE-REG QWEN-METAL-S16-V5-0: derives every
registered measurement from the A/B receipt and the qualification
summary in committed code (never hand-typed), refusing partial or smoke
inputs, and writes the observations document the adjudicator reads.

  bar 1  improvement_percent_ctx126   100 (control - candidate) / control
                                      of the per-version medians of the
                                      round medians at context 126
  bar 2  improvement_percent_ctx1024  the same at context 1,024
  1:worst, 2:worst                    the worst pairing: the slowest
                                      candidate round against the fastest
                                      control round
  refutation key min_improvement_percent  the smaller of the two bar values
Arm admissibility: CONTROL is generation 4 as shipped; CANDIDATE is
admissible only when every bit-identity check of the A/B is true, the
greedy tokens of the two versions agree at both contexts with a zero
logit difference, the qualification census is argmax-identical to the
CPU rows at every position with a cached decode argmax-identical to the
one-forward logits, and the peak unified memory of the candidate rounds
is within one percent of the control rounds at both contexts.
Usage: qmetal_s16_obs.py <ab.json> <qual summary.json> <prereg.json> > observations.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from llmopt.lab.prereg import load, registration_digest  # noqa: E402


def main(ab_path: str, qual_path: str, prereg_path: str) -> None:
    ab = json.load(open(ab_path))
    qual = json.load(open(qual_path))
    prereg = load(prereg_path)
    vc, vk = int(ab["control_version"]), int(ab["candidate_version"])
    if vc != 4 or vk != 5:
        raise SystemExit(f"REFUSING: A/B versions {vc}/{vk} are not the registered 4 against 5")
    if qual.get("smoke") or qual["tower"]["kernel_version"] != 5 or qual["n_positions"] != 4722:
        raise SystemExit("REFUSING: qualification summary is smoke, not generation 5, or not the full census")
    rows = [r for r in ab["tower_decode"] if "version" in r]
    cross = {r["context"]: r["cross_version"] for r in ab["tower_decode"] if "cross_version" in r}
    meas = {}
    reasons = []
    valid = True
    mem_ok = True
    for ctx, bar in ((126, "1"), (1024, "2")):
        c = [r["median_step_s"] for r in rows if r["context"] == ctx and r["version"] == vc]
        k = [r["median_step_s"] for r in rows if r["context"] == ctx and r["version"] == vk]
        if len(c) < 4 or len(k) < 4:
            valid = False
            reasons.append(f"context {ctx}: {len(c)} control and {len(k)} candidate rounds, fewer than the registered 4")
            continue
        mc, mk = float(np.median(c)), float(np.median(k))
        imp = 100.0 * (mc - mk) / mc
        worst = 100.0 * (min(c) - max(k)) / min(c)
        pc = [r["peak_mem_bytes"] for r in rows if r["context"] == ctx and r["version"] == vc]
        pk = [r["peak_mem_bytes"] for r in rows if r["context"] == ctx and r["version"] == vk]
        mem_ok &= max(pk) <= 1.01 * max(pc)
        meas[bar] = {"value": imp, "metric": f"improvement_percent_ctx{ctx}", "population": f"steps:ctx{ctx}-{ab['steps']}x{len(k)}rounds",
                     "aggregation": "percent_change_of_median_of_round_medians",
                     "provenance": f"control round medians {[round(x*1e3, 2) for x in c]} ms, candidate {[round(x*1e3, 2) for x in k]} ms; "
                                   f"medians {mc*1e3:.2f} and {mk*1e3:.2f} ms; peak bytes control {max(pc)} candidate {max(pk)}"}
        meas[f"{bar}:worst"] = {"value": worst, "metric": f"improvement_percent_worst_pairing_ctx{ctx}",
                                "population": f"steps:ctx{ctx}-{ab['steps']}x{len(k)}rounds", "aggregation": "percent_change_worst_pairing"}
    if valid:
        meas["min_improvement_percent"] = {"value": min(meas["1"]["value"], meas["2"]["value"]), "metric": "min_improvement_percent",
                                           "population": "steps:ctx126+ctx1024", "aggregation": "min_over_contexts"}
    bits = all(all(v) for m in ab["matrices"] for v in m["bit_identical_every_round"].values())
    tokens_equal = all(cross[c]["greedy_tokens_control_equal_candidate"] for c in cross)
    logit_zero = all(cross[c]["max_abs_logit_diff_control_vs_candidate"] == 0.0 for c in cross)
    repro = all(r.get("greedy_tokens_same_as_first_round_of_this_version", True) for r in rows)
    census_ok = qual["argmax_identical"] == qual["n_positions"]
    decode_ok = all(d["argmax_same_as_one_forward"] == d["steps"] for d in qual["cached_decode"])
    top16_ok = qual["top16_abs_err"]["max"] <= 1.1e-3   # the generation 4 census read 1.02e-3
    cand_ok = bits and tokens_equal and logit_zero and repro and census_ok and decode_ok and top16_ok and mem_ok
    obs = {"note": f"Mechanical projection of {ab_path} (driver scratch/bench_qmetal_ab_s16.py at {ab['provenance']['commit']}) and "
                   f"{qual_path} (scratch/qwen_metal_qual.py at {qual['code_commit']}, kernel generation {qual['tower']['kernel_version']}); "
                   f"adapter scratch/qmetal_s16_obs.py",
           "registration_sha256": registration_digest(prereg),
           "measurement_valid": valid, **({} if valid else {"measurement_reason": "; ".join(reasons)}),
           "arms": {"CONTROL": {"admissible": True, "reason": "generation 4 as shipped (W4 v4 dispatch, the original S16 kernel)"},
                    "CANDIDATE": {"admissible": bool(cand_ok),
                                  "reason": f"bit identity on every A/B check {bits}; greedy tokens equal at both contexts {tokens_equal}; "
                                            f"zero logit difference {logit_zero}; rounds reproduce their first round {repro}; census argmax identical "
                                            f"{qual['argmax_identical']}/{qual['n_positions']} {census_ok}; cached decode argmax identical to one forward {decode_ok}; "
                                            f"top-16 max abs error {qual['top16_abs_err']['max']:.3e} within 1.1e-3 {top16_ok}; "
                                            f"candidate peak memory within one percent of control {mem_ok}"}},
           "measurements": meas}
    json.dump(obs, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
