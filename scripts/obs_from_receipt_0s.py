"""0S receipt -> typed observations, with no hand transcription.

    .venv/bin/python scripts/obs_from_receipt_0s.py \
        logs/streamwd/pass0s_B1.jsonl > obs.json

Adopted 2026-08-16 (external review of the adjudication layer),
COMMITTED BEFORE the 0S receipt existed: the adjudicator was
deterministic but its observations document was hand-authored, so
transcription, wrong-operand, wrong-denominator, and
partial-population errors all lived at that seam — the exact class
this thread has been bitten by three times. This adapter closes the
path:  locked receipt -> typed observations -> adjudicator.

What it derives, deterministically, from the single receipt row:
  S-best        argmin pooled operator error over {S1-T, S1-U4, S2}
  bar 1         (S_best - W32) / S_best
  bar 2         (S2 - W4) / S2
  bar 3         (mean of three shuffled W32 - natural W32) / mean
  3:twin<seed>  (twin - natural) / twin, one per shuffle seed
  arm admissibility  from arm_within_budget
  measurement validity  smoke row REFUSED; walled arms make every
                arm they touch INADMISSIBLE (a partially trained
                stack must never produce an ordinary bar-bearing
                number — the wall hazard the review named)

The adapter never reads the pre-reg's bar values, so it cannot bend
a measurement toward a threshold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCALARS = ("S1-T", "S1-U4", "S2")
TWIN_SEEDS = (20260816, 20260817, 20260818)
M = {"metric": "operator_rel_gain", "population": "experts:0:256",
     "aggregation": "pooled_ratio"}


def observations(row: dict) -> dict:
    if row.get("smoke"):
        raise SystemExit("REFUSING: smoke receipt can never be evidence")
    if row.get("n_experts") != 256:
        return {"measurement_valid": False,
                "measurement_reason":
                    f"partial population n_experts={row.get('n_experts')}"}

    walled = set()
    for tag in row.get("codebook_walled_arms", []):
        walled.add(tag.split("/")[0])           # "W32-shuf.../w2" -> arm
    arms = {}
    for a, ok in row["arm_within_budget"].items():
        base = a.split("-shuf")[0] if a.startswith("W") else a
        name = a if not a.startswith("W") else base
        # collapse the three shuffled twins into the W32-shuf logical arm
        if "-shuf" in a:
            name = f"{base}-shuf"
        rec = arms.setdefault(name, {"admissible": True})
        if not ok:
            rec.update(admissible=False,
                       reason=f"{a} over budget at realized bytes")
        if a in walled or any(w.startswith(a) for w in walled):
            rec.update(admissible=False,
                       reason=f"{a} codebook walled: partial stack")
    op = row["operator_layer"]
    sbest_name = min(SCALARS, key=lambda a: op[a])
    arms["S-best"] = dict(arms[sbest_name],
                          **({"reason": f"derived from {sbest_name}"}
                             if arms[sbest_name]["admissible"] else {}))

    twins = [op[f"W32-shuf{s}"] for s in TWIN_SEEDS]
    mean_twin = sum(twins) / len(twins)
    meas = {
        "1": dict(M, value=(op[sbest_name] - op["W32"]) / op[sbest_name],
                  provenance=f"(S_best={sbest_name} {op[sbest_name]:.6f}"
                             f" - W32 {op['W32']:.6f})/S_best"),
        "2": dict(M, value=(op["S2"] - op["W4"]) / op["S2"],
                  provenance=f"(S2 {op['S2']:.6f} - W4 "
                             f"{op['W4']:.6f})/S2"),
        "3": dict(M, value=(mean_twin - op["W32"]) / mean_twin,
                  provenance=f"(mean_shuf {mean_twin:.6f} - W32 "
                             f"{op['W32']:.6f})/mean_shuf"),
    }
    for s, t in zip(TWIN_SEEDS, twins):
        meas[f"3:twin{s}"] = dict(
            M, value=(t - op["W32"]) / t,
            provenance=f"(twin{s} {t:.6f} - W32 {op['W32']:.6f})/twin")
    return {"measurement_valid": True, "arms": arms,
            "measurements": meas,
            "receipt": {"code_commit": row.get("code_commit"),
                        "revision": row.get("revision"),
                        "wall_s": row.get("wall_s")}}


def main() -> int:
    row = json.loads(Path(sys.argv[1]).read_text().splitlines()[0])
    print(json.dumps(observations(row), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
