"""EX6-LOC-0 rescue-typology rider (desk, frozen pre-read of the
typology): per-level mechanism split of the booked factorial from
the existing perprob stream — no new inference.

For each problem (seed, idx, level) with the four paired arm
outcomes: among NONE-FAIL problems classify rescue type
  joint_only   only PROMPT solves it
  prefill_only PREFILL_ONLY solves, TOKEN1_ONLY does not
  token1_only  TOKEN1_ONLY solves, PREFILL_ONLY does not
  both_singles both single arms solve it
  (each type also tallies whether PROMPT retains the rescue)
and among NONE-SOLVE problems tally per-arm breaks. Question on
the record before reading: is L1's interaction +11 joint-only
rescue enrichment, and L3's -5 overlap/saturation of independent
single-arm rescues?

Receipt: logs/ex6loc/rider.json (refuse-if-exists).

    .venv/bin/python scratch/ex6loc_rider.py               (Mac desk)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

OUT = Path("logs/ex6loc/rider.json")
ARMS = ("loc_NONE", "loc_PREFILL_ONLY", "loc_TOKEN1_ONLY",
        "loc_PROMPT")


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(["scratch/ex6loc_rider.py",
                              "scratch/ex6loc.py"])
    ok = defaultdict(dict)
    level = {}
    for line in open("logs/ex6loc/ex6loc_perprob.jsonl"):
        r = json.loads(line)
        ok[(r["seed"], r["idx"])][r["frac"]] = r["ok"]
        level[(r["seed"], r["idx"])] = str(r["level"])
    out = {L: defaultdict(int) for L in ("1", "2", "3")}
    for key, arms in ok.items():
        assert len(arms) == 4, key
        L = level[key]
        n, p, t, pr = (arms[a] for a in ARMS)
        if not n:
            if pr and not p and not t:
                out[L]["joint_only_rescue"] += 1
            elif p and not t:
                out[L]["prefill_only_rescue"] += 1
                out[L]["prefill_only_kept_by_prompt"] += pr
            elif t and not p:
                out[L]["token1_only_rescue"] += 1
                out[L]["token1_only_kept_by_prompt"] += pr
            elif p and t:
                out[L]["both_singles_rescue"] += 1
                out[L]["both_singles_kept_by_prompt"] += pr
            else:
                out[L]["never_solved"] += 1
        else:
            out[L]["none_solved_base"] += 1
            for a, nm in (("loc_PREFILL_ONLY", "break_prefill"),
                          ("loc_TOKEN1_ONLY", "break_token1"),
                          ("loc_PROMPT", "break_prompt")):
                out[L][nm] += not arms[a]
    rcpt = {"note": "EX6-LOC rescue typology by level (desk, "
                    "from the booked perprob stream)",
            "start": START, "completion_commit": completion_commit(),
            "n_problems": len(ok),
            "by_level": {L: dict(d) for L, d in out.items()}}
    OUT.write_text(json.dumps(rcpt, indent=1) + "\n")
    for L in ("1", "2", "3"):
        print(L, dict(out[L]))
    print(f"receipt -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
