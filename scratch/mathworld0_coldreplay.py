"""MATH-CYBER-0 rung 0, cold-process replay qualification
(GPT post-booking fence): replay the FROZEN ACTIVE receipts of
OBSERVATION MATH-CYBER-0-RUNG0 in a FRESH python process with a
cold derivation _RULE_CACHE, comparing every causal field except
transition_wall_ms. Bridges same-process replay (the booked
101/101) to PROCESS-INDEPENDENT replay, ahead of any
python -> C++ parity claim. No new treatment claim; the frozen
driver scratch/mathworld0.py is imported, never edited.

Receipts: logs/mathworld0/coldreplay.jsonl,
coldreplay_verdict.json (refuse-if-exists).

    .venv/bin/python scratch/mathworld0_coldreplay.py         (Mac)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

import scratch.mathworld0 as w  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.mathgen.problems import make_integrate  # noqa: E402

OUT = Path("logs/mathworld0/coldreplay.jsonl")
VERDICT = Path("logs/mathworld0/coldreplay_verdict.json")


def main():
    for p in (OUT, VERDICT):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    START = start_provenance(
        ["scratch/mathworld0_coldreplay.py", "scratch/mathworld0.py",
         "llmopt/search/derivation.py", "llmopt/mathgen/problems.py"])
    active = {}
    for line in Path("logs/mathworld0/active.jsonl").read_text() \
            .splitlines():
        r = json.loads(line)
        if "meta" in r:
            continue
        active.setdefault(r["episode_id"], []).append(r)
    mismatches = []
    n_causal = 0
    matched = 0  # counted DIRECTLY per row (auditor hardening note)
    with OUT.open("a") as f:
        for eid, rows in active.items():
            level = int(eid.split("-")[0][1:])
            seed = int(eid.split("-s")[1])
            prob = make_integrate(level, seed)
            root = sp.Integral(prob._expr, w.X)
            script = [r["chosen_action"] for r in rows]
            rep = w.run_episode(eid, root, f, script=script)
            a = [r for r in rows if r["outcome"] != "wall_cap"]
            b = [r for r in rep if r["outcome"] != "wall_cap"]
            n_causal += len(a)
            if len(a) != len(b):
                mismatches.append((eid, "row_count", len(a), len(b)))
                continue
            for ra, rb in zip(a, b):
                bad = [k for k in ra if k != "transition_wall_ms"
                       and ra[k] != rb[k]]
                if bad:
                    mismatches.extend(
                        (eid, ra["step_id"], k, ra[k], rb[k])
                        for k in bad)
                else:
                    matched += 1
        f.write(json.dumps({"meta": {
            "note": "cold-process replay of the frozen ACTIVE "
                    "receipts (fresh interpreter, cold _RULE_CACHE)",
            "start": START,
            "completion_commit": completion_commit()}}) + "\n")
    verdict = {"episodes": len(active), "causal_rows": n_causal,
               "matched_rows": matched,
               "mismatches": mismatches[:50],
               "pass": not mismatches,
               "start": START,
               "completion_commit": completion_commit()}
    VERDICT.write_text(json.dumps(verdict, indent=1))
    print(f"[mw0-cold] {matched}/{n_causal} causal rows identical; "
          f"pass={verdict['pass']}", flush=True)
    return 0 if verdict["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
