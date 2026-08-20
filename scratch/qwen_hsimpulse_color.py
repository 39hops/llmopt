"""QWEN-HEADSWAP-IMPULSE-0 tail-period color producer (descriptive,
gates nothing). The committed reproduction of the ad-hoc pass that
wrote logs/qwenhsimpulse/impulse_color.json: if that file already
exists, this script recomputes everything from the sidecars and
ASSERTS equality with the existing receipt (repro mode); otherwise
it writes the file.

    .venv/bin/python scratch/qwen_hsimpulse_color.py         (3080)
"""
import glob
import json
import os
import sys

OUT = "logs/qwenhsimpulse"
FROZEN = "logs/qweneffort2_probe/traj_xhigh_{i}.json"


def tail_period(ids, cap=400, window=800):
    tail = ids[-window:]
    for L in range(1, cap + 1):
        if all(tail[i] == tail[i + L] for i in range(len(tail) - L)):
            return L
    return None


def main():
    rows = [json.loads(l) for l in
            open(os.path.join(OUT, "impulse_rows.jsonl"))]
    bykey = {(r["item"], r["inject_pos"]): r for r in rows}
    out = {"note": "exact-tail period test on the impulse sidecars "
                   "(cap 400 / window 800, stated); "
                   "tail_cycle_occurs_in_frozen = the final period-L "
                   "segment appears verbatim in the item frozen "
                   "greedy trajectory; descriptive color, gates "
                   "nothing",
           "branches": []}
    for sp in sorted(glob.glob(os.path.join(OUT, "traj_*.json"))):
        d = json.load(open(sp))
        ids = d["gen_token_ids"]
        r = d["row"]
        rr = bykey[(r["item"], r["inject_pos"])]
        L = tail_period(ids)
        fro = json.load(open(FROZEN.format(
            i=r["item"])))["gen_token_ids"]
        same = None
        if L:
            seg = ids[-L:]
            same = any(fro[i:i + L] == seg
                       for i in range(len(fro) - L))
        out["branches"].append({
            "sidecar": os.path.basename(sp), "item": r["item"],
            "pos": r["inject_pos"], "tail_period": L,
            "tail_cycle_occurs_in_frozen": same,
            "return_gap": rr["return_gap"],
            "outcome": rr["outcome"]})
    p = os.path.join(OUT, "impulse_color.json")
    if os.path.exists(p):
        prev = json.load(open(p))
        assert prev == out, "REPRO MISMATCH v existing impulse_color"
        print("REPRO MATCH: recomputed color equals the existing "
              "receipt byte-for-byte (modulo json formatting)")
    else:
        with open(p, "w") as f:
            f.write(json.dumps(out, indent=1) + "\n")
        print("written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
