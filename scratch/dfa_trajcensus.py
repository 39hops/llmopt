"""WRITER-DFA-1 trajectory census (PRE-REG L68321 item 6 (i)-(ii), bar
T-1 with the literal thresholds of L68543 B5): DFA v CTRL (shared W_0 =
seed 2, asserted from both step_00000.pt files and the seed
regeneration) at the 17 snapshots under the frozen tensor law of
WRITER-TRAJECTORY-CENSUS-0 (scratch/writertraj_census.py pair_census,
imported, not re-implemented): cumulative cosine C, relative divergence
R, velocity cosine V, per set. T-1 fires iff C_final < 0.7771 AND
R_final > 0.6677. Writes logs/writerdfa1/census.json (refuses to
overwrite). Zero gates.
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

from writertraj_census import pair_census, SETS, w0_seed, mid_median, load_model_sd  # noqa: E402
from atomtraj_pins import state_digest  # noqa: E402

OUT = Path("logs/writerdfa1")
T_TOTAL = 15_420
THRESH = {"C_max": 0.7771, "R_min": 0.6677}


def main():
    target = OUT / "census.json"
    if target.exists():
        raise SystemExit(f"REFUSING: {target} exists")
    assert not os.environ.get("VOCAB_EXTRA"), "W_0 law requires VOCAB_EXTRA unset"
    births = [json.loads(l) for l in (OUT / "births.jsonl").open() if '"kind": "birth"' in l]
    by_mode = {b["mode"]: b for b in births if b["phase"] == "disc" and b.get("final")}
    assert set(by_mode) == {"bp", "dfa"}, f"discovery births incomplete: {sorted(by_mode)}"
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    rec = {"prereg": "WRITER-DFA-1", "kind": "trajcensus", "commit": commit, "tree_dirty": dirty,
           "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "tensor_law": {k: len(v) for k, v in SETS.items()}, "artifacts": {}}
    X = {int(s): Path(by_mode["dfa"]["outdir"]) / f"step_{int(s):05d}.pt" for s in by_mode["dfa"]["snapshots"]}
    Y = {int(s): Path(by_mode["bp"]["outdir"]) / f"step_{int(s):05d}.pt" for s in by_mode["bp"]["snapshots"]}
    for p in list(X.values()) + list(Y.values()):
        if not p.exists():
            raise SystemExit(f"NOT-RUN: missing {p}")
    w0x, w0y = load_model_sd(X[0]), load_model_sd(Y[0])
    dx, dy, regen = state_digest(w0x), state_digest(w0y), state_digest(w0_seed(2))
    assert dx == dy == regen, "shared W_0 law: step_0 digests must equal each other and the seed-2 regeneration"
    rec["artifacts"] = {"w0_state_digest": dx, "w0_matches_seed_regeneration": True,
                        "DFA_paths": {str(s): str(p) for s, p in X.items()}, "CTRL_paths": {str(s): str(p) for s, p in Y.items()},
                        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in list(X.values()) + list(Y.values())}}
    pc = pair_census("DFA_CTRL", sorted(X), X, sorted(Y), Y, w0x, w0y, SETS)
    rec["pairs"] = {"DFA_CTRL": pc}
    f = str(max(pc["steps"]))
    C_f, R_f = pc["per_set"]["GLOBAL"][f]["C"], pc["per_set"]["GLOBAL"][f]["R"]
    rec["bars"] = {"final_step": int(f), "C_final": C_f, "R_final": R_f, "thresholds": THRESH,
                   "T-1": (C_f is not None and R_f is not None and C_f < THRESH["C_max"] and R_f > THRESH["R_min"]),
                   "V_mid_median": mid_median(pc, "V"),
                   "C_blocks_final": {str(l): pc["per_set"][f"BLOCK{l}"][f]["C"] for l in range(8)}}
    rec["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    target.write_text(json.dumps(rec, indent=1))
    print("[trajcensus] written; bars:", json.dumps(rec["bars"]))


if __name__ == "__main__":
    main()
