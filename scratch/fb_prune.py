"""FROZEN-BACKBONE-1 artifact disposition (AMENDMENT SYNTHETIC-GRADIENT-
WRITER-1-ARENA fold 7, Artin direction 2026-09-10, after OBSERVATION
SG-PREDICTOR-AUDIT-0 is booked and receipt-complete): digest inventory of
every file under checkpoints/frozenbb1/ against logs/frozenbb1/births.jsonl
(file sha256 and canonical state digest re-read), written to
logs/frozenbb1/prune_inventory.json (refuses to overwrite); then keep, per
arm, step_00463.pt and final.pt, plus one W_0 anchor per seed
(step_00000.pt of the FULL arm; the FROZEN arm's step_00000 is the same
W_0 by the pair assertion of fb_gate.py and is removed); remove every
other snapshot. Any digest mismatch aborts before anything is removed.
Usage: .venv/bin/python scratch/fb_prune.py
"""
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import torch  # noqa: E402

from atomtraj_pins import state_digest  # noqa: E402

OUT = Path("logs/frozenbb1/prune_inventory.json")
KEEP_FILES = {"final.pt", "step_00463.pt"}


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    births = [json.loads(l) for l in open("logs/frozenbb1/births.jsonl") if '"kind": "birth"' in l]
    assert len(births) == 6
    inv = {"prereg": "FROZEN-BACKBONE-1", "direction": "AMENDMENT SYNTHETIC-GRADIENT-WRITER-1-ARENA fold 7 (Artin 2026-09-10), after SG-PREDICTOR-AUDIT-0",
           "arms": {}, "kept": [], "removed": [], "mismatch": []}
    plan = []
    w0_by_seed = {}
    for b in births:
        d = Path(b["outdir"])
        arm = {"files": {}, "seed": b["seed"], "arm": b["arm"], "init_state_digest": b["init_state_digest"]}
        for p in sorted(d.iterdir()):
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            rec = {"sha256": sha, "bytes": p.stat().st_size}
            if p.name.startswith("step_"):
                step = str(int(p.stem.split("_")[1]))
                exp = b["snapshots"][step]
                rec["state_digest"] = state_digest(torch.load(p, map_location="cpu"))
                rec["matches_receipt"] = (sha == exp["file_sha256"] and rec["state_digest"] == exp["state_digest"])
                if step == "0":
                    rec["matches_init_digest"] = rec["state_digest"] == b["init_state_digest"]
                    rec["matches_receipt"] = rec["matches_receipt"] and rec["matches_init_digest"]
            elif p.name == "final.pt":
                rec["state_digest"] = state_digest(torch.load(p, map_location="cpu"))
                rec["matches_receipt"] = (sha == b["final"]["file_sha256"] and rec["state_digest"] == b["final"]["state_digest"])
            else:
                rec["matches_receipt"] = None
            if rec["matches_receipt"] is False:
                inv["mismatch"].append(str(p))
            arm["files"][p.name] = rec
            keep = p.name in KEEP_FILES or (p.name == "step_00000.pt" and b["arm"] == "FULL")
            if keep and p.name == "step_00000.pt":
                w0_by_seed[b["seed"]] = str(p)
            (inv["kept"] if keep else plan).append(str(p))
        inv["arms"][b["cell"]] = arm
    inv["w0_anchor_by_seed"] = w0_by_seed
    assert sorted(w0_by_seed) == [24, 25, 26], w0_by_seed
    if inv["mismatch"]:
        OUT.write_text(json.dumps(inv, indent=1))
        raise SystemExit(f"ABORT: digest mismatches, nothing removed: {inv['mismatch']}")
    freed = 0
    for p in plan:
        freed += Path(p).stat().st_size
        os.remove(p)
        inv["removed"].append(p)
    inv["bytes_freed"] = freed
    inv["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    OUT.write_text(json.dumps(inv, indent=1))
    print(f"[prune] kept {len(inv['kept'])} files, removed {len(inv['removed'])} ({freed / 1024**3:.2f} GiB), 0 mismatches -> {OUT}")


if __name__ == "__main__":
    main()
