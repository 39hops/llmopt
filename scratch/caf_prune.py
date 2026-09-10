"""CREDIT-ANCHOR-FRONTIER-1 artifact disposition (Artin direction 2026-09-09,
after DFA-LOWER-HARM-DESK-0 is booked): digest inventory of every file under
checkpoints/writercaf1/ against logs/writercaf1/qual.jsonl (file sha256 and
canonical state digest re-read), written to logs/writercaf1/prune_inventory.json
(refuses to overwrite); then keep, per arm: for the three zero controls and
the three lr 3e-4 hybrids final.pt and step_00463.pt (plus feedback.pt where
present); one canonical W_0 = qual_hybrid_k1_s23_S1_lr0.0003/step_00000.pt;
remove everything else (the lr 1e-4 hybrids entirely, the other snapshots).
Any digest mismatch aborts before anything is removed. To be copied into
scratch/ and run only after the desk is booked (no repo edits while a run is
live)."""
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

import torch  # noqa: E402

from atomtraj_pins import state_digest  # noqa: E402

OUT = Path("logs/writercaf1/prune_inventory.json")
KEEP_ARMS = {"qual_zero_k1_s23_lr0.0003", "qual_zero_k2_s23_lr0.0003", "qual_zero_k4_s23_lr0.0003",
             "qual_hybrid_k1_s23_S1_lr0.0003", "qual_hybrid_k2_s23_S1_lr0.0003", "qual_hybrid_k4_s23_S1_lr0.0003"}
KEEP_FILES = {"final.pt", "step_00463.pt", "feedback.pt"}
CANON_W0 = ("qual_hybrid_k1_s23_S1_lr0.0003", "step_00000.pt")


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    births = [json.loads(l) for l in open("logs/writercaf1/qual.jsonl") if '"kind": "birth"' in l]
    inv = {"prereg": "CREDIT-ANCHOR-FRONTIER-1", "direction": "Artin 2026-09-09 after DFA-LOWER-HARM-DESK-0", "arms": {}, "kept": [], "removed": [], "mismatch": []}
    plan = []
    for b in births:
        d = Path(b["outdir"])
        arm = {"files": {}}
        for p in sorted(d.iterdir()):
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            rec = {"sha256": sha, "bytes": p.stat().st_size}
            if p.name.startswith("step_"):
                step = str(int(p.stem.split("_")[1]))
                exp = b["snapshots"][step]
                rec["state_digest"] = state_digest(torch.load(p, map_location="cpu"))
                rec["matches_receipt"] = (sha == exp["file_sha256"] and rec["state_digest"] == exp["state_digest"])
            elif p.name == "final.pt":
                rec["state_digest"] = state_digest(torch.load(p, map_location="cpu"))
                rec["matches_receipt"] = (sha == b["final"]["file_sha256"] and rec["state_digest"] == b["final"]["state_digest"])
            elif p.name == "feedback.pt":
                rec["matches_receipt"] = (sha == b["feedback"]["file_sha256"])
            else:
                rec["matches_receipt"] = None
            if rec["matches_receipt"] is False:
                inv["mismatch"].append(str(p))
            arm["files"][p.name] = rec
            keep = (b["cell"] in KEEP_ARMS and p.name in KEEP_FILES) or ((b["cell"], p.name) == CANON_W0)
            (inv["kept"] if keep else plan).append(str(p))
        inv["arms"][b["cell"]] = arm
    if inv["mismatch"]:
        OUT.write_text(json.dumps(inv, indent=1))
        raise SystemExit(f"ABORT: digest mismatches, nothing removed: {inv['mismatch']}")
    freed = 0
    for p in plan:
        freed += Path(p).stat().st_size
        os.remove(p)
        inv["removed"].append(p)
    for b in births:
        d = Path(b["outdir"])
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
    inv["bytes_freed"] = freed
    inv["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    OUT.write_text(json.dumps(inv, indent=1))
    print(f"[prune] kept {len(inv['kept'])} files, removed {len(inv['removed'])} ({freed / 1024**3:.2f} GB), 0 mismatches -> {OUT}")


if __name__ == "__main__":
    main()
