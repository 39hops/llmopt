"""WRITER-DFA-1 qualification-set disposition (AMENDMENT CREDIT-ANCHOR-
FRONTIER-1-PRECISION L69223 F5, Artin GO 2026-09-09): one final digest
inventory of every file under checkpoints/writerdfa1/ against the birth
receipts in logs/writerdfa1/qual.jsonl (file sha256 and canonical state
digest re-read), written to logs/writerdfa1/prune_inventory.json (refuses
to overwrite), then removal of every snapshot except step_00000.pt,
step_00463.pt, final.pt and feedback.pt per arm. Receipts are untouched.
Any digest mismatch aborts before anything is removed.
"""
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

OUT = Path("logs/writerdfa1/prune_inventory.json")
KEEP = {"step_00000.pt", "step_00463.pt", "final.pt", "feedback.pt"}


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    births = [json.loads(l) for l in open("logs/writerdfa1/qual.jsonl") if '"kind": "birth"' in l]
    inv = {"prereg": "WRITER-DFA-1", "amendment": "CREDIT-ANCHOR-FRONTIER-1-PRECISION F5", "arms": {}, "kept": [], "removed": [], "mismatch": []}
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
                sd = torch.load(p, map_location="cpu")
                rec["state_digest"] = state_digest(sd)
                rec["matches_receipt"] = (sha == exp["file_sha256"] and rec["state_digest"] == exp["state_digest"])
            elif p.name == "final.pt":
                sd = torch.load(p, map_location="cpu")
                rec["state_digest"] = state_digest(sd)
                rec["matches_receipt"] = (sha == b["final"]["file_sha256"] and rec["state_digest"] == b["final"]["state_digest"])
            elif p.name == "feedback.pt":
                rec["matches_receipt"] = (sha == b["feedback"]["file_sha256"])
            else:
                rec["matches_receipt"] = None
            if rec["matches_receipt"] is False:
                inv["mismatch"].append(str(p))
            arm["files"][p.name] = rec
            (inv["kept"] if p.name in KEEP else plan).append(str(p))
        inv["arms"][b["cell"]] = arm
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
    print(f"[prune] kept {len(inv['kept'])} files, removed {len(inv['removed'])} ({freed / 1024**3:.2f} GB), 0 mismatches -> {OUT}")


if __name__ == "__main__":
    main()
