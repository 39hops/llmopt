"""Housekeeping prune of checkpoints/sgwriter1/ to the compact keep set
proposed in OBSERVATION SG-FAILURE-DESK-0 and approved with the
SG-CROSSPOS-REPRESENTABILITY-0 GO (executes only after that desk is booked
and receipt-complete). KEEP per cell: step_00463, step_03084, step_05140,
step_15420 and the matching pred_step_ files; the control's step_00000
(the shared W_0). REMOVE everything else (the other 12 snapshots per cell,
final.pt / pred_final.pt whose state digests equal step_15420's, the
remaining pred_step_ files). Every file under the tree is sha256- and
state-digest-matched against logs/sgwriter1/qual.jsonl before anything is
removed; any mismatch or unreceipted file ABORTS. The inventory (every
file, kept or removed, with its digests) is written to
logs/housekeeping/sgwriter1_keepset_prune_<date>.json (refuses to
overwrite). Usage: .venv/bin/python scratch/sgwriter1_keepset_prune.py
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import torch  # noqa: E402

from atomtraj_pins import state_digest  # noqa: E402

TREE = Path("checkpoints/sgwriter1")
RECEIPT = Path("logs/sgwriter1/qual.jsonl")
KEEP_STEPS = {463, 3084, 5140, 15420}
CONTROL = "sgq_control_s27_lr0.0003"
OUT = Path(f"logs/housekeeping/sgwriter1_keepset_prune_{datetime.date.today().isoformat()}.json")


def expected(b, name):
    if name.startswith("step_"):
        return b["snapshots"].get(str(int(name[5:10])))
    if name == "final.pt":
        return b.get("final")
    if name.startswith("pred_step_"):
        return b.get("pred_snapshots", {}).get(str(int(name[10:15])))
    if name == "pred_final.pt":
        return b.get("pred_final")
    return None


def keep(p):
    n = p.name
    if n.startswith("step_") or n.startswith("pred_step_"):
        step = int(n[5:10]) if n.startswith("step_") else int(n[10:15])
        return step in KEEP_STEPS or (step == 0 and p.parent.name == CONTROL and n.startswith("step_"))
    return False


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    births = [json.loads(l) for l in RECEIPT.open() if '"kind": "birth"' in l]
    by_dir = {b["outdir"]: b for b in births}
    inv = {"kind": "sgwriter1_keepset_prune", "direction": "Artin GO 2026-09-11 (after SG-CROSSPOS-REPRESENTABILITY-0 is booked)", "keep_steps": sorted(KEEP_STEPS),
           "files": {}, "mismatch": [], "unreceipted": [], "removed": [], "kept": [], "bytes_freed": 0, "bytes_kept": 0}
    for p in sorted(TREE.rglob("*")):
        if p.is_dir():
            continue
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        rec = {"sha256": sha, "bytes": p.stat().st_size, "keep": keep(p)}
        b = by_dir.get(str(p.parent))
        exp = expected(b, p.name) if b else None
        if exp is None:
            rec["matches_receipt"] = None
            inv["unreceipted"].append(str(p))
        else:
            rec["state_digest"] = state_digest(torch.load(p, map_location="cpu"))
            rec["matches_receipt"] = (sha == exp["file_sha256"] and rec["state_digest"] == exp["state_digest"])
            if not rec["matches_receipt"]:
                inv["mismatch"].append(str(p))
        inv["files"][str(p)] = rec
    if inv["mismatch"] or inv["unreceipted"]:
        OUT.write_text(json.dumps(inv, indent=1))
        raise SystemExit(f"ABORT: mismatch {inv['mismatch']} unreceipted {inv['unreceipted']} (inventory written, nothing removed)")
    for path, rec in inv["files"].items():
        if rec["keep"]:
            inv["kept"].append(path)
            inv["bytes_kept"] += rec["bytes"]
        else:
            Path(path).unlink()
            inv["removed"].append(path)
            inv["bytes_freed"] += rec["bytes"]
    inv["n_removed"], inv["n_kept"] = len(inv["removed"]), len(inv["kept"])
    OUT.write_text(json.dumps(inv, indent=1))
    print(f"kept {inv['n_kept']} files ({inv['bytes_kept'] / 2**30:.2f} GiB), removed {inv['n_removed']} ({inv['bytes_freed'] / 2**30:.2f} GiB) -> {OUT}")


if __name__ == "__main__":
    main()
