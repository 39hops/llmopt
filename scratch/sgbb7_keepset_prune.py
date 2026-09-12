"""Housekeeping prune of checkpoints/sgbb7/ (Artin GO 2026-09-11 22:34 EDT,
after VERDICT SG-BOUNDARY-BLOCK7-1): KEEP for BOTH cells step_00000 (W_0),
step_00463, step_03084, step_05140, step_15420 and final.pt, plus the SG7
cell's matching pred_step_ files and pred_final.pt; REMOVE the other 12
model snapshots per cell and the other predictor snapshots. Every file is
sha256- and state-digest-matched against logs/sgbb7/qual.jsonl before
anything is removed; any mismatch or unreceipted file ABORTS. Inventory to
logs/housekeeping/sgbb7_keepset_prune_<date>.json (refuses to overwrite).
Usage: .venv/bin/python scratch/sgbb7_keepset_prune.py
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

TREE = Path("checkpoints/sgbb7")
RECEIPT = Path("logs/sgbb7/qual.jsonl")
KEEP_STEPS = {0, 463, 3084, 5140, 15420}
CONTROL = "sgb7_control_s28_lr0.0003"
OUT = Path(f"logs/housekeeping/sgbb7_keepset_prune_{datetime.date.today().isoformat()}.json")


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
    if n in ("final.pt", "pred_final.pt"):
        return True
    if n.startswith("step_") or n.startswith("pred_step_"):
        step = int(n[5:10]) if n.startswith("step_") else int(n[10:15])
        return step in KEEP_STEPS
    return False


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    births = [json.loads(l) for l in RECEIPT.open() if '"kind": "birth"' in l]
    by_dir = {b["outdir"]: b for b in births}
    inv = {"kind": "sgbb7_keepset_prune", "direction": "Artin GO 2026-09-11 22:34 EDT (after VERDICT SG-BOUNDARY-BLOCK7-1)", "keep_steps": sorted(KEEP_STEPS),
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
