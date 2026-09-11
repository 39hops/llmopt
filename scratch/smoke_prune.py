"""Housekeeping prune of the three SMOKE checkpoint trees (Artin GO
2026-09-11, after SG-FAILURE-DESK-0): checkpoints/frozenbb1_smoke/,
checkpoints/sgwriter1_smoke/, checkpoints/sgwriter1_smoke_seal2/. For every
file under each tree the sha256 (and, for step_/final/pred_ files, the
canonical state digest) is re-read and matched against the smoke receipt
that booked it (logs/frozenbb1/smoke.jsonl, logs/sgwriter1/smoke.jsonl,
logs/sgwriter1/smoke_seal2.jsonl); the inventory is written to
logs/housekeeping/smoke_prune_inventory_<date>.json (refuses to overwrite);
any mismatch or any file without a receipt row ABORTS before anything is
removed; on 0 mismatches every file and the trees are removed. The
registered checkpoints/sgwriter1/ and checkpoints/frozenbb1/ keep sets are
never touched. Usage: .venv/bin/python scratch/smoke_prune.py
"""
import datetime
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import torch  # noqa: E402

from atomtraj_pins import state_digest  # noqa: E402

TREES = {"checkpoints/frozenbb1_smoke": "logs/frozenbb1/smoke.jsonl",
         "checkpoints/sgwriter1_smoke": "logs/sgwriter1/smoke.jsonl",
         "checkpoints/sgwriter1_smoke_seal2": "logs/sgwriter1/smoke_seal2.jsonl"}
OUT = Path(f"logs/housekeeping/smoke_prune_inventory_{datetime.date.today().isoformat()}.json")


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


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    inv = {"kind": "smoke_prune", "direction": "Artin GO 2026-09-11 (housekeeping after SG-FAILURE-DESK-0)", "trees": {}, "mismatch": [], "unreceipted": [], "removed": [], "bytes_freed": 0}
    plan = []
    for tree, receipt in TREES.items():
        births = [json.loads(l) for l in open(receipt) if '"kind": "birth"' in l]
        by_dir = {b["outdir"]: b for b in births}
        trec = {"receipt": receipt, "files": {}}
        root = Path(tree)
        if not root.exists():
            trec["absent"] = True
            inv["trees"][tree] = trec
            continue
        for p in sorted(root.rglob("*")):
            if p.is_dir():
                continue
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            rec = {"sha256": sha, "bytes": p.stat().st_size}
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
            trec["files"][str(p)] = rec
            plan.append(p)
        inv["trees"][tree] = trec
    if inv["mismatch"] or inv["unreceipted"]:
        OUT.write_text(json.dumps(inv, indent=1))
        raise SystemExit(f"ABORT: mismatches {inv['mismatch']} unreceipted {inv['unreceipted']}; nothing removed")
    for p in plan:
        inv["bytes_freed"] += p.stat().st_size
        os.remove(p)
        inv["removed"].append(str(p))
    for tree in TREES:
        if Path(tree).exists():
            shutil.rmtree(tree)
    inv["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    OUT.write_text(json.dumps(inv, indent=1))
    print(f"[smoke-prune] removed {len(inv['removed'])} files ({inv['bytes_freed'] / 1024**3:.2f} GiB), 0 mismatches, 0 unreceipted -> {OUT}")


if __name__ == "__main__":
    main()
