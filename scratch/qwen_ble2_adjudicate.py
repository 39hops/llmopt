"""QWEN-BLE-FREEGEN-2 adjudicator: fail-closed row gate, then bars.

Independent of the screen driver's summary (which is never trusted):
counts are recomputed from the rows. Gate (all fail-closed):
  exactly 60 rows, 30 per cell (nothink, xhigh)
  unique (cell, id), id set exactly 0..29 per cell
  every row arm=BLe, runtime=qcuda_tower, greedy=true, one device
  no unexpected cells
Bars (PRE-REG QWEN-BLE-FREEGEN-2, verbatim):
  1 TERMINATION FIRE iff xhigh think_terminated count >= 1
  2 COMPETENCE  FIRE iff total correct across both cells >= 1

    .venv/bin/python scratch/qwen_ble2_adjudicate.py \
        [logs/qweneffort2/tower_rows_BLe.jsonl]
Writes logs/qweneffort2/ble2_adjudication.json (append-refused).
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

CELLS = ("nothink", "xhigh")
N_PER_CELL = 30


def gate_rows(rows):
    """Fail-closed row gate; returns the violation list (empty =
    pass). Never raises — the caller decides, so fixtures can assert
    on the specific violation."""
    v = []
    if len(rows) != N_PER_CELL * len(CELLS):
        v.append(f"row count {len(rows)} != {N_PER_CELL*len(CELLS)}")
    seen = set()
    for r in rows:
        key = (r.get("cell"), r.get("id"))
        if key in seen:
            v.append(f"duplicate {key}")
        seen.add(key)
        if r.get("cell") not in CELLS:
            v.append(f"unexpected cell {r.get('cell')!r}")
        if r.get("arm") != "BLe":
            v.append(f"arm {r.get('arm')!r} != BLe")
        if r.get("runtime") != "qcuda_tower":
            v.append(f"runtime {r.get('runtime')!r} != qcuda_tower")
        if r.get("greedy") is not True:
            v.append(f"non-greedy row {key}")
    for cell in CELLS:
        ids = sorted(r["id"] for r in rows if r.get("cell") == cell)
        if ids != list(range(N_PER_CELL)):
            v.append(f"{cell} id set != 0..{N_PER_CELL-1}: "
                     f"n={len(ids)}")
    devices = {r.get("device_actual") for r in rows}
    if len(devices) != 1:
        v.append(f"multiple devices {devices}")
    return v


def adjudicate(rows):
    """Recomputed counts + bar outcomes. Only call after gate passes."""
    xterm = sum(1 for r in rows
                if r["cell"] == "xhigh" and r["think_terminated"])
    correct = sum(1 for r in rows if r["correct"])
    per_cell = {c: {"correct": sum(1 for r in rows
                                   if r["cell"] == c and r["correct"]),
                    "terminated": sum(1 for r in rows
                                      if r["cell"] == c
                                      and r["think_terminated"]),
                    "truncated": sum(1 for r in rows
                                     if r["cell"] == c
                                     and r["truncated"])}
                for c in CELLS}
    return {"xhigh_terminated": xterm, "total_correct": correct,
            "per_cell": per_cell,
            "bar1_termination": "FIRE" if xterm >= 1 else "NO-FIRE",
            "bar2_competence": "FIRE" if correct >= 1 else "NO-FIRE"}


def main():
    rows_path = sys.argv[1] if len(sys.argv) > 1 else \
        "logs/qweneffort2/tower_rows_BLe.jsonl"
    out = "logs/qweneffort2/ble2_adjudication.json"
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    raw = open(rows_path, "rb").read()
    rows = [json.loads(line) for line in raw.splitlines() if line]
    violations = gate_rows(rows)
    if violations:
        raise SystemExit("ROW GATE FAILED (bars unadjudicated):\n  "
                         + "\n  ".join(violations[:20]))
    res = adjudicate(rows)
    res["rows_sha256"] = hashlib.sha256(raw).hexdigest()
    res["rows_path"] = rows_path
    res["n_rows"] = len(rows)
    res["start"] = start_provenance(
        ["scratch/qwen_ble2_adjudicate.py",
         "scratch/qwen_effort_tower.py"])
    res["completion_commit"] = completion_commit()
    with open(out, "w") as f:
        f.write(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in
                      ("xhigh_terminated", "total_correct",
                       "bar1_termination", "bar2_competence",
                       "per_cell")}, indent=1))
    print(f"[ba] -> {out}")


if __name__ == "__main__":
    main()
