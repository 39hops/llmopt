"""CREDIT-ANCHOR-FRONTIER-1 qualification gates and the frozen selection
(PRE-REG L69122 item 2, AMENDMENT -PRECISION L69223 F3). Reads the QUAL
birth rows in logs/writercaf1/qual.jsonl (hybrid cells and zero-credit
controls, seed 23), gates every finished arm with gate_eval on mps (120),
appends kind=gate rows, and writes logs/writercaf1/qual_selection.json
(refuses to overwrite): per k the hybrid cells (1, 3e-4) and (1, 1e-4);
STABLE iff final gate > 0; FLOOR iff final gate >= 24; the ladder's
candidate = the smallest k with a hybrid cell at or above the floor,
its best cell (highest gate, ties lr 3e-4 first); zero-credit controls
are gated and reported descriptively and never enter the selection.
The script gates whatever k values have both hybrid cells present; the
caller decides (per the ladder) whether the next k is born. Selection is
written ONLY when a candidate exists or every k is complete without one
(FRONTIER-CLOSED); an incomplete frontier exits 3 with no artifact.

Usage: .venv/bin/python scratch/caf_qualgate.py            (gate + select; refuses if qual_selection.json exists)
       GATE_ONLY=1 .venv/bin/python scratch/caf_qualgate.py (gate new arms, no selection file)
"""
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

import torch  # noqa: E402

import train_mathnative as TM  # noqa: E402
from atomtraj_pins import state_digest  # noqa: E402
from llmopt.lab.gate import gate_eval  # noqa: E402

QUAL = Path(os.environ.get("CAF_QUAL_PATH", "logs/writercaf1/qual.jsonl"))       # override: unit tests only
SEL = Path(os.environ.get("CAF_SEL_PATH", "logs/writercaf1/qual_selection.json"))
CELLS = [(1.0, 3e-4), (1.0, 1e-4)]
FLOOR = 24
KS = (1, 2, 4)
GATE_ONLY = os.environ.get("GATE_ONLY") == "1"


def main():
    if SEL.exists() and not GATE_ONLY:
        raise SystemExit(f"REFUSING: {SEL} exists")
    if not QUAL.exists():
        raise SystemExit(f"NOT-RUN: {QUAL} absent (no qualification birth yet)")
    rows = [json.loads(l) for l in QUAL.open()]
    births = [b for b in rows if b.get("kind") == "birth" and b["phase"] == "qual"]
    gated = {r["cell"] for r in rows if r.get("kind") == "gate" and r.get("phase") == "qual"}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    tree_dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    paths_overridden = bool(os.environ.get("CAF_QUAL_PATH") or os.environ.get("CAF_SEL_PATH"))
    if tree_dirty and not paths_overridden:
        raise SystemExit("REFUSING: registered gating on a dirty tree")
    tok = TM.MathTokenizer()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    gates = {r["cell"]: r for r in rows if r.get("kind") == "gate" and r.get("phase") == "qual"}
    for b in births:
        if b["cell"] in gated:
            continue
        if b.get("final") is None:
            row = {"kind": "gate", "phase": "qual", "cell": b["cell"], "mode": b["mode"], "k_bp": b["k_bp"], "s": b["s"], "peak_lr": b["peak_lr"],
                   "trained": False, "nonfinite_step": b.get("nonfinite_step"), "total": None, "code_commit": commit,
                   "gated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
        else:
            p = Path(b["outdir"]) / "final.pt"
            sd = torch.load(p, map_location="cpu")
            assert state_digest(sd) == b["final"]["state_digest"], f"{p}: digest v receipt"
            m = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
            m.load_state_dict(sd)
            m = m.to(dev).eval()
            t0 = time.time()
            solves, valid = gate_eval(m, tok, dev)
            row = {"kind": "gate", "phase": "qual", "cell": b["cell"], "mode": b["mode"], "k_bp": b["k_bp"], "s": b["s"], "peak_lr": b["peak_lr"], "trained": True,
                   "solves": {str(k): int(v) for k, v in solves.items()}, "total": int(sum(solves.values())), "valid_pct": round(float(valid), 2),
                   "device": dev, "wall_s": round(time.time() - t0, 1), "state_digest": b["final"]["state_digest"], "final_file_sha256": b["final"]["file_sha256"],
                   "outdir": b["outdir"], "code_commit": commit, "tree_dirty": tree_dirty, "qual_path": str(QUAL), "paths_overridden": paths_overridden,
                   "gated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
            del m
        with QUAL.open("a") as f:
            f.write(json.dumps(row) + "\n")
        gates[b["cell"]] = row
        print(f"[caf-qual] {b['cell']}: gate {row['total']}", flush=True)
    if GATE_ONLY:
        return
    per_k = {}
    for k in KS:
        hyb = {c: g for c, g in gates.items() if g["mode"] == "hybrid" and g["k_bp"] == k}
        zero = [g for g in gates.values() if g["mode"] == "zero" and g["k_bp"] == k]
        if not hyb:
            continue
        cells = {(g["s"], g["peak_lr"]): g for g in hyb.values()}
        stable = [c for c in CELLS if c in cells and cells[c]["total"] is not None and cells[c]["total"] > 0]
        floor = [c for c in CELLS if c in cells and cells[c]["total"] is not None and cells[c]["total"] >= FLOOR]
        per_k[str(k)] = {"cells": {f"S{c[0]:g}_lr{c[1]:g}": {"gate": cells[c]["total"], "solves": cells[c].get("solves")} for c in CELLS if c in cells},
                         "complete": all(c in cells for c in CELLS), "stable": [list(c) for c in stable], "floor": [list(c) for c in floor],
                         "zero_credit": [{"gate": z["total"], "solves": z.get("solves"), "cell": z["cell"]} for z in zero],
                         "zero_clears_floor": any(z["total"] is not None and z["total"] >= FLOOR for z in zero),
                         "hybrid_clears_floor": bool(floor)}
    selected = None
    for k in KS:
        pk = per_k.get(str(k))
        if pk and pk["floor"]:
            smaller = [kk for kk in KS if kk < k]
            if not all(per_k.get(str(kk), {}).get("complete") for kk in smaller):
                # a larger k cleared while a smaller k is incomplete: the ladder order was violated; no selection
                print(f"[caf-qual] INCOMPLETE: k={k} clears the floor but a smaller k is not complete ({[kk for kk in smaller if not per_k.get(str(kk), {}).get('complete')]}); qual_selection.json NOT written", flush=True)
                sys.exit(3)
            cells = {(g["s"], g["peak_lr"]): g for g in gates.values() if g["mode"] == "hybrid" and g["k_bp"] == k}
            best = max(cells[tuple(c)]["total"] for c in pk["floor"])
            tied = [tuple(c) for c in pk["floor"] if cells[tuple(c)]["total"] == best]
            c = min(tied, key=lambda c: (0 if c[1] == 3e-4 else 1))
            selected = {"k": k, "s": c[0], "lr": c[1], "gate": best, "cell": cells[c]["cell"]}
            break
    complete_ks = [k for k in KS if per_k.get(str(k), {}).get("complete")]
    frontier_closed = (selected is None and complete_ks == list(KS))
    if selected is None and not frontier_closed:
        # incomplete frontier: no candidate yet and not every k complete -> no selection artifact (Artin fold 2026-09-09)
        print(f"[caf-qual] INCOMPLETE: no hybrid cell at or above {FLOOR} yet and ks complete = {complete_ks}; qual_selection.json NOT written", flush=True)
        sys.exit(3)
    rec = {"prereg": "CREDIT-ANCHOR-FRONTIER-1", "kind": "selection", "commit": commit, "tree_dirty": tree_dirty, "device": dev, "floor": FLOOR, "per_k": per_k,
           "qual_path": str(QUAL), "sel_path": str(SEL), "paths_overridden": paths_overridden,
           "selected": selected, "frontier_closed": frontier_closed, "ks_complete": complete_ks,
           "law": "smallest k with a hybrid cell >= 24; best cell = highest gate, ties lr 3e-4 first; zero-credit controls descriptive only",
           "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    SEL.write_text(json.dumps(rec, indent=1))
    with QUAL.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print("[caf-qual] selection:", json.dumps(selected), "frontier_closed:", frontier_closed)


if __name__ == "__main__":
    main()
