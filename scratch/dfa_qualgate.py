"""WRITER-DFA-1 qualification gates and the frozen selection (PRE-REG
L68321 item 4, S5 of L68543, P5f of L68644). Reads the four QUAL birth
rows in logs/writerdfa1/qual.jsonl, gates every cell that finished
training (final.pt) with llmopt.lab.gate.gate_eval on mps (the standard
120), appends kind=gate rows to qual.jsonl, and writes
logs/writerdfa1/qual_selection.json (refuses to overwrite): a cell is
STABLE iff it finished with a finite loss and its final gate > 0;
selection = the highest stable final gate, ties broken by lr 3e-4
before 1e-4, then s = 1, 0.25, 4; no stable cell -> selected null,
stop = "DFA-UNSTABLE". Nothing about seed 2 is read here.

Usage: .venv/bin/python scratch/dfa_qualgate.py
"""
import datetime
import json
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

QUAL = Path("logs/writerdfa1/qual.jsonl")
SEL = Path("logs/writerdfa1/qual_selection.json")
CELLS = [(1.0, 3e-4), (1.0, 1e-4), (0.25, 3e-4), (4.0, 1e-4)]
TIE_ORDER = {(1.0, 3e-4): 0, (0.25, 3e-4): 1, (1.0, 1e-4): 2, (4.0, 1e-4): 3}   # lr 3e-4 first, then s = 1, 0.25, 4


def main():
    if SEL.exists():
        raise SystemExit(f"REFUSING: {SEL} exists")
    rows = [json.loads(l) for l in QUAL.open()]
    births = {(float(b["s"]), float(b["peak_lr"])): b for b in rows if b.get("kind") == "birth" and b["phase"] == "qual"}
    missing = [c for c in CELLS if c not in births]
    assert not missing, f"qualification births missing: {missing}"
    assert len([b for b in rows if b.get("kind") == "birth"]) == 4, "expected exactly four qualification birth rows"
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    tok = TM.MathTokenizer()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    results = {}
    for cell in CELLS:
        b = births[cell]
        if b.get("final") is None:
            results[cell] = {"trained": False, "nonfinite_step": b.get("nonfinite_step"), "gate": None, "stable": False}
            print(f"[qual] {b['cell']}: UNSTABLE (non-finite at step {b.get('nonfinite_step')})", flush=True)
            continue
        p = Path(b["outdir"]) / "final.pt"
        sd = torch.load(p, map_location="cpu")
        assert state_digest(sd) == b["final"]["state_digest"], f"{p}: digest v receipt"
        m = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
        m.load_state_dict(sd)
        m = m.to(dev).eval()
        t0 = time.time()
        solves, valid = gate_eval(m, tok, dev)
        tot = int(sum(solves.values()))
        row = {"kind": "gate", "phase": "qual", "cell": b["cell"], "s": cell[0], "peak_lr": cell[1], "solves": {str(k): int(v) for k, v in solves.items()},
               "total": tot, "valid_pct": round(float(valid), 2), "device": dev, "wall_s": round(time.time() - t0, 1),
               "state_digest": b["final"]["state_digest"], "code_commit": commit,
               "gated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
        with QUAL.open("a") as f:
            f.write(json.dumps(row) + "\n")
        results[cell] = {"trained": True, "gate": tot, "solves": row["solves"], "stable": tot > 0, "final_loss": b.get("final_loss")}
        print(f"[qual] {b['cell']}: gate {tot}/120 ({row['wall_s']}s)", flush=True)
        del m
    stable = [c for c in CELLS if results[c]["stable"]]
    selected = None
    if stable:
        best = max(results[c]["gate"] for c in stable)
        tied = [c for c in stable if results[c]["gate"] == best]
        c = min(tied, key=lambda c: TIE_ORDER[c])
        selected = {"s": c[0], "lr": c[1], "gate": best, "cell": births[c]["cell"], "tied": [list(t) for t in tied]}
    rec = {"prereg": "WRITER-DFA-1", "kind": "selection", "commit": commit, "device": dev,
           "cells": {f"S{c[0]:g}_lr{c[1]:g}": results[c] for c in CELLS}, "stable_cells": [list(c) for c in stable],
           "selected": selected, "stop": None if selected else "DFA-UNSTABLE",
           "law": "highest stable final gate; ties lr 3e-4 before 1e-4, then s = 1, 0.25, 4; stable = finite loss and gate > 0",
           "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    SEL.write_text(json.dumps(rec, indent=1))
    with QUAL.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print("[qual] selection:", json.dumps(selected), "stop:", rec["stop"])


if __name__ == "__main__":
    main()
