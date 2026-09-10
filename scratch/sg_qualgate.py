"""SYNTHETIC-GRADIENT-WRITER-1 qualification gates and the sealed ladder
law (AMENDMENT -SEAL). Reads logs/sgwriter1/qual.jsonl, gates every
finished ungated birth with llmopt.lab.gate.gate_eval on mps (the standard
120), appends kind=gate rows, and applies the pure law adjudicate():

  control c   = gate of the seed-27 CONTROL (MODE=zero K_BP=4).
  ADEQUATE    iff the control finished finite and c >= FLOOR (24); else
                NOT-RESOLVABLE-CONTROL and no SG cell is adjudicated.
  ladder      the frozen order LADDER = (linear 3e-4, linear 3e-5,
                mlp256 3e-4, mlp256 3e-5); a cell is FUNCTION-MATCH iff it
                finished finite and c - BAND <= g <= c + BAND (BAND 7).
  FIRST MATCH the first cell in the order that is FUNCTION-MATCH is the
                selection; every later cell is NOT BORN (stop: true).
  no match    when every cell has run without a match: ACCESSIBILITY-ONLY
                (each cell's gate booked; STABLE iff finite and g > 0;
                FLOOR g >= 24 descriptive); no discovery.
  in progress cells not yet run: stop false, next = the next cell.

Writes logs/sgwriter1/ladder.json (rewritten on every call: the current
ladder state) and, once, logs/sgwriter1/selection.json (refuses to
overwrite) when the verdict is final (FUNCTION-MATCH / ACCESSIBILITY-ONLY /
NOT-RESOLVABLE-CONTROL). Exit 0 in every regular state; the driver reads
ladder.json 'stop'. Pair integrity asserted before any gate: every cell
shares the control's init_state_digest, stream digests and steps.

SMOKE=1: reads logs/sgwriter1/smoke.jsonl (seed 11), gates on mps too,
writes logs/sgwriter1/smoke_ladder.json / smoke_selection.json only.
Usage: .venv/bin/python scratch/sg_qualgate.py
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
os.environ.setdefault("ARM", "off")
os.environ.setdefault("BIRTH_SEED", "0")

SMOKE = os.environ.get("SMOKE", "0") == "1"
BIRTHS = Path("logs/sgwriter1/smoke.jsonl" if SMOKE else "logs/sgwriter1/qual.jsonl")
LADDER_OUT = Path("logs/sgwriter1/smoke_ladder.json" if SMOKE else "logs/sgwriter1/ladder.json")
SELECTION = Path("logs/sgwriter1/smoke_selection.json" if SMOKE else "logs/sgwriter1/selection.json")
SEED = 11 if SMOKE else 27
FLOOR = 24
BAND = 7
LADDER = (("linear", 3e-4), ("linear", 3e-5), ("mlp256", 3e-4), ("mlp256", 3e-5))


def adjudicate(control, cells):
    """control: None (not run) or {'finite': bool, 'gate': int|None}.
    cells: {(family, plr): None (not run) | {'finite': bool, 'gate': int|None}}.
    Returns the ladder state dict (pure; no I/O)."""
    st = {"floor": FLOOR, "band": BAND, "ladder": [list(c) for c in LADDER], "control": control, "cells": {}}
    if control is None:
        st.update({"verdict": None, "stop": False, "next": "control", "c": None, "control_adequate": None})
        return st
    c = control["gate"] if control["finite"] else None
    adequate = bool(control["finite"] and c is not None and c >= FLOOR)
    st.update({"c": c, "control_adequate": adequate})
    if not adequate:
        st.update({"verdict": "NOT-RESOLVABLE-CONTROL", "stop": True, "next": None, "selected": None})
        return st
    for fam, plr in LADDER:
        key = f"{fam}:{plr:g}"
        r = cells.get((fam, plr))
        if r is None:
            st["cells"][key] = None
            st.update({"verdict": None, "stop": False, "next": key, "selected": None})
            return st
        g = r["gate"] if r["finite"] else None
        match = bool(r["finite"] and g is not None and c - BAND <= g <= c + BAND)
        st["cells"][key] = {"finite": r["finite"], "gate": g, "stable": bool(r["finite"] and g is not None and g > 0),
                            "floor": bool(g is not None and g >= FLOOR), "function_match": match, "delta_v_control": (None if g is None else g - c)}
        if match:
            st.update({"verdict": "FUNCTION-MATCH", "stop": True, "next": None, "selected": {"family": fam, "plr": plr, "gate": g, "c": c}})
            return st
    st.update({"verdict": "ACCESSIBILITY-ONLY", "stop": True, "next": None, "selected": None})
    return st


def main():
    import torch
    import train_mathnative as TM
    from atomtraj_pins import state_digest
    from llmopt.lab.gate import gate_eval

    rows = [json.loads(l) for l in BIRTHS.open()] if BIRTHS.exists() else []
    births = [b for b in rows if b.get("kind") == "birth" and b["seed"] == SEED]
    cells = {}
    control = None
    for b in births:
        key = ("control",) if b["mode"] == "zero" else (b["family"], float(b["plr"]))
        assert key not in cells, f"duplicate birth for {key}: refusing"
        cells[key] = b
    ctrl_b = cells.pop(("control",), None)
    if ctrl_b is not None:
        for b in cells.values():
            assert b["init_state_digest"] == ctrl_b["init_state_digest"], f"{b['cell']}: W_0 differs from the control"
            assert b["stream_sha256"] == ctrl_b["stream_sha256"] and b["steps_total"] == ctrl_b["steps_total"], f"{b['cell']}: stream / steps differ"
            assert b["k_bp"] == ctrl_b["k_bp"] == 4 and b["frozen_tensors"] == ctrl_b["frozen_tensors"], f"{b['cell']}: frozen set differs"
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    if not SMOKE and dirty:
        raise SystemExit("REFUSING: registered gating on a dirty tree")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    assert dev == "mps", "SYNTHETIC-GRADIENT-WRITER-1 gates are sealed on mps"
    tok = TM.MathTokenizer()
    gated = {r["cell"]: r for r in rows if r.get("kind") == "gate"}

    def gate_of(b):
        if b["cell"] in gated:
            return gated[b["cell"]]
        if not b["stable_training"]:
            return None
        p = Path(b["outdir"]) / "final.pt"
        sd = torch.load(p, map_location="cpu")
        assert state_digest(sd) == b["final"]["state_digest"], f"{p}: digest v receipt"
        m = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
        m.load_state_dict(sd)
        m = m.to(dev).eval()
        t0 = time.time()
        solves, valid = gate_eval(m, tok, dev)
        g = {"kind": "gate", "phase": b["phase"], "cell": b["cell"], "seed": b["seed"], "arm": b["arm"], "family": b.get("family"), "plr": b.get("plr"),
             "solves": {str(k): int(v) for k, v in solves.items()}, "total": int(sum(solves.values())), "valid_pct": round(float(valid), 2),
             "device": dev, "wall_s": round(time.time() - t0, 1), "state_digest": b["final"]["state_digest"], "final_file_sha256": b["final"]["file_sha256"],
             "outdir": b["outdir"], "code_commit": commit, "tree_dirty": dirty, "gated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
        with BIRTHS.open("a") as f:
            f.write(json.dumps(g) + "\n")
        gated[b["cell"]] = g
        print(f"[sgq] {b['cell']}: gate {g['total']} valid {g['valid_pct']}", flush=True)
        return g

    def rec_of(b):
        if b is None:
            return None
        if not b["stable_training"]:
            return {"finite": False, "gate": None, "cell": b["cell"]}
        g = gate_of(b)
        return {"finite": True, "gate": g["total"], "cell": b["cell"], "solves": g["solves"], "valid_pct": g["valid_pct"]}

    control = rec_of(ctrl_b)
    cell_recs = {k: rec_of(v) for k, v in cells.items()}
    st = adjudicate(control, cell_recs)
    st.update({"prereg": "SYNTHETIC-GRADIENT-WRITER-1", "kind": "ladder", "seed": SEED, "commit": commit, "device": dev, "smoke": SMOKE,
               "cell_records": {f"{k[0]}:{k[1]:g}": v for k, v in cell_recs.items()},
               "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
    LADDER_OUT.write_text(json.dumps(st, indent=1))
    if st["verdict"] is not None:
        if SELECTION.exists():
            raise SystemExit(f"REFUSING: {SELECTION} exists")
        SELECTION.write_text(json.dumps(st, indent=1))
        with BIRTHS.open("a") as f:
            f.write(json.dumps({k: v for k, v in st.items() if k not in ("cell_records",)}) + "\n")
    print("[sgq] ladder:", json.dumps({"c": st["c"], "control_adequate": st["control_adequate"], "verdict": st["verdict"], "stop": st["stop"], "next": st["next"],
                                       "selected": st.get("selected")}), flush=True)


if __name__ == "__main__":
    main()
