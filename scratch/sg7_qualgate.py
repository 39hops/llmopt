"""SG-BOUNDARY-BLOCK7-1 qualification gate and law (PRE-REG
SG-BOUNDARY-BLOCK7-1). Reads logs/sgbb7/qual.jsonl, gates every finished
ungated birth with llmopt.lab.gate.gate_eval on mps (the standard 120),
appends kind=gate rows, and applies the pure law adjudicate():

  control c        = gate of the seed-28 CONTROL (MODE=zero K_BP=4).
  CONTROL-ADEQUATE iff the control finished finite and c >= FLOOR (24);
                   else NOT-RESOLVABLE-CONTROL and the cell is not
                   adjudicated.
  FUNCTION-MATCH   iff the one SG7 cell finished finite and
                   c - BAND <= g_SG7 <= c + BAND (BAND 7).
  MISS             finite, outside the band: SG credit-writer births CLOSE
                   completely (the sealed consequence).
  UNSTABLE         the cell did not finish finite: no gate; booked as
                   SG7-UNSTABLE (closes likewise: the cell was born and
                   did not qualify).
  in progress      stop false, next = the missing arm.

Writes logs/sgbb7/ladder.json (rewritten on every call) and, once,
logs/sgbb7/selection.json (refuses to overwrite) when the verdict is
final. Pair integrity asserted before any gate: the cell shares the
control's init_state_digest, stream digests, steps and frozen set.
SMOKE=1: logs/sgbb7/smoke{SMOKE_TAG}.jsonl (seed 11), writes
smoke{SMOKE_TAG}_ladder.json / _selection.json only.
Usage: .venv/bin/python scratch/sg7_qualgate.py
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
SMOKE_TAG = os.environ.get("SMOKE_TAG", "")
assert not SMOKE_TAG or SMOKE, "SMOKE_TAG is smoke-only"
BIRTHS = Path(f"logs/sgbb7/smoke{SMOKE_TAG}.jsonl" if SMOKE else "logs/sgbb7/qual.jsonl")
LADDER_OUT = Path(f"logs/sgbb7/smoke{SMOKE_TAG}_ladder.json" if SMOKE else "logs/sgbb7/ladder.json")
SELECTION = Path(f"logs/sgbb7/smoke{SMOKE_TAG}_selection.json" if SMOKE else "logs/sgbb7/selection.json")
SEED = 11 if SMOKE else 28
FLOOR = 24
BAND = 7


def adjudicate(control, cell):
    """control / cell: None (not run) or {'finite': bool, 'gate': int|None}.
    Returns the state dict (pure; no I/O)."""
    st = {"floor": FLOOR, "band": BAND, "control": control, "cell": None}
    if control is None:
        st.update({"verdict": None, "stop": False, "next": "control", "c": None, "control_adequate": None})
        return st
    c = control["gate"] if control["finite"] else None
    adequate = bool(control["finite"] and c is not None and c >= FLOOR)
    st.update({"c": c, "control_adequate": adequate})
    if not adequate:
        st.update({"verdict": "NOT-RESOLVABLE-CONTROL", "stop": True, "next": None})
        return st
    if cell is None:
        st.update({"verdict": None, "stop": False, "next": "sg7"})
        return st
    g = cell["gate"] if cell["finite"] else None
    match = bool(cell["finite"] and g is not None and c - BAND <= g <= c + BAND)
    st["cell"] = {"finite": cell["finite"], "gate": g, "stable": bool(cell["finite"] and g is not None and g > 0),
                  "floor": bool(g is not None and g >= FLOOR), "function_match": match, "delta_v_control": (None if g is None else g - c)}
    if not cell["finite"]:
        st.update({"verdict": "SG7-UNSTABLE", "stop": True, "next": None})
    elif match:
        st.update({"verdict": "FUNCTION-MATCH", "stop": True, "next": None})
    else:
        st.update({"verdict": "SG7-MISS", "stop": True, "next": None})
    return st


def main():
    import torch
    import train_mathnative as TM
    from atomtraj_pins import state_digest
    from llmopt.lab.gate import gate_eval

    rows = [json.loads(l) for l in BIRTHS.open()] if BIRTHS.exists() else []
    births = [b for b in rows if b.get("kind") == "birth" and b["seed"] == SEED]
    ctrl_b, cell_b = None, None
    for b in births:
        if b["mode"] == "zero":
            assert ctrl_b is None, "duplicate control birth: refusing"
            ctrl_b = b
        else:
            assert b["mode"] == "sg7" and cell_b is None, "duplicate / foreign cell birth: refusing"
            cell_b = b
    if ctrl_b is not None and cell_b is not None:
        assert cell_b["init_state_digest"] == ctrl_b["init_state_digest"], "W_0 differs from the control"
        assert cell_b["stream_sha256"] == ctrl_b["stream_sha256"] and cell_b["steps_total"] == ctrl_b["steps_total"], "stream / steps differ"
        assert cell_b["k_bp"] == ctrl_b["k_bp"] == 4 and cell_b["frozen_tensors"] == ctrl_b["frozen_tensors"], "frozen set differs"
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    if not SMOKE and dirty:
        raise SystemExit("REFUSING: registered gating on a dirty tree")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    assert dev == "mps", "SG-BOUNDARY-BLOCK7-1 gates are sealed on mps"
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
        print(f"[sgb7] {b['cell']}: gate {g['total']} valid {g['valid_pct']}", flush=True)
        return g

    def rec_of(b):
        if b is None:
            return None
        if not b["stable_training"]:
            return {"finite": False, "gate": None, "cell": b["cell"]}
        g = gate_of(b)
        return {"finite": True, "gate": g["total"], "cell": b["cell"], "solves": g["solves"], "valid_pct": g["valid_pct"]}

    control = rec_of(ctrl_b)
    cell = rec_of(cell_b)
    st = adjudicate(control, cell)
    st.update({"prereg": "SG-BOUNDARY-BLOCK7-1", "kind": "ladder", "seed": SEED, "commit": commit, "device": dev, "smoke": SMOKE,
               "cell_record": cell, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
    LADDER_OUT.write_text(json.dumps(st, indent=1))
    if st["verdict"] is not None:
        if SELECTION.exists():
            raise SystemExit(f"REFUSING: {SELECTION} exists")
        SELECTION.write_text(json.dumps(st, indent=1))
        with BIRTHS.open("a") as f:
            f.write(json.dumps({k: v for k, v in st.items() if k != "cell_record"}) + "\n")
    print("[sgb7] law:", json.dumps({"c": st["c"], "control_adequate": st["control_adequate"], "verdict": st["verdict"], "stop": st["stop"], "next": st["next"], "cell": st["cell"]}), flush=True)


if __name__ == "__main__":
    main()
