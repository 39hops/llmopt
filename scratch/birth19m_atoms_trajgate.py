"""ATOM-DIET-TRAJECTORY-1 post-hoc gate runner (pre-reg RESULTS L66546).
Runs only after ALL registered trainings are complete: loads the saved
snapshots at the frozen gate steps (2,056 / 5,140 / 10,280 / 15,420),
checks each file's canonical state_digest against the birth receipt,
runs the standard 120 gate (llmopt.lab.gate.gate_eval) and appends one
row per gated snapshot. No training state is touched. Real mode refuses
to run with fewer than six birth rows and refuses if gates.jsonl exists.

Env: SMOKE=1 gates the smoke birth rows in logs/atomtraj1/smoke.jsonl
(snapshots at the gate steps that exist plus the capped final) and
appends kind=gate rows to the same file.

Usage: .venv/bin/python scratch/birth19m_atoms_trajgate.py
"""
import datetime
import hashlib
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

SMOKE = os.environ.get("SMOKE") == "1"
GATE_STEPS = [2_056, 5_140, 10_280, 15_420]
BIRTHS = Path("logs/atomtraj1/smoke.jsonl" if SMOKE else "logs/atomtraj1/births.jsonl")
GATES = Path("logs/atomtraj1/smoke.jsonl" if SMOKE else "logs/atomtraj1/gates.jsonl")
EXPECTED_BIRTHS = 6


def main():
    rows = [json.loads(l) for l in BIRTHS.open()]
    births = [r for r in rows if r.get("kind") == "birth" and r.get("emit") and r.get("device") != "cpu"]
    if SMOKE:
        births = births[-1:]
        assert births, "no smoke birth row"
    else:
        assert not GATES.exists(), f"REFUSING: {GATES} exists"
        assert len(births) == EXPECTED_BIRTHS, f"need {EXPECTED_BIRTHS} birth rows, have {len(births)}"
        assert all(not r["smoke"] for r in births)
        assert sorted((r["arm"], r["seed"]) for r in births) == sorted((a, s) for a in ("stock", "atoms") for s in (5, 6, 7))
    tok = TM.MathTokenizer()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    for r in births:
        outdir = Path(r["outdir"])
        steps = [s for s in GATE_STEPS if str(s) in r["snapshots"]]
        if SMOKE and str(r["steps"]) not in map(str, steps):
            steps.append(r["steps"])
        for s in steps:
            p = outdir / f"step_{s:05d}.pt"
            sd = torch.load(p, map_location="cpu")
            fsha = hashlib.sha256(p.read_bytes()).hexdigest()
            sdig = state_digest(sd)
            rec = r["snapshots"][str(s)]
            assert fsha == rec["file_sha256"] and sdig == rec["state_digest"], f"snapshot drift {p}"
            model = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
            model.load_state_dict(sd)
            model = model.to(dev).eval()
            t0 = time.time()
            solves, valid = gate_eval(model, tok, dev)
            row = {"kind": "gate", "arm": r["arm"], "seed": r["seed"], "step": s, "smoke": SMOKE,
                   "solves": {str(k): int(v) for k, v in solves.items()}, "total": int(sum(solves.values())),
                   "valid_pct": round(float(valid), 2), "device": dev, "wall_s": round(time.time() - t0, 1),
                   "file_sha256": fsha, "state_digest": sdig, "birth_code_commit": r["code_commit"],
                   "code_commit": commit, "gated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
            with GATES.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[trajgate] {r['arm']} s{r['seed']} step {s}: {row['solves']} = {row['total']}/120 @ {row['valid_pct']}% ({row['wall_s']}s)", flush=True)
            del model


if __name__ == "__main__":
    main()
