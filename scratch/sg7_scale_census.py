"""SG-BOUNDARY-BLOCK7-1 desk census (zero training): the per-feature scale
of the LOCAL predictor's inputs x_8 and e = dL/dlogits across the retained
seed-27 frozen-BP control snapshots (digests asserted against
logs/sgwriter1/qual.jsonl), over the eligible tokens of the frozen probe's
FIT chunks 0, 2, 4, 6 (the same tokens the birth's W_0 input statistics
use). Receipts the drift that justifies dropping the desk's +-5 input clip
for the online predictor. Writes logs/sgbb7/x8_scale_census.json (refuses
to overwrite). Usage: .venv/bin/python scratch/sg7_scale_census.py
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
os.environ.setdefault("ARM", "off")
os.environ.setdefault("BIRTH_SEED", "0")

import torch  # noqa: E402

import train_mathnative as TM  # noqa: E402
from atomtraj_pins import state_digest  # noqa: E402
from dfa_credit import freeze_lower  # noqa: E402
from dfa_probe import probe_rows, probe_tensors  # noqa: E402
from sg7_credit import input_stats  # noqa: E402

OUT = Path("logs/sgbb7/x8_scale_census.json")
QUAL = Path("logs/sgwriter1/qual.jsonl")
CONTROL_CELL = "sgq_control_s27_lr0.0003"
STEPS = [0, 463, 3084, 5140, 15420]
FIT_CHUNKS = [0, 2, 4, 6]


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    tok = TM.MathTokenizer()
    _, rows, probe_digest, _ = probe_rows(tok)
    fit = [probe_tensors(rows[c * 32:(c + 1) * 32], tok) for c in FIT_CHUNKS]
    ctrl = next(json.loads(l) for l in QUAL.open() if '"kind": "birth"' in l and f'"cell": "{CONTROL_CELL}"' in l)
    rec = {"kind": "sg7_scale_census", "prereg": "SG-BOUNDARY-BLOCK7-1", "commit": commit, "tree_dirty": dirty, "cell": CONTROL_CELL, "fit_chunks": FIT_CHUNKS,
           "probe_token_digest": probe_digest, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "states": {},
           "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    for step in STEPS:
        p = Path(ctrl["outdir"]) / f"step_{step:05d}.pt"
        sd = torch.load(p, map_location="cpu")
        assert state_digest(sd) == ctrl["snapshots"][str(step)]["state_digest"], f"{p}: digest v receipt"
        m = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
        m.load_state_dict(sd)
        m.train()
        freeze_lower(m, 4)
        mu, sdv = input_stats(m, [f[0] for f in fit], [f[1] for f in fit], [f[2] for f in fit])
        x8, e = sdv[0, :384], sdv[0, 384:]
        rec["states"][str(step)] = {"path": str(p), "state_digest": ctrl["snapshots"][str(step)]["state_digest"], "n_tokens": int(sum(int((f[2] != -100).sum()) for f in fit)),
                                    "x8_sd_median": float(x8.median()), "x8_sd_max": float(x8.max()), "x8_mu_abs_max": float(mu[0, :384].abs().max()),
                                    "e_sd_median": float(e.median()), "e_sd_max": float(e.max()), "n_sd_floor": int((sdv <= 1.0000001e-6).sum())}
        print(f"[census] step {step}: x8 sd median {x8.median():.3f} max {x8.max():.3f} | e sd median {e.median():.2e} max {e.max():.2e}", flush=True)
    s0, sf = rec["states"]["0"], rec["states"][str(STEPS[-1])]
    rec["drift"] = {"x8_sd_median_ratio_final_over_w0": sf["x8_sd_median"] / s0["x8_sd_median"], "x8_sd_max_ratio_final_over_w0": sf["x8_sd_max"] / s0["x8_sd_max"]}
    OUT.write_text(json.dumps(rec, indent=1))
    print(f"[census] drift {rec['drift']} -> {OUT}")


if __name__ == "__main__":
    main()
