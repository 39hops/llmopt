"""FROZEN-BACKBONE-1 post-hoc gates and the replication law (PRE-REG
FROZEN-BACKBONE-1). Reads logs/frozenbb1/births.jsonl (six FB birth rows:
FULL and FROZEN at seeds 24, 25, 26), gates every final with
llmopt.lab.gate.gate_eval on mps (the standard 120), appends kind=gate rows,
computes delta_s = gate_FROZEN_s - gate_FULL_s per seed and writes
logs/frozenbb1/replication.json (refuses to overwrite):
REPLICATES iff delta_s >= -7 on all three pairs. Individual gates, deltas
and the mean are booked descriptively; no averaging rescues a failed pair.
Also the frozen ACT vector (scratch/dfa_act.py act_vector) on the six
finals, descriptive.

Usage: .venv/bin/python scratch/fb_gate.py
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

import torch  # noqa: E402

import train_mathnative as TM  # noqa: E402
from atomtraj_pins import state_digest  # noqa: E402
from dfa_act import act_vector, dist  # noqa: E402
from dfa_probe import probe_rows  # noqa: E402
from llmopt.lab.gate import gate_eval  # noqa: E402

BIRTHS = Path("logs/frozenbb1/births.jsonl")
OUT = Path("logs/frozenbb1/replication.json")
SEEDS = (24, 25, 26)
LAW = -7


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    rows = [json.loads(l) for l in BIRTHS.open()]
    births = {(b["seed"], b["arm"]): b for b in rows if b.get("kind") == "birth" and b["phase"] == "fb"}
    missing = [(s, a) for s in SEEDS for a in ("FULL", "FROZEN") if (s, a) not in births]
    assert not missing, f"births missing: {missing}"
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    tree_dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    if tree_dirty:
        raise SystemExit("REFUSING: registered gating on a dirty tree")
    tok = TM.MathTokenizer()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    _, prows, probe_digest, _ = probe_rows(tok)
    gated = {r["cell"] for r in rows if r.get("kind") == "gate"}
    gates, acts = {}, {}
    for (seed, arm), b in sorted(births.items()):
        assert b["final"] is not None and b["stable_training"], f"{b['cell']} did not finish with finite loss"
        p = Path(b["outdir"]) / "final.pt"
        sd = torch.load(p, map_location="cpu")
        assert state_digest(sd) == b["final"]["state_digest"], f"{p}: digest v receipt"
        if b["cell"] in gated:
            g = next(r for r in rows if r.get("kind") == "gate" and r["cell"] == b["cell"])
        else:
            m = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
            m.load_state_dict(sd)
            m = m.to(dev).eval()
            t0 = time.time()
            solves, valid = gate_eval(m, tok, dev)
            g = {"kind": "gate", "phase": "fb", "cell": b["cell"], "seed": seed, "arm": arm, "solves": {str(k): int(v) for k, v in solves.items()},
                 "total": int(sum(solves.values())), "valid_pct": round(float(valid), 2), "device": dev, "wall_s": round(time.time() - t0, 1),
                 "state_digest": b["final"]["state_digest"], "final_file_sha256": b["final"]["file_sha256"], "outdir": b["outdir"],
                 "code_commit": commit, "tree_dirty": tree_dirty, "gated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
            with BIRTHS.open("a") as f:
                f.write(json.dumps(g) + "\n")
            del m
        gates[(seed, arm)] = g
        av = act_vector(sd, tok, prows)
        assert av["selfcheck_ok"]
        acts[(seed, arm)] = av
        print(f"[fb] {b['cell']}: gate {g['total']} valid {g['valid_pct']}", flush=True)
    per_seed = {}
    for s in SEEDS:
        gf, gz = gates[(s, "FULL")]["total"], gates[(s, "FROZEN")]["total"]
        per_seed[str(s)] = {"gate_full": gf, "gate_frozen": gz, "delta": gz - gf, "pair_holds": (gz - gf) >= LAW,
                            "solves_full": gates[(s, "FULL")]["solves"], "solves_frozen": gates[(s, "FROZEN")]["solves"],
                            "act_dist_frozen_v_full": dist(acts[(s, "FROZEN")], acts[(s, "FULL")]),
                            "act_full": acts[(s, "FULL")]["act"], "act_frozen": acts[(s, "FROZEN")]["act"]}
    deltas = [per_seed[str(s)]["delta"] for s in SEEDS]
    rec = {"prereg": "FROZEN-BACKBONE-1", "kind": "replication", "commit": commit, "device": dev, "probe_token_digest": probe_digest,
           "law": f"REPLICATES iff delta_s >= {LAW} on all three pairs", "per_seed": per_seed, "deltas": deltas,
           "mean_delta_descriptive": sum(deltas) / len(deltas), "REPLICATES": all(d >= LAW for d in deltas),
           "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    OUT.write_text(json.dumps(rec, indent=1))
    with BIRTHS.open("a") as f:
        f.write(json.dumps({k: v for k, v in rec.items() if k != "per_seed"} | {"per_seed_gates": {s: (v["gate_full"], v["gate_frozen"]) for s, v in per_seed.items()}}) + "\n")
    print("[fb] replication:", json.dumps({"deltas": deltas, "REPLICATES": rec["REPLICATES"]}))


if __name__ == "__main__":
    main()
