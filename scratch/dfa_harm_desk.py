"""DFA-LOWER-HARM-DESK-0 transplants (zero training). For each k in {1, 2, 4}
on the CREDIT-ANCHOR-FRONTIER-1 seed-23 specimens (VERDICT RESULTS L69409):
H = the lr 3e-4 hybrid final (blocks 0..7-k DFA-trained, top k + norm / head
BP; gate 0), Z = the same-seed zero-credit control final (emb + blocks
0..7-k frozen at W_0, top k + norm / head BP; gate 28 / 55 / 62), W_0 = the
shared seed-23 init (step_00000.pt, digest-checked on both arms). Exact
boundary partition of the frontier: LOWER = emb.weight + every tensor of
blocks 0..7-k; TOP = blocks 8-k..7 + norm.g + head.weight.
  T1  ZERO-TOP <- HYBRID-LOWER : TOP from Z, LOWER from H
  T2  HYBRID-TOP <- W0-LOWER   : TOP from H, LOWER from W_0 (= Z's lower, asserted)
Each transplant is gated once with llmopt.lab.gate.gate_eval on mps (the
standard 120). Endpoints (H full = 0, Z full = 28 / 55 / 62) are fixed
observations from qual.jsonl and are not re-gated. Readings registered:
COLLAPSE iff T1 gate < 24; RESCUE iff T2 gate > 0. Rows stream to
logs/dfaharm0/gates.jsonl, table to logs/dfaharm0/harm.json (refuses to
overwrite). SMOKE=1: the seed-11 smoke arms (hybrid k = 2 and zero k = 2,
300 steps) at the 8-prompt proxy tier, receipts logs/dfaharm0/smoke.jsonl.
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
OUT = Path("logs/dfaharm0")
FLOOR = 24
KS = (2,) if SMOKE else (1, 2, 4)
ROOT = Path("checkpoints/writercaf1_smoke" if SMOKE else "checkpoints/writercaf1")
QUAL = Path("logs/writercaf1/smoke.jsonl" if SMOKE else "logs/writercaf1/qual.jsonl")
ENDPOINTS = {1: 28, 2: 55, 4: 62}


def load(p):
    return torch.load(p, map_location="cpu")


def lower_keys(k):
    return ["emb.weight"] + [f"blocks.{l}.{t}" for l in range(8 - k) for t in ("qkv.weight", "o.weight", "gate.weight", "up.weight", "down.weight", "n1.g", "n2.g")]


def transplant(top_sd, lower_sd, k):
    out = {kk: v.clone() for kk, v in top_sd.items()}
    for kk in lower_keys(k):
        out[kk] = lower_sd[kk].clone()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / ("smoke.jsonl" if SMOKE else "harm.json")
    rows_p = OUT / ("smoke_gates.jsonl" if SMOKE else "gates.jsonl")
    if not SMOKE and (target.exists() or rows_p.exists()):
        raise SystemExit(f"REFUSING: {target} or {rows_p} exists")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    if dirty and not SMOKE:
        raise SystemExit("REFUSING: registered desk on a dirty tree")
    births = {r["cell"]: r for r in (json.loads(l) for l in QUAL.open()) if r.get("kind") == "birth"}
    tok = TM.MathTokenizer()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    n = 8 if SMOKE else None
    rec = {"prereg": "DFA-LOWER-HARM-DESK-0", "kind": "harm", "smoke": SMOKE, "commit": commit, "tree_dirty": dirty, "device": dev,
           "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "floor": FLOOR, "per_k": {}}
    for k in KS:
        seed = "s11" if SMOKE else "s23"
        hcell = f"{'smoke' if SMOKE else 'qual'}_hybrid_k{k}_{seed}_S1_lr0.0003"
        zcell = f"{'smoke' if SMOKE else 'qual'}_zero_k{k}_{seed}_lr0.0003"
        H, Z = load(ROOT / hcell / "final.pt"), load(ROOT / zcell / "final.pt")
        W0 = load(ROOT / hcell / "step_00000.pt")
        dH, dZ, dW = state_digest(H), state_digest(Z), state_digest(W0)
        assert dH == births[hcell]["final"]["state_digest"] and dZ == births[zcell]["final"]["state_digest"], f"k={k}: final digests v receipts"
        assert dW == births[hcell]["init_state_digest"] == births[zcell]["init_state_digest"], f"k={k}: W_0 digest v receipts"
        assert all(torch.equal(Z[kk], W0[kk]) for kk in lower_keys(k)), f"k={k}: the zero control's lower package is not W_0"
        assert not all(torch.equal(H[kk], W0[kk]) for kk in lower_keys(k)), f"k={k}: the hybrid's lower package equals W_0"
        pk = {"hybrid": hcell, "zero": zcell, "hybrid_digest": dH, "zero_digest": dZ, "w0_digest": dW, "n_lower_keys": len(lower_keys(k)),
              "endpoint_hybrid_full": 0, "endpoint_zero_full": ENDPOINTS[k], "transplants": {}}
        for name, sd in (("T1_zero_top_hybrid_lower", transplant(Z, H, k)), ("T2_hybrid_top_w0_lower", transplant(H, W0, k))):
            assert state_digest(sd) not in (dH, dZ), f"{name}: transplant equals an endpoint"
            m = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
            m.load_state_dict(sd)
            m = m.to(dev).eval()
            t0 = time.time()
            solves, valid = gate_eval(m, tok, dev, n=n) if n else gate_eval(m, tok, dev)
            row = {"kind": "gate", "prereg": "DFA-LOWER-HARM-DESK-0", "k": k, "transplant": name, "smoke": SMOKE, "solves": {str(a): int(b) for a, b in solves.items()},
                   "total": int(sum(solves.values())), "valid_pct": round(float(valid), 2), "device": dev, "wall_s": round(time.time() - t0, 1),
                   "state_digest": state_digest(sd), "code_commit": commit, "tree_dirty": dirty,
                   "gated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
            with rows_p.open("a") as f:
                f.write(json.dumps(row) + "\n")
            pk["transplants"][name] = {kk: row[kk] for kk in ("solves", "total", "valid_pct", "state_digest", "wall_s")}
            print(f"[harm] k={k} {name}: {row['total']} ({row['wall_s']}s)", flush=True)
            del m
        t1, t2 = pk["transplants"]["T1_zero_top_hybrid_lower"]["total"], pk["transplants"]["T2_hybrid_top_w0_lower"]["total"]
        pk["readings"] = {"COLLAPSE_T1_below_floor": t1 < FLOOR, "RESCUE_T2_above_zero": t2 > 0, "T1_minus_zero_full": t1 - ENDPOINTS[k], "T2_minus_hybrid_full": t2 - 0}
        rec["per_k"][str(k)] = pk
    rec["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    if SMOKE:
        with target.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print("[harm] smoke row appended")
    else:
        target.write_text(json.dumps(rec, indent=1))
        print("[harm] written", target, json.dumps({k: v["readings"] for k, v in rec["per_k"].items()}))


if __name__ == "__main__":
    main()
