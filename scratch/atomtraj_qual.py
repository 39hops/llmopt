"""ATOM-DIET-TRAJECTORY-1 qualification ladder (pre-reg RESULTS L66546,
items 1 to 5), stock-only on the permanently excluded smoke seed 11.
Runs, in order, and writes logs/atomtraj1/qual.json:
  1. the source-invariant tests (pytest tests/test_atomtraj_source_invariant.py);
  2. DRYRUN of both arms at a registered seed (stream digests, counts;
     no output);
  3. CPU deterministic emission-neutrality probe: SMOKE=1 DEVICE=cpu
     SMOKE_STEPS=300, EMIT=1 v EMIT=0, seed 11: final state_digest,
     optimizer-state digest and torch RNG digest must be identical;
  4. path-isolated mps smoke: SMOKE=1 stock seed 11 capped at 1,100
     steps (snapshots 0, 463, 1,028, 1,100), then the post-hoc gate
     runner, census and verifier in SMOKE mode;
  5. disk preflight (>= 15 GB free).
Refuses to run if qual.json exists. A re-run after a failed attempt
needs the previous smoke outputs set aside first (rename
checkpoints/atomtraj1_smoke/ and logs/atomtraj1/smoke.jsonl to an
_attemptN_* name; the driver's refuse-if-exists guards are unconditional).

Usage: .venv/bin/python scratch/atomtraj_qual.py
"""
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PY = sys.executable
LOGS = Path("logs/atomtraj1")
QUAL = LOGS / "qual.json"
SMOKE_SEED = "11"


def run(cmd, env=None, must=True):
    e = dict(os.environ, **(env or {}))
    r = subprocess.run(cmd, env=e, capture_output=True, text=True)
    if must and r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"qualification step failed: {cmd} {env}")
    return r


def main():
    assert not QUAL.exists(), "REFUSING: qual.json exists"
    LOGS.mkdir(parents=True, exist_ok=True)
    out = {"prereg": "ATOM-DIET-TRAJECTORY-1", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "smoke_seed": int(SMOKE_SEED)}
    # 1 source invariants
    r = run([PY, "-m", "pytest", "-q", "tests/test_atomtraj_source_invariant.py"])
    out["source_invariant"] = r.stdout.strip().splitlines()[-1]
    # 2 dry runs
    out["dryrun"] = {}
    for arm in ("stock", "atoms"):
        r = run([PY, "scratch/birth19m_atoms_traj.py"], {"ARM": arm, "SEED": "5", "DRYRUN": "1"})
        out["dryrun"][arm] = json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][-1])
    # 3 CPU emission-neutrality probe
    smoke_rows_before = len(open(LOGS / "smoke.jsonl").readlines()) if (LOGS / "smoke.jsonl").exists() else 0
    for emit in ("1", "0"):
        run([PY, "scratch/birth19m_atoms_traj.py"], {"ARM": "stock", "SEED": SMOKE_SEED, "SMOKE": "1", "DEVICE": "cpu",
                                                     "SMOKE_STEPS": "300", "EMIT": emit, "TAG": f"_probe_emit{emit}"})
    rows = [json.loads(l) for l in open(LOGS / "smoke.jsonl")][smoke_rows_before:]
    probe = {r["tag"]: r for r in rows if r.get("kind") == "birth" and r["device"] == "cpu"}
    a, b = probe["_probe_emit1"], probe["_probe_emit0"]
    out["cpu_probe"] = {"steps": a["steps"], "emit1_final_state_digest": a["final"]["state_digest"], "emit0_final_state_digest": b["final"]["state_digest"],
                        "state_identical": a["final"]["state_digest"] == b["final"]["state_digest"],
                        "optimizer_identical": a["optimizer_state_digest"] == b["optimizer_state_digest"],
                        "rng_identical": a["torch_rng_digest"] == b["torch_rng_digest"],
                        "emit1_snapshots": sorted(int(k) for k in a["snapshots"]), "emit0_snapshots": sorted(int(k) for k in b["snapshots"])}
    assert out["cpu_probe"]["state_identical"] and out["cpu_probe"]["optimizer_identical"] and out["cpu_probe"]["rng_identical"], out["cpu_probe"]
    # 4 mps smoke + plumbing
    r = run([PY, "scratch/birth19m_atoms_traj.py"], {"ARM": "stock", "SEED": SMOKE_SEED, "SMOKE": "1", "SMOKE_STEPS": "1100"})
    out["smoke_train_tail"] = r.stdout.strip().splitlines()[-2:]
    run([PY, "scratch/birth19m_atoms_trajgate.py"], {"SMOKE": "1"})
    run([PY, "scratch/atomtraj_census.py"], {"SMOKE": "1"})
    r = run([PY, "scratch/atomtraj_verify.py"], {"SMOKE": "1"})
    rows = [json.loads(l) for l in open(LOGS / "smoke.jsonl")]
    sm = [x for x in rows if x.get("kind") == "birth" and x["device"] != "cpu"][-1]
    out["smoke"] = {"device": sm["device"], "steps": sm["steps"], "snapshots": sorted(int(k) for k in sm["snapshots"]), "wall_s": sm["wall_s"],
                    "final_eq_last_snapshot": sm["final"]["state_digest"] == sm["snapshots"][str(sm["steps"])]["state_digest"],
                    "gate_rows": [{k: g[k] for k in ("step", "total", "wall_s")} for g in rows if g.get("kind") == "gate"],
                    "verify": {k: [x for x in rows if x.get("kind") == "verify"][-1][k] for k in ("verdict", "n_discrepancies", "discrepancies")}}
    assert out["smoke"]["verify"]["verdict"] == "VERIFIED", out["smoke"]["verify"]
    # 5 disk
    out["disk_free_gb"] = round(shutil.disk_usage(".").free / 1024 ** 3, 1)
    assert out["disk_free_gb"] >= 15
    out["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    out["verdict"] = "QUALIFIED"
    QUAL.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
