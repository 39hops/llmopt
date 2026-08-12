"""TENET W1 population build (spec 2026-08-05-tenet-battery.md, W1
prerequisite; Artin GO 2026-08-06, 3080 window into 17:00 EST).

Births PAIRED micro-crystals on the D2-certified diets: seed k ->
one forward-certified birth + one reverse-certified birth, identical
config (d64/L8/FFN256/H4, lr 1.5e-3, bs 8, EPOCHS env, default 2),
only the diet direction varies. SKIP_GATE=1 — the object W1 needs
is direction signal in weights, not gate capability; W1 itself
pre-registers separately once this population exists. ONE-DEVICE
CONTRACT: population born on the 3080, W1 trains/evals on the 3080;
no cross-device pooling.

Loud-failure contract: one manifest row per birth streamed+flushed
(data/w1_population_manifest.jsonl: seed, direction, checkpoint,
weights sha256, wall, rc); a failed birth books rc!=0 and the loop
CONTINUES (the pair is marked incomplete, never silently retried);
terminal record with the census closes the manifest. Deadline
discipline: a PAIR only starts if 2x the EMA birth wall fits before
DEADLINE (complete pairs only — a dangling forward birth is booked
as status=dangling in the terminal record).

Usage (on the 3080/WSL):
  DEADLINE=<epoch> SEED_MAX=50 EPOCHS=2 \
    .venv/bin/python scratch/tenet_w1_population.py
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MANIFEST = Path("data/w1_population_manifest.jsonl")
DEADLINE = float(os.environ.get("DEADLINE", str(time.time() + 14 * 3600)))
SEED_START = int(os.environ.get("SEED_START", "1"))
SEED_MAX = int(os.environ.get("SEED_MAX", "50"))
EPOCHS = os.environ.get("EPOCHS", "2")
DIETS = {"fwd": "data/gen4_forward_certified.jsonl",
         "rev": "data/gen4_reverse_certified.jsonl"}


def birth(seed, direction):
    tag = f"_w1{direction[0]}{seed}"
    out = f"checkpoints/sym_birth_dense{tag}.pt"
    env = {**os.environ,
           "ARM": "dense", "TAG": tag, "DIET": DIETS[direction],
           "SEED": str(seed), "SKIP_GATE": "1", "EPOCHS": EPOCHS,
           "TORCH_DISABLE_NATIVE_JIT": "1"}
    t0 = time.time()
    rc = subprocess.run(
        [sys.executable, "scratch/sym_birth.py"], env=env).returncode
    wall = time.time() - t0
    sha = ""
    if rc == 0 and Path(out).exists():
        sha = hashlib.sha256(Path(out).read_bytes()).hexdigest()[:16]
    row = {"seed": seed, "direction": direction, "ckpt": out,
           "sha": sha, "wall_s": round(wall, 1), "rc": rc,
           "epochs": EPOCHS}
    with MANIFEST.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[w1pop] seed {seed} {direction}: rc {rc} sha {sha} "
          f"{wall:.0f}s", flush=True)
    return rc == 0, wall


def main():
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    for d in DIETS.values():
        if not Path(d).exists():  # verify file deps at arm time
            print(f"[w1pop] MISSING DIET {d} — refusing to start",
                  flush=True)
            sys.exit(1)
    done = set()
    if MANIFEST.exists():  # lossless restart: skip completed births
        for line in MANIFEST.open():
            r = json.loads(line)
            if "terminal" not in r and r.get("rc") == 0:
                done.add((r["seed"], r["direction"]))
    ema, pairs_ok, fails, dangling = 0.0, 0, 0, None
    for seed in range(SEED_START, SEED_MAX + 1):
        need = [d for d in ("fwd", "rev") if (seed, d) not in done]
        if not need:
            pairs_ok += 1
            continue
        if ema and time.time() + len(need) * ema * 1.15 > DEADLINE:
            print(f"[w1pop] deadline fence: stopping before seed "
                  f"{seed} (ema {ema:.0f}s/birth)", flush=True)
            break
        results = []
        for direction in need:
            if ema and time.time() + ema * 1.15 > DEADLINE:
                dangling = (seed, [d for d, ok in results if ok])
                break
            ok, wall = birth(seed, direction)
            results.append((direction, ok))
            ema = wall if not ema else 0.3 * wall + 0.7 * ema
            fails += not ok
        got = {d for d, ok in results if ok} | \
              {d for d in ("fwd", "rev") if (seed, d) in done}
        if got == {"fwd", "rev"}:
            pairs_ok += 1
        elif got:
            dangling = (seed, sorted(got))
    census = {"terminal": True, "pairs_complete": pairs_ok,
              "birth_failures": fails, "dangling": dangling,
              "epochs": EPOCHS, "ema_wall_s": round(ema, 1)}
    with MANIFEST.open("a") as f:
        f.write(json.dumps(census) + "\n")
    print(f"[w1pop] TERMINAL {census}", flush=True)
    if fails or pairs_ok == 0:
        sys.exit(1)  # marker fires on clean success only


if __name__ == "__main__":
    main()
