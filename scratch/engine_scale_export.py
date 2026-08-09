#!/usr/bin/env python3
"""ENGINE-SCALE-1 per-cell export (PRE-REG 2026-08-07, RESULTS L22317).

Emits, for each unique (params, NWIN) combo in the 30-cell grid,
the DIET-BRIDGE artifact format axiom's engine already consumes:
init param bytes (param_items order, seed 17) followed by window
token ids (int64 [NWIN, T+1]) — plus a manifest jsonl carrying the
full contract per cell (dims, knobs, STEPS, SCHED, shas).

Knob mapping (all env-driven at module import, hence one
subprocess per combo):
  31k  -> NBLK=1            60k -> defaults (NBLK=2, FFN=128)
  110k -> FFN=256           DIM=64 everywhere (clamp law), SHIFT=14
BUILT-IN CHECK: the 60k-w8 artifact must reproduce the certified
DIET-BRIDGE diet_init.bin sha (880e4e295f1e9544...) and windows
ids sha 99caaa646925d150 — the exporter verifies itself against
frozen receipts before writing the manifest.

Usage:
  .venv/bin/python scratch/engine_scale_export.py --all
  (child mode, internal): NBLK/FFN/NWIN env + --one OUTPATH
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

CELLS_SPEC = "docs/superpowers/specs/2026-08-07-engine-scale-cells.jsonl"
OUT_DIR = "docs/superpowers/specs/engine_scale_cells"
T_PLUS_1 = 33  # detbwd T=32 windows
DIET_BRIDGE_INIT_SHA16 = "880e4e295f1e9544"
DIET_BRIDGE_WIN_SHA16 = "99caaa646925d150"


def export_one(out_path: str) -> None:
    """Child: env already carries NBLK/FFN/NWIN; emit one .bin and
    print its shas as one json line on stdout."""
    sys.path.insert(0, ".")
    sys.path.insert(0, "scratch")
    os.environ.setdefault("SHIFT", "14")
    import torch  # noqa: E402
    import detbwd_mb as M  # noqa: E402
    from scripts.train_mathnative import MathTokenizer  # noqa: E402

    V = 40
    M.V = V
    nwin = int(os.environ["NWIN"])
    T = T_PLUS_1 - 1

    tok = MathTokenizer()
    assert len(tok.vocab) == V, f"vocab drifted: {len(tok.vocab)}"
    wins = []
    with open("data/micromodel_gen4_sidecar.jsonl") as f:
        for line in f:
            r = json.loads(line)
            t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
            try:
                ids = tok.encode(t) + [tok.eos_id]
            except ValueError:
                continue
            if len(ids) >= T + 1:
                wins.append(ids[:T + 1])
            if len(wins) == nwin:
                break
    assert len(wins) == nwin, f"only {len(wins)} usable rows"
    wins = torch.tensor(wins, dtype=torch.int64)
    win_sha = hashlib.sha256(wins.numpy().tobytes()).hexdigest()

    torch.manual_seed(M.SEED)
    m = M.MB()
    n_params = sum(p.numel() for _, p in m.param_items())
    with open(out_path, "wb") as f:
        for _, p in m.param_items():
            f.write(p.numpy().tobytes())
        f.write(wins.numpy().tobytes())
    bin_sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    print(json.dumps({"bin_sha": bin_sha, "win_sha": win_sha,
                      "n_params": n_params, "seed": M.SEED,
                      "nblk": M.NBLK, "shift": M.SHIFT}))


PARAM_ENV = {"31k": {"NBLK": "1"}, "60k": {}, "110k": {"FFN": "256"}}


def main() -> None:
    cells = [json.loads(l) for l in open(CELLS_SPEC)]
    os.makedirs(OUT_DIR, exist_ok=True)
    combo_meta = {}
    for c in cells:
        combo = (c["params"], c["NWIN"])
        if combo in combo_meta:
            continue
        env = dict(os.environ)
        env.update(PARAM_ENV[c["params"]])
        env["NWIN"] = str(c["NWIN"])
        out = f"{OUT_DIR}/{c['params']}-w{c['NWIN']}.bin"
        r = subprocess.run(
            [sys.executable, __file__, "--one", out],
            env=env, capture_output=True, text=True, check=True)
        meta = json.loads(r.stdout.strip().splitlines()[-1])
        combo_meta[combo] = {"bin": os.path.basename(out), **meta}
        print(f"{out}: params {meta['n_params']} bin {meta['bin_sha'][:16]}"
              f" win {meta['win_sha'][:16]}")

    anchor = combo_meta[("60k", 8)]
    assert anchor["bin_sha"].startswith(DIET_BRIDGE_INIT_SHA16), (
        "60k-w8 bin does NOT reproduce the certified DIET-BRIDGE "
        f"artifact: {anchor['bin_sha'][:16]} != {DIET_BRIDGE_INIT_SHA16}")
    assert anchor["win_sha"].startswith(DIET_BRIDGE_WIN_SHA16), (
        f"windows ids drifted: {anchor['win_sha'][:16]}")
    print("[verify] 60k-w8 == certified DIET-BRIDGE artifact: OK")

    with open(f"{OUT_DIR}/manifest.jsonl", "w") as f:
        for c in cells:
            meta = combo_meta[(c["params"], c["NWIN"])]
            f.write(json.dumps({
                "cell": c["cell"], "bin": meta["bin"],
                "bin_sha": meta["bin_sha"], "win_sha": meta["win_sha"],
                "n_params": meta["n_params"],
                "contract": {"V": 40, "DIM": 64, "DHEAD": 16,
                             "FFN": 256 if c["params"] == "110k" else 128,
                             "NBLK": 1 if c["params"] == "31k" else 2,
                             "SHIFT": 14, "SEED": 17, "T": 32,
                             "NWIN": c["NWIN"], "STEPS": c["STEPS"],
                             "SCHED": c["SCHED"],
                             "LR": "lrd=1000 (lr 1/1000); SCHED=1 "
                                   "doubles lrd at ABSOLUTE steps "
                                   "250/500/750 (the certified "
                                   "s4000-sched convention, traj "
                                   "sha 15934bb8...)",
                             }}) + "\n")
    print(f"[done] manifest + {len(combo_meta)} bins -> {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--one", metavar="OUT")
    a = ap.parse_args()
    if a.one:
        export_one(a.one)
    else:
        main()
