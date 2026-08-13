"""CAP-V-TRAJ-2 driver (pre-reg RESULTS 2026-08-13): the standard
120 gate over the 18 BACKWARD-SCHEDULE-1 milestones. Thin sibling
of the frozen CAP-V-TRAJ-1 driver (gate_phase19m.py, results-cited
— not edited): imports its bisect_order and repeats its loop
against checkpoints/backsched19m/.

Usage: .venv/bin/python scratch/gate_backsched.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from gate_phase19m import bisect_order  # noqa: E402
from llmopt.lab.gate import gate_eval, pick_device  # noqa: E402
from llmopt.lab.hash import git_sha  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

CKPT_DIR = Path("checkpoints/backsched19m")
OUT = Path("logs/backsched_gate/gates.jsonl")
D, LAYERS, FFN, HEADS = 384, 8, 1536, 6


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["ckpt"] for l in OUT.read_text().splitlines() if l}
    ckpts = sorted(CKPT_DIR.glob("m*.pt"))
    assert len(ckpts) == 18, f"expected 18 milestones, found {len(ckpts)}"
    tok = MathTokenizer()
    dev = pick_device()
    sha = git_sha(short=True)
    model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN).to(dev)
    for p in bisect_order(ckpts):
        if p.name in done:
            print(f"{p.name} already gated, skip", flush=True)
            continue
        blob = torch.load(p, map_location="cpu")
        model.load_state_dict(blob["model"])
        model.eval()
        t0 = time.time()
        solves, valid = gate_eval(model, tok, dev)
        tot = sum(solves.values())
        row = {"ckpt": p.name, "step": blob["step"], "solves": solves,
               "total": tot, "valid_pct": round(valid, 2),
               "device": dev, "wall_s": round(time.time() - t0, 1),
               "code_commit": sha}
        with OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{p.name} step={blob['step']} gate: {solves} = {tot}/120 "
              f"@ {valid:.2f}% [{row['wall_s']}s]", flush=True)
    print("ALL 18 GATED", flush=True)


if __name__ == "__main__":
    main()
