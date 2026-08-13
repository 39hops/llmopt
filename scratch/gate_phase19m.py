"""CAP-V-TRAJ-1 driver (pre-reg RESULTS 2026-08-13): standard 120
gate on each of the 18 phase19m milestones. Coarse-first order
(endpoints, then bisection) so any prefix of the run still shapes the
whole curve; one jsonl row streamed per milestone as it lands.

Milestones are {model, opt, step} dicts; only ["model"] is loaded
(strict). Stock gate_checkpoint path otherwise — no rebuild, no wrap.

Usage: .venv/bin/python scratch/gate_phase19m.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from llmopt.lab.gate import gate_eval, pick_device  # noqa: E402
from llmopt.lab.hash import git_sha  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

CKPT_DIR = Path("checkpoints/phase19m")
OUT = Path("logs/phase19m_gate/gates.jsonl")
D, LAYERS, FFN, HEADS = 384, 8, 1536, 6


def bisect_order(items):
    """Endpoints first, then repeated interval midpoints."""
    out, seen = [], set()
    for i in (0, len(items) - 1):
        out.append(i); seen.add(i)
    spans = [(0, len(items) - 1)]
    while spans:
        lo, hi = spans.pop(0)
        mid = (lo + hi) // 2
        if mid not in seen:
            out.append(mid); seen.add(mid)
        if mid - lo > 1:
            spans.append((lo, mid))
        if hi - mid > 1:
            spans.append((mid, hi))
    return [items[i] for i in out]


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
