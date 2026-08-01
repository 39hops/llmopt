"""Re-gate sigma cell (pre-reg 2026-07-31 night): gate ONE
untouched checkpoint in a fresh process — run N times via the bash
loop below to measure cross-process re-gate spread on mps (the
GRAV-0T control came back 37 v the booked 44).
Usage: for i in 1 2 3; do CKPT=checkpoints/umoe_lb_s1.pt \
       python scratch/gate_regate.py; done
"""
import os
import sys

os.environ.setdefault("ARM", "lb")
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

import umoe_conserve as U  # noqa: E402
import step_grpo_micro as G  # noqa: E402

CKPT = os.environ.get("CKPT", "checkpoints/umoe_lb_s1.pt")


def main():
    dev = ("mps" if torch.backends.mps.is_available() else
           "cuda" if torch.cuda.is_available() else "cpu")
    import hashlib
    sha = hashlib.sha256(open(CKPT, "rb").read()).hexdigest()[:16]
    print(f"[regate] ckpt sha {sha}", flush=True)
    tok, m = U.build()
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)["sd"]
    m.load_state_dict(sd)
    m = m.to(dev).eval()
    solves, valid = G.gate_eval(m, tok, dev)
    print(f"[regate] {CKPT} dev {dev} solves {solves} = "
          f"{sum(solves.values())}/120 valid {valid}", flush=True)


if __name__ == "__main__":
    main()
