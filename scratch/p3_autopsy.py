"""HARDENING-P3 R2 wrapper (autopsy leg): per-(layer,head) deletion
map on a FRESH-SEED h8 crystal — frozen scratch/head_autopsy.py
untouched (it is __main__-guarded; we import, point CKPT at the
seed's EMA checkpoint, and run main()). The autopsy READS only —
no save-redirect needed. After the driver's own map + min/max/ctrl
full gates, two FIXED-CELL full gates (L1h7, L1h4 — the original
crystal's extremes) answer the secondary identity question: do the
named cells transport across seeds, or is only the STRUCTURE
(sparse critical circuit) seed-stable?

Usage: SEED=<n> .venv/bin/python scratch/p3_autopsy.py
Device: Mac/mps (device-of-origin of the HEAD AUTOPSY claim).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

SEED = os.environ["SEED"]
ckpt = Path(f"checkpoints/sym_birth_dense_mps_h8_s{SEED}_ema.pt")
if not ckpt.exists():
    raise SystemExit(f"MISSING birth: {ckpt} (run the birth leg first)")

import torch  # noqa: E402
import head_autopsy as HA  # noqa: E402
import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

HA.CKPT = str(ckpt)
HA.main()

# secondary identity read: the ORIGINAL crystal's extreme cells,
# full-gated on this seed (descriptive, no bar)
tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
base = torch.load(str(ckpt), map_location="cpu", weights_only=True)
D, HEADS = HA.D, HA.HEADS
HD = D // HEADS


def drop(li, h):
    sd = {k: v.clone() for k, v in base.items()}
    a, b = h * HD, (h + 1) * HD
    W = sd[f"blocks.{li}.qkv.weight"]
    for part in range(3):
        W[part * D + a:part * D + b, :] = 0
    sd[f"blocks.{li}.o.weight"][:, a:b] = 0
    return sd


for (li, h, tag) in [(1, 7, "orig-min L1h7"), (1, 4, "orig-max L1h4")]:
    m = build_model(len(tok.vocab), d=D, layers=HA.LAYERS,
                    heads=HEADS, ffn=HA.FFN).to(dev)
    m.load_state_dict({k: v.to(dev) for k, v in drop(li, h).items()})
    m.eval()
    with torch.no_grad():
        solves, valid = G.gate_eval(m, tok, dev)
    print(f"AUTOPSY-S{SEED} {tag} (FULL): {solves} = "
          f"{sum(solves.values())}/120 @ {valid:.2f}%", flush=True)
    del m
