"""GRAV-0T + GRAV-REV (pre-reg 2026-07-31 night, Artin's riff):
post-hoc gravity — does the merge-free pull work with NO training?

Load a trained Mac lb checkpoint, observe router co-routing over
train-side rows (no_grad forward passes; ARM=gravmoe activates the
model's own overlap-EMA plumbing), then apply R relaxation steps of
the exact training-side pull (w_i += lam*E[i,j]*(w_j - w_i)) and
re-gate. Arms: control (untouched re-gate, harness check), pull
lam=+0.5 at R milestones, repel lam=-0.5 (white-hole arm).
Usage: CKPT=checkpoints/umoe_lb_s1.pt RS=10,50 REPEL_R=10 \
       python scratch/grav_posthoc.py
"""
import os
import sys

os.environ["ARM"] = "gravmoe"   # activates overlap-EMA in forward
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

import umoe_conserve as U  # noqa: E402
import step_grpo_micro as G  # noqa: E402

CKPT = os.environ.get("CKPT", "checkpoints/umoe_lb_s1.pt")
RS = [int(r) for r in os.environ.get("RS", "10,50").split(",")]
REPEL_R = int(os.environ.get("REPEL_R", "10"))
LAM = float(os.environ.get("LAM", "0.5"))
OBS_ROWS = int(os.environ.get("OBS_ROWS", "512"))
OBS_BS = 16


def load(dev):
    tok, m = U.build()
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)["sd"]
    m.load_state_dict(sd)
    return tok, m.to(dev).eval()


def observe(tok, m, dev):
    """Build each block's co-routing EMA over train-side rows."""
    rows = U.load_rows(gen4=True)[:OBS_ROWS * 2]
    enc = []
    for r in rows:
        try:
            ids = tok.encode(f"Current: {r['cur']}\nHints: none\n"
                             f"Step: {r['nxt']}\n") + [tok.eos_id]
        except ValueError:
            continue
        if len(ids) <= 512:
            enc.append(ids)
        if len(enc) >= OBS_ROWS:
            break
    enc.sort(key=len)
    with torch.no_grad():
        for off in range(0, len(enc) - OBS_BS + 1, OBS_BS):
            batch = enc[off:off + OBS_BS]
            L = max(len(q) for q in batch)
            x = torch.tensor([q + [tok.pad_id] * (L - len(q))
                              for q in batch], device=dev)
            m(x)
    emas = [blk.moe.ema.clone() for blk in m.blocks]
    off_mass = [float(E.sum() - E.diagonal().sum()) for E in emas]
    print(f"[0T] observed {len(enc)} rows; off-diag mass/layer "
          f"{[round(v, 4) for v in off_mass]}", flush=True)
    return emas


@torch.no_grad()
def relax(m, emas, lam, steps):
    for _ in range(steps):
        for blk, E in zip(m.blocks, emas):
            for i in range(U.NE):
                for j in range(U.NE):
                    if i == j:
                        continue
                    c = lam * float(E[i, j])
                    for k in ("g", "u", "d"):
                        wi = blk.moe.exp[i][k].weight
                        wj = blk.moe.exp[j][k].weight
                        wi.add_(c * (wj - wi))


def gate(tag, m, tok, dev):
    solves, valid = G.gate_eval(m, tok, dev)
    print(f"[0T gate] {tag} solves {solves} = "
          f"{sum(solves.values())}/120 valid {valid}", flush=True)


def main():
    dev = ("mps" if torch.backends.mps.is_available() else
           "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[0T] ckpt {CKPT} dev {dev} lam {LAM} RS {RS} "
          f"repel_r {REPEL_R}", flush=True)
    tok, m = load(dev)
    gate("control", m, tok, dev)
    emas = observe(tok, m, dev)
    done = 0
    for R in RS:
        relax(m, emas, LAM, R - done)
        done = R
        gate(f"pull lam={LAM} R={R}", m, tok, dev)
    tok, m = load(dev)          # fresh weights for the repel arm
    relax(m, emas, -LAM, REPEL_R)
    gate(f"repel lam={-LAM} R={REPEL_R}", m, tok, dev)


if __name__ == "__main__":
    main()
