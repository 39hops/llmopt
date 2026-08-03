"""Probe: are the integer-battery primitives bit-identical off the CPU?

Motivation (AMENDMENT P4-DEVICE-SCOPE, 2026-08-02): both P4 legs ran on
CPU, so the ladder's "2 devices" means two CPU architectures. The battery
is CPU-only by PLUMBING, not capability — int_mm is a broadcast multiply
plus a sum reduction, not torch.matmul. Integer addition is associative
and exact, so reduction order (the measured fp transport wedge) cannot
change a value; a GPU leg is therefore predicted to be bit-identical
rather than merely close.

This probe establishes only that the PRIMITIVES agree. Trajectory
transport still requires a pinned-sha run of the full battery, which is
banked and pre-registered separately.

Usage: python scratch/probe_int_device_parity.py [device]
       (device defaults to cuda, then mps, whichever is available)
Exit 0 iff every primitive is bit-identical to CPU; 1 on mismatch or
kernel failure; 2 if there was no accelerator to compare against.
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

from detbwd_r1 import Q, int_mm, rdiv  # noqa: E402
from detbwd_r2b import (  # noqa: E402
    PQ, build_exp_table, rms_bwd, rms_fwd, softmax_bwd, softmax_rows)

SEED = 17


def pick_device(argv):
    if len(argv) > 1:
        return argv[1]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return ""


def primitives(dev, tensors):
    """Every integer op the battery's forward and backward chain uses."""
    a, w, g, t_exp, s = (t.to(dev) for t in tensors)
    out = {}
    out["int_mm"] = int_mm(a, w)
    out["rdiv"] = rdiv(out["int_mm"], Q)
    out["softmax_rows"] = softmax_rows(s, t_exp, PQ)
    out["softmax_bwd"] = softmax_bwd(out["softmax_rows"], s, PQ)
    y, isq = rms_fwd(a, g)
    out["rms_fwd"], out["rms_isq"] = y, isq
    dx, dg = rms_bwd(y, a, g, isq)
    out["rms_bwd_dx"], out["rms_bwd_dg"] = dx, dg
    return out


def main():
    dev = pick_device(sys.argv)
    if not dev:
        # exit 2, NOT 0: a harness scoring this by exit status would
        # otherwise read "pass" from "never ran" (reviewer catch).
        print("[probe] no accelerator available — NOTHING COMPARED")
        return 2
    torch.manual_seed(SEED)
    tensors = (
        torch.randint(-4 * Q, 4 * Q, (32, 64), dtype=torch.int64),
        torch.randint(-Q, Q + 1, (128, 64), dtype=torch.int64),
        torch.full((64,), Q, dtype=torch.int64),
        build_exp_table(),
        torch.randint(-3 * Q, 3 * Q, (32, 32), dtype=torch.int64),
    )
    print(f"[probe] torch {torch.__version__} device {dev} seed {SEED}")
    cpu = primitives("cpu", tensors)
    try:
        acc = primitives(dev, tensors)
    except Exception as exc:                     # kernel gaps are the finding
        print(f"[probe] {dev} FAILED to execute: "
              f"{type(exc).__name__}: {exc}")
        return 1
    ok = True
    for name, ref in cpu.items():
        same = torch.equal(ref, acc[name].cpu())
        ok &= same
        print(f"[probe] {name:14s} cpu vs {dev}: "
              f"{'BIT-IDENTICAL' if same else 'DIFFERS'}")
    print(f"[probe] VERDICT {dev}: "
          f"{'ALL PRIMITIVES BIT-IDENTICAL' if ok else 'MISMATCH PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
