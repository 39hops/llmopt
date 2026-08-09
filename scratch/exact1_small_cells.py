#!/usr/bin/env python3
"""EXACT1-SMALL: d8/d16 ladder+anchor cells on axiom's ENGINE-EXACT-1.

Registered arm driver (pre-reg EXACT1-SMALL, RESULTS 2026-08-09) for
the two-regime disagreement-#3 restatement: ring-grain error (rung
divergence vs PRECISION) versus frozen-carry error (anchor offset,
constant in p by the PQ-freeze mechanism). Runs the Q9/Q32/Q64 rungs
plus the exact-rational anchor per cell over IDENTICAL tables/init
bytes, dumps de-grained (shipped-scale i64) weights per step, and
appends per-step divergence rows incrementally (streaming doctrine:
a wall-killed run must leave its completed steps readable).

Tables/init bytes are generated HERE in portable numpy (not the C++
fixture's std::uniform_int_distribution, whose algorithm is
implementation-defined and would silently vary across stdlibs);
seeds derive from stable strings per house doctrine.

Usage:
  python scratch/exact1_small_cells.py <axiom_build_dir> \
      [--cell d8|d16] [--rung-steps N] [--anchor-steps N]
      [--budget SECONDS] [--out DIR]
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time

import numpy as np

CELLS = {
    "d8": dict(T=4, D=8, DH=4, F=16, V=8),
    "d16": dict(T=8, D=16, DH=8, F=32, V=16),
}

ap = argparse.ArgumentParser()
ap.add_argument("build_dir")
ap.add_argument("--cell", choices=list(CELLS), default="d8")
ap.add_argument("--rung-steps", type=int, default=12)
ap.add_argument("--anchor-steps", type=int, default=6)
ap.add_argument("--ceiling", type=int, default=1 << 22)
ap.add_argument("--budget", type=float, default=7200.0)
ap.add_argument("--out", default=None)
args = ap.parse_args()

sys.path.insert(0, args.build_dir)
import intbirth  # noqa: E402

dims = CELLS[args.cell]
out = args.out or f"logs/exact1small/{args.cell}"
os.makedirs(out, exist_ok=True)


def put_tensor(b, name, v):
    b += struct.pack("<H", len(name)) + name.encode() + b"\x01"
    b += struct.pack("<Q", len(v))
    b += np.asarray(v, dtype=np.int64).tobytes()
    return b


def make_tables(T, DH, **_):
    """Fixture-style synthetic tables (identity rope, silu/exp ramps)
    at shipped scale; the SAME bytes feed every arm."""
    ts, tse, RS = 100, 50, 1 << 14
    sil = [(i - ts) // 2 if i >= ts else -((ts - i) // 2)
           for i in range(2 * ts + 1)]  # C++ (i-ts)/2 truncates to 0
    dsl = [256] * (2 * ts + 1)
    ex = [1 + i * 7 for i in range(tse + 1)]
    rc = [RS] * (T * (DH // 2))
    rs = [0] * (T * (DH // 2))
    b = b"AXP3" + struct.pack("<I", 5)
    for name, v in [("silu.tab", sil), ("dsilu.tab", dsl),
                    ("exp.tab", ex), ("rope.cos", rc), ("rope.sin", rs)]:
        b = put_tensor(b, name, v)
    return b


def make_init(T, D, DH, F, V, seed_str):
    """11 KEYS tensors at shipped Q9 scale, then x [T,D], then tgt
    [T] — the full_birth ctor convention; portable numpy PCG64
    seeded from a stable string."""
    seed = int.from_bytes(hashlib.sha256(seed_str.encode()).digest()[:8],
                          "little")
    rng = np.random.default_rng(seed)
    sizes = [DH * D, DH * D, DH * D, D * DH, F * D, F * D, D * F,
             V * D, D, D, D]
    parts = [rng.integers(-256, 257, n, dtype=np.int64) for n in sizes]
    parts.append(rng.integers(-256, 257, T * D, dtype=np.int64))
    parts.append(np.arange(T, dtype=np.int64) % V)
    return b"".join(p.tobytes() for p in parts)


tables = make_tables(**dims)
init = make_init(**dims, seed_str=f"exact1-small-{args.cell}-0")
print(f"cell {args.cell} dims {dims} tables sha "
      f"{hashlib.sha256(tables).hexdigest()[:16]} init sha "
      f"{hashlib.sha256(init).hexdigest()[:16]}", flush=True)


def run_arm(name, obj, steps, budget):
    t0 = time.time()
    path = os.path.join(out, f"{name}.jsonl")
    with open(path, "w") as f:
        for s in range(1, steps + 1):
            try:
                st = time.time()
                obj.run(1)
                dt = time.time() - st
            except RuntimeError as e:  # bit ceiling — loud, expected
                f.write(json.dumps({"step": s, "aborted": str(e)}) + "\n")
                f.flush()
                print(f"{name}: step {s} aborted: {e}", flush=True)
                break
            with open(os.path.join(out, f"{name}_step{s}.w9"), "wb") as w:
                w.write(obj.weights_grain9_bytes())
            f.write(json.dumps({"step": s, "loss": obj.loss,
                                "wall_s": round(dt, 3)}) + "\n")
            f.flush()
            print(f"{name}: step {s} loss {obj.loss} ({dt:.2f}s)",
                  flush=True)
            if time.time() - t0 > budget:
                f.write(json.dumps({"step": s,
                                    "aborted": "wall-clock budget"}) + "\n")
                f.flush()
                print(f"{name}: budget hit after step {s}", flush=True)
                break


for prec in (9, 32, 64):
    fb = intbirth.FullBirth(tables, init, {**dims, "PRECISION": prec})
    run_arm(f"q{prec}", fb, args.rung_steps, args.budget)

intbirth.ExactAnchor.set_bit_ceiling(args.ceiling)
an = intbirth.ExactAnchor(tables, init, dims)
run_arm("anchor", an, args.anchor_steps, args.budget)

# Divergence pass over whatever landed on disk (survives partial runs).
with open(os.path.join(out, "divergence.jsonl"), "w") as f:
    for s in range(1, args.rung_steps + 1):
        row = {"step": s, "pairs": {}}
        w = {}
        for n in ("q9", "q32", "q64", "anchor"):
            p = os.path.join(out, f"{n}_step{s}.w9")
            if os.path.exists(p):
                w[n] = np.fromfile(p, dtype=np.int64)
        for a, b_ in [("q9", "q32"), ("q9", "q64"), ("q32", "q64"),
                      ("anchor", "q9"), ("anchor", "q32"),
                      ("anchor", "q64")]:
            if a in w and b_ in w:
                d = np.abs(w[a] - w[b_])
                row["pairs"][f"{a}-{b_}"] = {
                    "mean": float(d.mean()), "max": int(d.max()),
                    "identical": bool((d == 0).all())}
        f.write(json.dumps(row) + "\n")
print("done ->", out, flush=True)
