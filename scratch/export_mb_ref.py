"""Export the multi-block deterministic-birth reference for
axiom's leg: init bytes (all params in param_items order, then
tok, then tgt; int64 LE) + milestone trajectory digests at the
R2b contract grown by n_blocks (SHIFT=12, GBOOST=256, constant
lr 1/1000, 1000 steps, NBLK=2, seed 17). Artifacts land in
scratch/detbwd_mb_ref/ (committed — small).
Usage: .venv/bin/python scratch/export_mb_ref.py
"""
import hashlib
import json
import os
import sys

os.environ.setdefault("SHIFT", "12")
os.environ.setdefault("NBLK", "2")
import torch  # noqa: E402

import detbwd_mb as M  # noqa: E402
from detbwd_r1 import Q, rdiv  # noqa: E402
from detbwd_r2b import (  # noqa: E402
    ACT_CLAMP, EPS32, GBOOST, PQ, T, V,
    build_exp_table, build_silu_tables, rope_tables, softmax_rows)
from detbwd_r3_qw import IntAdamWQw  # noqa: E402

OUT = "scratch/detbwd_mb_ref"
STEPS = 1000


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(M.SEED)
    m = M.MB()
    tok = torch.randint(0, V, (T,))
    tgt = torch.randint(0, V, (T,))
    names = [n for n, _ in m.param_items()]
    with open(f"{OUT}/mb_init.bin", "wb") as f:
        for _, p in m.param_items():
            f.write(p.numpy().tobytes())
        f.write(tok.to(torch.int64).numpy().tobytes())
        f.write(tgt.to(torch.int64).numpy().tobytes())
    ish = hashlib.sha256(open(f"{OUT}/mb_init.bin", "rb").read())
    print(f"[export] mb_init.bin sha {ish.hexdigest()}")

    ts, td = build_silu_tables()
    t_exp = build_exp_table()
    cos, sin = rope_tables()
    tab = {"silu": ts, "dsilu": td, "exp": t_exp,
           "cos": cos, "sin": sin}
    onehot = torch.nn.functional.one_hot(tgt, V).to(torch.int64)
    flat = dict(m.param_items())
    wide = {n: flat[n] << M.SHIFT for n in names}
    opt = IntAdamWQw([wide[n] for n in names], M.SHIFT, lrd=1000)
    ref = {"contract": {"SHIFT": M.SHIFT, "GBOOST": GBOOST,
                        "PQ": PQ, "ACT_CLAMP": ACT_CLAMP,
                        "EPS32": EPS32, "n_blocks": M.NBLK,
                        "steps": STEPS, "lr": "1/1000",
                        "seed": M.SEED},
           "param_order": names,
           "init_sha": ish.hexdigest(), "milestones": []}
    th = hashlib.sha256()
    for step in range(1, STEPS + 1):
        nar = {n: rdiv(wide[n], 1 << M.SHIFT) for n in names}
        m.emb, m.g_f = nar["emb"], nar["g_f"]
        for i, b in enumerate(m.bodies):
            b.w = {k: nar[f"b{i}.{k}"] for k in M.Body.KEYS}
        lg, cc = m.fwd(tok, tab)
        pp = softmax_rows(lg, t_exp)
        loss = int((Q - pp[torch.arange(T), tgt]).sum())
        GG = m.bwd((pp - Q * onehot) * GBOOST, cc, tab)
        opt.step([rdiv(GG[n], Q * GBOOST) for n in names])
        if step % 125 == 0:
            for n in names:
                th.update(wide[n].numpy().tobytes())
            ref["milestones"].append(
                {"step": step, "loss": loss,
                 "traj_sha": th.hexdigest()})
            print(f"[export] step {step} loss {loss} "
                  f"sha {th.hexdigest()}", flush=True)
    with open(f"{OUT}/mb_ref.json", "w") as f:
        json.dump(ref, f, indent=1)
    print(f"[export] wrote {OUT}/mb_ref.json")


if __name__ == "__main__":
    main()
