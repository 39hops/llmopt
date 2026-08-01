"""Export the R2b full-birth reference for axiom's C++ leg
(relay 2026-08-01-0): init bytes in seed-17 draw order + the
reference trajectory digests at the amended contract (SHIFT=12,
constant lr 1/1000, 1000 steps). Artifacts land in
scratch/detbwd_r2b_ref/ (committed — small).
Usage: python scratch/export_r2b_ref.py
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

os.environ["SHIFT"] = "12"
os.environ["STEPS"] = "1000"
os.environ.pop("SCHED", None)
import detbwd_r2b as R  # noqa: E402  (reads envs at import)

OUT = "scratch/detbwd_r2b_ref"


def main():
    os.makedirs(OUT, exist_ok=True)
    # --- init bytes, exact draw order (seed 17: Block() then x, tgt)
    torch.manual_seed(R.SEED)
    blk = R.Block()
    x = torch.randint(-R.Q, R.Q + 1, (R.T, R.D), dtype=torch.int64)
    tgt = torch.randint(0, R.V, (R.T,))
    with open(f"{OUT}/r2b_init.bin", "wb") as f:
        for k in R.Block.KEYS:
            f.write(blk.w[k].numpy().tobytes())
        f.write(x.numpy().tobytes())
        f.write(tgt.to(torch.int64).numpy().tobytes())
    ish = hashlib.sha256(open(f"{OUT}/r2b_init.bin", "rb").read())
    print(f"[export] r2b_init.bin sha {ish.hexdigest()}")

    # --- reference trajectory at the contract (SHIFT=12)
    ts, td = R.build_silu_tables()
    t_exp = R.build_exp_table()
    cos, sin = R.rope_tables()
    tab = {"silu": ts, "dsilu": td, "exp": t_exp,
           "cos": cos, "sin": sin}
    onehot = torch.nn.functional.one_hot(tgt, R.V).to(torch.int64)
    wide = {k: blk.w[k] << R.SHIFT for k in R.Block.KEYS}
    opt = R.IntAdamWQw([wide[k] for k in R.Block.KEYS], R.SHIFT,
                       lrd=1000)
    ref = {"contract": {"SHIFT": R.SHIFT, "GBOOST": R.GBOOST,
                        "PQ": R.PQ, "ACT_CLAMP": R.ACT_CLAMP,
                        "EPS32": R.EPS32, "steps": R.STEPS,
                        "lr": "1/1000", "seed": R.SEED},
           "init_sha": ish.hexdigest(), "milestones": []}
    th = hashlib.sha256()
    for step in range(1, R.STEPS + 1):
        blk.w = {k: R.rdiv(wide[k], 1 << R.SHIFT)
                 for k in R.Block.KEYS}
        lg, cc = blk.fwd(x, tab)
        pp = R.softmax_rows(lg, t_exp)
        loss = int((R.Q - pp[torch.arange(R.T), tgt]).sum())
        GG, _ = blk.bwd((pp - R.Q * onehot) * R.GBOOST, cc, tab)
        opt.step([R.rdiv(GG[k], R.Q * R.GBOOST)
                  for k in R.Block.KEYS])
        if step % 125 == 0:
            for k in R.Block.KEYS:
                th.update(wide[k].numpy().tobytes())
            ref["milestones"].append(
                {"step": step, "loss": loss,
                 "traj_sha": th.hexdigest()})
            print(f"[export] step {step} loss {loss} "
                  f"sha {th.hexdigest()}", flush=True)
    with open(f"{OUT}/r2b_ref.json", "w") as f:
        json.dump(ref, f, indent=1)
    print(f"[export] wrote {OUT}/r2b_ref.json")


if __name__ == "__main__":
    main()
