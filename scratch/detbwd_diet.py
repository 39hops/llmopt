"""Deterministic mini-crystal birth on the REAL MATH DIET
(queued by Artin 2026-08-01 pre-compact): the multi-block integer
model (detbwd_mb anatomy, V=40 = MathTokenizer vocab) trained
with true next-token CE on gen-4 diet windows — the bridge from
random-target demos to the actual curriculum.

Diet handling: 8 windows of 33 tokens are drawn ONCE from
data/micromodel_gen4_sidecar.jsonl (first strictly-encodable rows
long enough, file order — deterministic given the file), then the
token ids travel IN the exported init artifact, so the trajectory
is reproducible from the artifact alone (the diet file is
untracked and never a dependency). Steps cycle windows
round-robin, one per step.

Contract: MB's at SHIFT=14 const lr (MB-S14 default when chasing
loss), NBLK=2, seed 17. Env: STEPS (default 1000), SHIFT, SCHED.
Usage: .venv/bin/python scratch/detbwd_diet.py
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
os.environ.setdefault("SHIFT", "14")
import torch  # noqa: E402

import detbwd_mb as M  # noqa: E402
from detbwd_r1 import Q, rdiv  # noqa: E402
from detbwd_r2b import (  # noqa: E402
    GBOOST, T, build_exp_table, build_silu_tables, rope_tables,
    softmax_rows)
from detbwd_r3_qw import IntAdamWQw  # noqa: E402
from scripts.train_mathnative import MathTokenizer  # noqa: E402

V = 40                      # MathTokenizer base vocab
M.V = V                     # MB.__init__ reads module global
NWIN = 8
STEPS = int(os.environ.get("STEPS", "1000"))
DIET = "data/micromodel_gen4_sidecar.jsonl"


def draw_windows():
    """First NWIN strictly-encodable diet rows with >= T+1 tokens,
    file order. Returns int64 [NWIN, T+1]."""
    tok = MathTokenizer()
    assert len(tok.vocab) == V, f"vocab drifted: {len(tok.vocab)}"
    wins = []
    with open(DIET) as f:
        for line in f:
            r = json.loads(line)
            t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
            try:
                ids = tok.encode(t) + [tok.eos_id]
            except ValueError:
                continue
            if len(ids) >= T + 1:
                wins.append(ids[:T + 1])
            if len(wins) == NWIN:
                break
    assert len(wins) == NWIN, f"only {len(wins)} usable rows"
    return torch.tensor(wins, dtype=torch.int64)


def main():
    wins = draw_windows()
    print(f"[diet] {NWIN} windows drawn, ids sha "
          f"{hashlib.sha256(wins.numpy().tobytes()).hexdigest()[:16]}")
    torch.manual_seed(M.SEED)
    m = M.MB()
    names = [n for n, _ in m.param_items()]
    print(f"[diet] V={V} NBLK={M.NBLK} SHIFT={M.SHIFT} params "
          f"{sum(p.numel() for _, p in m.param_items())}")
    ts, td = build_silu_tables()
    t_exp = build_exp_table()
    cos, sin = rope_tables()
    tab = {"silu": ts, "dsilu": td, "exp": t_exp,
           "cos": cos, "sin": sin}
    eye = torch.eye(V, dtype=torch.int64)

    flat = dict(m.param_items())
    wide = {n: flat[n] << M.SHIFT for n in names}
    opt = IntAdamWQw([wide[n] for n in names], M.SHIFT, lrd=1000)
    sched = os.environ.get("SCHED") == "1"
    losses, th = [], hashlib.sha256()
    for step in range(1, STEPS + 1):
        if sched and step in (250, 500, 750):
            opt.lrd *= 2
        w = wins[(step - 1) % NWIN]
        tok_in, tgt = w[:T], w[1:T + 1]
        nar = {n: rdiv(wide[n], 1 << M.SHIFT) for n in names}
        m.emb, m.g_f = nar["emb"], nar["g_f"]
        for i, b in enumerate(m.bodies):
            b.w = {k: nar[f"b{i}.{k}"] for k in M.Body.KEYS}
        lg, cc = m.fwd(tok_in, tab)
        pp = softmax_rows(lg, t_exp)
        losses.append(int((Q - pp[torch.arange(T), tgt]).sum()))
        GG = m.bwd((pp - Q * eye[tgt]) * GBOOST, cc, tab)
        opt.step([rdiv(GG[n], Q * GBOOST) for n in names])
        if step % 125 == 0:
            for n in names:
                th.update(wide[n].numpy().tobytes())
            cyc = sum(losses[-NWIN:]) // NWIN
            print(f"[diet] step {step} loss(win) {losses[-1]} "
                  f"loss(cyc8) {cyc} nz {opt.nz_last:.3f} "
                  f"traj-sha {th.hexdigest()[:16]}", flush=True)
    c0 = sum(losses[:NWIN]) // NWIN
    cf = sum(losses[-NWIN:]) // NWIN
    print(f"[diet] cycle-mean loss {c0} -> {cf}  "
          f"falling: {cf < c0}")
    print(f"[diet] FINAL trajectory sha {th.hexdigest()}")

    if os.environ.get("EXPORT") == "1":
        out = "scratch/detbwd_diet_ref"
        os.makedirs(out, exist_ok=True)
        torch.manual_seed(M.SEED)
        m2 = M.MB()
        with open(f"{out}/diet_init.bin", "wb") as f:
            for _, p in m2.param_items():
                f.write(p.numpy().tobytes())
            f.write(wins.numpy().tobytes())
        ish = hashlib.sha256(
            open(f"{out}/diet_init.bin", "rb").read()).hexdigest()
        print(f"[diet] diet_init.bin sha {ish}")


if __name__ == "__main__":
    main()
