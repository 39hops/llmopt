"""SG-CROSSPOS-REPRESENTABILITY-0: zero-main-model-training desk on the
retained seed-27 frozen-BP control of SYNTHETIC-GRADIENT-WRITER-1 (PRE-REG
SG-CROSSPOS-REPRESENTABILITY-0). Nothing in the main model trains; no gate
is read; no birth.

Question (from OBSERVATION SG-FAILURE-DESK-0): the registered SAME-POSITION
map (x_{l+1,i}, e_i) -> delta^BP_{l,i} is not representable on the healthy
control once trained. Is the true hidden error at position i representable
once the predictor may read the SAME per-block sequence at the positions
j >= i (the positions whose losses backpropagate into x_{l+1,i} through the
causal downstream attention)?

Oracle: a small fixed-capacity REVERSE-CAUSAL sequence predictor per SG
block l in 4..7 and per snapshot, input Z_j = [x_{l+1,j}, e_j] (424 dims,
standardized by FIT statistics and clipped to [-Z_CLIP, Z_CLIP]) at every
real token j of the sequence, target Y_i = delta^BP_{l,i} / s_l (the sealed
arena constants) at the eligible label positions. Query i attends keys
j >= i only (real tokens), with a fixed ALiBi distance bias per head so the
predictor carries no absolute-position table (HELDOUT chunk 7 is longer
than any FIT chunk). No downstream model parameter, attention map,
Jacobian or true hidden error enters the input.
Two arms, same architecture / capacity / optimizer / seed:
  reverse_causal  allowed(i, j) = real(j) and j >= i   (PRIMARY)
  local           allowed(i, j) = j == i               (the same-position
                  ablation: the same trainable family with the cross-
                  position access removed)
Offline fit: AdamW(LR, wd 0), OneCycleLR pct_start 0.05 over EPOCHS epochs
of the FIT chunks in fixed order (one chunk per step), grad clip CLIP,
elementwise MSE over eligible positions and hidden dims (the FOLD B
reduction), float32 CPU, output layer zero-initialised. Stopping rule: the
fixed budget; the readout is the FINAL state (HELDOUT numbers logged every
LOG_EVERY epochs are for the record and never select anything).
Readouts per (snapshot, block, arm): held_ratio = sum ||Y - G||^2 /
sum ||Y||^2 over HELDOUT eligible tokens (the zero-baseline ratio),
held_cos pooled, fit_ratio, held_ratio per HELDOUT chunk (chunk 7 holds
47 % of the HELDOUT tokens and is longer than every FIT chunk), nonfinite
flag. Adjudication (adjudicate): BAR-1 = median over LATE_STEPS x blocks
4..7 of the reverse_causal held_ratio <= 0.5 FIRES; NOT-RESOLVABLE if more
than 4 of those 24 cells are nonfinite.
Snapshots: the control's DESK_STEPS (digests asserted against
logs/sgwriter1/qual.jsonl by sg_failure_desk.load_state). Writes
logs/sgxpos0/desk.json (refuses to overwrite) + a summary row to
logs/sgxpos0/desk.jsonl.
SMOKE=1: a freshly built random model (no checkpoint is read), FIT chunk 0,
HELDOUT chunk 1, EPOCHS 3, logs/sgxpos0/smoke<SMOKE_TAG>.jsonl only.
Usage: .venv/bin/python scratch/sg_crosspos_desk.py
"""
import datetime
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
os.environ.setdefault("ARM", "off")
os.environ.setdefault("BIRTH_SEED", "0")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train_mathnative as TM  # noqa: E402
from dfa_credit import ce_loss, freeze_lower  # noqa: E402
from dfa_probe import probe_rows, probe_tensors  # noqa: E402
from sg_credit import teacher_targets  # noqa: E402
import sg_failure_desk as FD  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"
SMOKE_TAG = os.environ.get("SMOKE_TAG", "")
assert not SMOKE_TAG or SMOKE, "SMOKE_TAG is smoke-only"
OUT_DIR = Path("logs/sgxpos0")
OUT = OUT_DIR / (f"smoke{SMOKE_TAG}.jsonl" if SMOKE else "desk.json")
BIRTHS = Path("logs/sgwriter1/qual.jsonl")
CONTROL_CELL = "sgq_control_s27_lr0.0003"
CHUNK = 32
FIT_CHUNKS = [0] if SMOKE else [0, 2, 4, 6]
HELD_CHUNKS = [1] if SMOKE else [1, 3, 5, 7]
DESK_STEPS = [0] if SMOKE else [0, 463, 1028, 2056, 3084, 5140, 7196, 10280, 12336, 15420]
LATE_STEPS = [3084, 5140, 7196, 10280, 12336, 15420]
SG_BLOCKS = [4, 5, 6, 7]
ARMS = ["reverse_causal", "local"]
D_IN, D_OUT = 384 + 40, 384
D_PRED, N_LAYERS, N_HEADS, D_FFN = 128, 2, 4, 512
EPOCHS = 3 if SMOKE else 300
LR, WD, PCT_START, CLIP = 1e-3, 0.0, 0.05, 1.0
Z_CLIP = 5.0
SEED_BASE = 4242
LOG_EVERY = 1 if SMOKE else 25
BAR_GOOD = 0.5
MAX_NONFINITE = 4
SEED = 27


def alibi_slopes(n_heads):
    return torch.tensor([2.0 ** (-8.0 * (h + 1) / n_heads) for h in range(n_heads)])


def allowed_mask(arm, key_real):
    """[B, T, T] bool: query i may read key j."""
    B, T = key_real.shape
    i = torch.arange(T).view(T, 1)
    j = torch.arange(T).view(1, T)
    if arm == "reverse_causal":
        m = (j >= i).unsqueeze(0) & key_real.unsqueeze(1)
        m = m | torch.eye(T, dtype=torch.bool).unsqueeze(0)     # a pad query keeps itself (no empty softmax row)
    elif arm == "local":
        m = torch.eye(T, dtype=torch.bool).unsqueeze(0).expand(B, T, T)
    else:
        raise ValueError(arm)
    return m


class Attn(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.register_buffer("slopes", alibi_slopes(n_heads), persistent=False)

    def forward(self, x, allowed):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).unbind(2)
        s = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.dh)
        dist = (torch.arange(T).view(1, T) - torch.arange(T).view(T, 1)).abs().to(x.dtype)   # |j - i|
        s = s - self.slopes.view(1, self.h, 1, 1) * dist.view(1, 1, T, T)
        s = s.masked_fill(~allowed.unsqueeze(1), float("-inf"))
        a = torch.softmax(s, -1)
        return self.o(torch.einsum("bhij,bjhd->bihd", a, v).reshape(B, T, d))


class SeqSG(nn.Module):
    """Fixed-capacity sequence predictor: in_proj -> N_LAYERS x (pre-LN attn +
    pre-LN FFN) -> zero-initialised out_proj."""

    def __init__(self, d_in=D_IN, d=D_PRED, n_layers=N_LAYERS, n_heads=N_HEADS, d_ffn=D_FFN, d_out=D_OUT):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.att = nn.ModuleList([Attn(d, n_heads) for _ in range(n_layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d, d_ffn), nn.GELU(), nn.Linear(d_ffn, d)) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(d)
        self.out = nn.Linear(d, d_out)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z, allowed):
        x = self.inp(z)
        for ln1, att, ln2, ffn in zip(self.ln1, self.att, self.ln2, self.ffn):
            x = x + att(ln1(x), allowed)
            x = x + ffn(ln2(x))
        return self.out(self.lnf(x))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def seq_arrays(model, tok, rows, chunk_ids, consts):
    """Per chunk: full padded sequences H[l] [B, T, 384] (x_{l+1}), E [B, T, 40]
    (dL/dlogits, zero at ineligible positions), Y[l] = delta^BP_l / s_l,
    elig [B, T], real [B, T] (the attention key mask). float32."""
    out = []
    for ci in chunk_ids:
        ch = rows[ci * CHUNK:(ci + 1) * CHUNK]
        ids, mask, labels = probe_tensors(ch, tok)
        elig = labels != -100
        logits_T, outs_T = teacher_targets(model, ids, mask)
        loss = ce_loss(logits_T, labels)
        grads = torch.autograd.grad(loss, [outs_T[l] for l in SG_BLOCKS] + [logits_T])
        out.append({"chunk": ci, "H": {l: outs_T[l].detach().float() for l in SG_BLOCKS},
                    "Y": {l: (g.detach() / float(consts[str(l)])).float() for g, l in zip(grads[:-1], SG_BLOCKS)},
                    "E": grads[-1].detach().float(), "elig": elig, "real": mask.bool(), "n_elig": int(elig.sum()), "ce": float(loss.detach())})
    return out


def fit_stats(fit, l):
    Z = torch.cat([torch.cat([c["H"][l], c["E"]], -1)[c["elig"]] for c in fit])
    return Z.mean(0, keepdim=True), Z.std(0, keepdim=True).clamp_min(1e-6)


def inputs(c, l, mu, sd):
    return ((torch.cat([c["H"][l], c["E"]], -1) - mu) / sd).clamp(-Z_CLIP, Z_CLIP)


def sse_ssy(pred, c, l):
    e = c["elig"]
    return float(((pred - c["Y"][l]) ** 2)[e].sum()), float((c["Y"][l] ** 2)[e].sum()), float((pred * c["Y"][l])[e].sum()), float((pred ** 2)[e].sum())


def evaluate(pred_model, chunks, l, mu, sd, arm):
    pred_model.eval()
    tot = [0.0, 0.0, 0.0, 0.0]
    per_chunk = {}
    with torch.no_grad():
        for c in chunks:
            G = pred_model(inputs(c, l, mu, sd), allowed_mask(arm, c["real"]))
            a = sse_ssy(G, c, l)
            per_chunk[str(c["chunk"])] = a[0] / a[1] if a[1] > 0 else None
            tot = [t + x for t, x in zip(tot, a)]
    pred_model.train()
    sse, ssy, dot, ssg = tot
    return {"ratio": sse / ssy if ssy > 0 else None, "cos": dot / math.sqrt(ssy * ssg) if ssy > 0 and ssg > 0 else None, "per_chunk": per_chunk}


def fit_arm(fit, held, l, arm, seed):
    """One offline fit. Returns the receipt dict for (block, arm)."""
    torch.manual_seed(seed)
    mu, sd = fit_stats(fit, l)
    pm = SeqSG()
    opt = torch.optim.AdamW(pm.parameters(), lr=LR, weight_decay=WD)
    total = EPOCHS * len(fit)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total, pct_start=PCT_START)
    masks = [allowed_mask(arm, c["real"]) for c in fit]
    curve = []
    nonfinite = False
    t0 = time.time()
    for ep in range(EPOCHS):
        for c, m in zip(fit, masks):
            G = pm(inputs(c, l, mu, sd), m)
            e = c["elig"]
            loss = ((G - c["Y"][l]) ** 2)[e].mean()          # elementwise MSE over eligible positions x hidden dims
            if not torch.isfinite(loss):
                nonfinite = True
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pm.parameters(), CLIP)
            opt.step()
            sched.step()
        if nonfinite:
            break
        if (ep + 1) % LOG_EVERY == 0 or ep + 1 == EPOCHS:
            curve.append({"epoch": ep + 1, "fit_ratio": evaluate(pm, fit, l, mu, sd, arm)["ratio"], "held_ratio": evaluate(pm, held, l, mu, sd, arm)["ratio"]})
    fr = evaluate(pm, fit, l, mu, sd, arm)
    hr = evaluate(pm, held, l, mu, sd, arm)
    return {"arm": arm, "seed": seed, "nonfinite": nonfinite, "epochs_run": ep + 1, "held_ratio": hr["ratio"], "held_cos": hr["cos"], "held_ratio_per_chunk": hr["per_chunk"],
            "fit_ratio": fr["ratio"], "fit_cos": fr["cos"], "curve": curve, "n_params": n_params(pm), "wall_s": round(time.time() - t0, 1)}


def adjudicate(rec):
    """BAR-1 over LATE_STEPS x SG_BLOCKS of the control's reverse_causal
    held_ratio: FIRES if the median <= BAR_GOOD; NOT-RESOLVABLE if more than
    MAX_NONFINITE cells are nonfinite / missing. Descriptive beside it: the
    local arm's median and the per-cell gap."""
    st = rec["states"]
    rc, loc, bad = [], [], []
    for s in LATE_STEPS:
        for l in SG_BLOCKS:
            cell = st.get(str(s), {}).get("blocks", {}).get(str(l))
            r = None if cell is None else cell["reverse_causal"]["held_ratio"]
            if cell is None or cell["reverse_causal"]["nonfinite"] or r is None or not math.isfinite(r):
                bad.append([s, l])
                continue
            rc.append(r)
            lr_ = cell["local"]["held_ratio"]
            loc.append(lr_ if (lr_ is not None and math.isfinite(lr_) and not cell["local"]["nonfinite"]) else None)
    out = {"n_late_cells": len(LATE_STEPS) * len(SG_BLOCKS), "n_not_resolvable": len(bad), "not_resolvable_cells": bad,
           "late_median_reverse_causal": statistics.median(rc) if rc else None,
           "late_median_local": statistics.median([x for x in loc if x is not None]) if any(x is not None for x in loc) else None,
           "late_min_reverse_causal": min(rc) if rc else None, "late_max_reverse_causal": max(rc) if rc else None}
    if len(bad) > MAX_NONFINITE:
        out["bar_1"] = "NOT-RESOLVABLE"
    else:
        out["bar_1"] = "FIRES" if out["late_median_reverse_causal"] <= BAR_GOOD else "NO-FIRE"
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SMOKE and OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    if not SMOKE and dirty:
        raise SystemExit("REFUSING: registered desk on a dirty tree")
    torch.use_deterministic_algorithms(True)
    tok = TM.MathTokenizer()
    _, rows, probe_digest, _ = probe_rows(tok)
    births = [json.loads(l) for l in BIRTHS.open() if '"kind": "birth"' in l]
    births = [b for b in births if b["seed"] == SEED and b["mode"] in ("zero", "sg") and b.get("final")]
    ctrl = next(b for b in births if b["cell"] == CONTROL_CELL)
    consts = next(b["constants"] for b in births if b["mode"] == "sg")
    rec = {"prereg": "SG-CROSSPOS-REPRESENTABILITY-0", "kind": "sg_crosspos_desk", "smoke": SMOKE, "commit": commit, "tree_dirty": dirty,
           "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "credit_source_sha256": hashlib.sha256(Path("scratch/sg_credit.py").read_bytes()).hexdigest(),
           "failure_desk_source_sha256": hashlib.sha256(Path("scratch/sg_failure_desk.py").read_bytes()).hexdigest(),
           "probe_token_digest": probe_digest, "cell": CONTROL_CELL if not SMOKE else "random_init_smoke", "fit_chunks": FIT_CHUNKS, "held_chunks": HELD_CHUNKS,
           "desk_steps": DESK_STEPS, "late_steps": LATE_STEPS, "arms": ARMS, "arch": {"d_in": D_IN, "d": D_PRED, "layers": N_LAYERS, "heads": N_HEADS, "ffn": D_FFN, "d_out": D_OUT, "alibi_slopes": alibi_slopes(N_HEADS).tolist(), "n_params": n_params(SeqSG())},
           "fit": {"epochs": EPOCHS, "lr": LR, "wd": WD, "pct_start": PCT_START, "clip": CLIP, "z_clip": Z_CLIP, "seed_base": SEED_BASE, "steps_per_epoch": len(FIT_CHUNKS), "loss": "elementwise MSE over eligible positions x hidden dims"},
           "constants": consts, "dtype": "float32", "device": "cpu", "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
           "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "states": {}}
    t0 = time.time()
    for step in DESK_STEPS:
        if SMOKE:
            torch.manual_seed(0)
            model = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536).double().train()
            freeze_lower(model, 4)
            path, digest = "random_init", None
        else:
            model, _, path = FD.load_state(ctrl, step, tok)
            digest = ctrl["snapshots"][str(step)]["state_digest"]
        fit = seq_arrays(model, tok, rows, FIT_CHUNKS, consts)
        held = seq_arrays(model, tok, rows, HELD_CHUNKS, consts)
        srec = {"path": path, "state_digest": digest, "n_fit": sum(c["n_elig"] for c in fit), "n_held": sum(c["n_elig"] for c in held),
                "probe_ce_fit": sum(c["ce"] for c in fit) / len(fit), "probe_ce_held": sum(c["ce"] for c in held) / len(held),
                "baseline_mse_held": {str(l): sum(float((c["Y"][l] ** 2)[c["elig"]].sum()) for c in held) / (srec_n := sum(c["n_elig"] for c in held)) / D_OUT for l in SG_BLOCKS}, "blocks": {}}
        del model
        for l in SG_BLOCKS:
            brec = {}
            for ai, arm in enumerate(ARMS):
                brec[arm] = fit_arm(fit, held, l, arm, SEED_BASE + l + 100 * ai)
            srec["blocks"][str(l)] = brec
            print(f"[sgxpos] step {step} block {l}: rc held {brec['reverse_causal']['held_ratio']:.3f} (fit {brec['reverse_causal']['fit_ratio']:.3f}, per chunk {[round(v, 2) for v in brec['reverse_causal']['held_ratio_per_chunk'].values()]}) "
                  f"local held {brec['local']['held_ratio']:.3f} (fit {brec['local']['fit_ratio']:.3f}) walls {brec['reverse_causal']['wall_s']}/{brec['local']['wall_s']}s ({time.time() - t0:.0f}s)", flush=True)
        rec["states"][str(step)] = srec
    rec["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["adjudication"] = adjudicate(rec) if not SMOKE else None
    if SMOKE:
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print("[sgxpos] smoke row appended")
    else:
        OUT.write_text(json.dumps(rec, indent=1))
        with (OUT_DIR / "desk.jsonl").open("a") as f:
            f.write(json.dumps({k: v for k, v in rec.items() if k != "states"} | {"n_states": len(rec["states"])}) + "\n")
        print(f"[sgxpos] written {OUT}; BAR-1 {rec['adjudication']['bar_1']} (late median rc {rec['adjudication']['late_median_reverse_causal']}, local {rec['adjudication']['late_median_local']})")


if __name__ == "__main__":
    main()
