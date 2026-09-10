"""SG-PREDICTOR-AUDIT-0: zero-main-model-training audit of the true
backprop hidden-error targets delta^BP_l = dL/dx_{l+1} on the retained
FROZEN-BACKBONE-1 states (PRE-REG SG-PREDICTOR-AUDIT-0). Nothing trains;
no gate is read; predictor capacity is not chosen here.

States (12): the six frozenbb1 arms (FULL / FROZEN at seeds 24, 25, 26) at
step_00463 and final. FULL states: blocks 0..7 inspected. FROZEN states:
blocks 4..7 inspected (the qualification arena; blocks 0..3 sit at W_0).

Probe and grain: the frozen WRITER-DFA-1 probe (256 rows, token digest
asserted against logs/writerdfa1/probe.json), 32-row chunks = the training
batch grain, so the mean-CE loss and therefore delta^BP carry exactly the
per-token scale a training step sees; float64 on CPU; label positions
only (labels != -100, the masking convention of dfa_align.align_snapshot).

Per state and inspected block l, over all eligible tokens of all chunks:
delta^BP_{l,t} (384-vector per token): element RMS, total Frobenius norm,
per-token-norm quantiles q01 / q10 / q50 / q90 / q99, min, max, dynamic
range log10(q99 / q01), exact-zero element count, exact-zero per-token
norm count, nonfinite element count; the same reductions for the block
output h_{l+1,t}; RMS(delta) / RMS(h) ratio. Per state: the output error
e_t = dL/dlogits (40-vector per token) with the same reductions, the probe
loss (mean over chunks), n_eligible. Normalization constants per the
registered rule (derived here, adopted or not by the SG amendment):
s_l = geometric mean over the FROZEN states (463 + final, three seeds) of
RMS_l(delta) for l in 4..7 (arena constants) and over the FULL states of
RMS_l(delta) for l in 0..7 (full-stack constants).

Writes logs/sgaudit0/audit.json (refuses to overwrite) and one row to
logs/sgaudit0/audit.jsonl. SMOKE=1: the two frozenbb1_smoke finals, 2
chunks, logs/sgaudit0/smoke.jsonl only.

Usage: .venv/bin/python scratch/sg_target_audit.py
"""
import datetime
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
os.environ.setdefault("ARM", "off")
os.environ.setdefault("BIRTH_SEED", "0")

import torch  # noqa: E402

import train_mathnative as TM  # noqa: E402
from atomtraj_pins import state_digest  # noqa: E402
from dfa_credit import causal_mask, ce_loss  # noqa: E402
from dfa_probe import probe_rows, probe_tensors  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"
CHUNK = 32
SEEDS = (24, 25, 26)
STEPS = ("step_00463", "final")
BIRTHS = Path("logs/frozenbb1/smoke.jsonl" if SMOKE else "logs/frozenbb1/births.jsonl")
OUT_DIR = Path("logs/sgaudit0")
OUT = OUT_DIR / ("smoke.jsonl" if SMOKE else "audit.json")
QS = (0.01, 0.10, 0.50, 0.90, 0.99)


def arena_blocks(arm):
    return list(range(8)) if arm == "FULL" else [4, 5, 6, 7]


def forward_with_hidden(model, ids, mask):
    """Attached forward returning (outs, logits): outs[l] = x_{l+1}."""
    m = causal_mask(ids, mask)
    x = model.emb(ids)
    outs = []
    for b in model.blocks:
        x, _ = b(x, m, None)
        outs.append(x)
    return outs, model.head(model.norm(x))


def reduce_vectors(v):
    """v: (N, D) float64, one row per eligible token. Returns the registered
    reductions (RMS of elements, Frobenius norm, per-token-norm quantiles,
    dynamic range, exact-zero and nonfinite counts)."""
    n, d = v.shape
    finite = torch.isfinite(v)
    nonfinite = int((~finite).sum())
    vf = torch.where(finite, v, torch.zeros_like(v))
    tn = vf.norm(dim=1)
    q = torch.quantile(tn, torch.tensor(QS, dtype=v.dtype)).tolist() if n else [float("nan")] * len(QS)
    q01, q99 = q[0], q[-1]
    return {"n_tokens": n, "dim": d,
            "rms": math.sqrt(float((vf * vf).sum()) / (n * d)) if n else float("nan"),
            "fro": float(vf.norm()),
            "token_norm_q": dict(zip([f"q{int(x * 100):02d}" for x in QS], q)),
            "token_norm_min": float(tn.min()) if n else float("nan"), "token_norm_max": float(tn.max()) if n else float("nan"),
            "dynamic_range_log10": (math.log10(q99 / q01) if q01 > 0 and q99 > 0 else None),
            "zero_elements": int((v == 0).sum()), "zero_token_norms": int((tn == 0).sum()), "nonfinite_elements": nonfinite}


def audit_state(sd, tok, rows, blocks, n_chunks=None):
    model = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
    model.load_state_dict(sd)
    model = model.double().eval()
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    if n_chunks:
        chunks = chunks[:n_chunks]
    d_acc = {l: [] for l in blocks}
    h_acc = {l: [] for l in blocks}
    e_acc, losses, n_elig = [], [], 0
    for ch in chunks:
        ids, mask, labels = probe_tensors(ch, tok)
        elig = labels != -100
        outs, logits = forward_with_hidden(model, ids, mask)
        loss = ce_loss(logits, labels)
        grads = torch.autograd.grad(loss, [outs[l] for l in blocks] + [logits])
        for g, l in zip(grads[:-1], blocks):
            d_acc[l].append(g.detach()[elig])
            h_acc[l].append(outs[l].detach()[elig])
        e_acc.append(grads[-1].detach()[elig])
        losses.append(float(loss.detach()))
        n_elig += int(elig.sum())
        model.zero_grad(set_to_none=True)
    per_block = {}
    for l in blocks:
        rd = reduce_vectors(torch.cat(d_acc[l]))
        rh = reduce_vectors(torch.cat(h_acc[l]))
        per_block[str(l)] = {"delta": rd, "h": rh, "rms_ratio_delta_over_h": (rd["rms"] / rh["rms"]) if rh["rms"] > 0 else None}
    return {"blocks": per_block, "e": reduce_vectors(torch.cat(e_acc)), "probe_loss_mean_chunk": sum(losses) / len(losses),
            "n_eligible": n_elig, "n_chunks": len(chunks), "state_digest": state_digest(sd)}


def geo_mean(xs):
    xs = [x for x in xs if x is not None and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def constants(states):
    """Registered rule: s_l = geometric mean of RMS_l(delta) over the FROZEN
    states for l in 4..7 (arena) and over the FULL states for l in 0..7."""
    out = {"arena_frozen_4_7": {}, "full_stack_0_7": {}}
    for l in range(4, 8):
        out["arena_frozen_4_7"][str(l)] = geo_mean([s["blocks"][str(l)]["delta"]["rms"] for s in states if s["arm"] == "FROZEN"])
    for l in range(8):
        out["full_stack_0_7"][str(l)] = geo_mean([s["blocks"][str(l)]["delta"]["rms"] for s in states if s["arm"] == "FULL"])
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SMOKE and OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    if not SMOKE and dirty:
        raise SystemExit("REFUSING: registered audit on a dirty tree")
    tok = TM.MathTokenizer()
    row_ids, rows, probe_digest, _ = probe_rows(tok)
    births = [json.loads(l) for l in BIRTHS.open() if '"kind": "birth"' in l]
    if SMOKE:
        births = [b for b in births if b.get("smoke") and b.get("final")]
        arms = {}
        for b in births:
            arms[("FULL" if b["mode"] == "bp" else "FROZEN", 11)] = b
        plan = [(arm, s, "final") for (arm, s) in sorted(arms)]
        n_chunks = 2
    else:
        arms = {(b["arm"], b["seed"]): b for b in births if b["phase"] == "fb"}
        assert len(arms) == 6, sorted(arms)
        plan = [(arm, s, st) for s in SEEDS for arm in ("FULL", "FROZEN") for st in STEPS]
        n_chunks = None
    rec = {"prereg": "SG-PREDICTOR-AUDIT-0", "kind": "sg_target_audit", "smoke": SMOKE, "commit": commit, "tree_dirty": dirty,
           "probe_token_digest": probe_digest, "probe_rows": len(rows), "chunk_rows": CHUNK, "dtype": "float64", "device": "cpu",
           "masking": "label positions only (labels != -100); mean CE over eligible tokens of a 32-row chunk",
           "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "states": []}
    for arm, seed, st in plan:
        b = arms[(arm, seed)]
        p = Path(b["outdir"]) / f"{st}.pt"
        sd = torch.load(p, map_location="cpu")
        expect = b["final"]["state_digest"] if st == "final" else b["snapshots"][str(int(st.split("_")[1]))]["state_digest"]
        assert state_digest(sd) == expect, f"{p}: digest v receipt"
        r = audit_state(sd, tok, rows, arena_blocks(arm), n_chunks)
        r.update({"arm": arm, "seed": seed, "state": st, "path": str(p), "cell": b["cell"]})
        rec["states"].append(r)
        summ = {l: f"{v['delta']['rms']:.3e}" for l, v in r["blocks"].items()}
        print(f"[sgaudit] {arm} s{seed} {st}: loss {r['probe_loss_mean_chunk']:.4f} n_elig {r['n_eligible']} RMS(delta) {summ} "
              f"nonfinite {sum(v['delta']['nonfinite_elements'] for v in r['blocks'].values())}", flush=True)
    rec["normalization_constants"] = constants(rec["states"])
    rec["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    if SMOKE:
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print("[sgaudit] smoke row appended")
    else:
        OUT.write_text(json.dumps(rec, indent=1))
        with (OUT_DIR / "audit.jsonl").open("a") as f:
            f.write(json.dumps({k: v for k, v in rec.items() if k != "states"} | {"n_states": len(rec["states"])}) + "\n")
        print(f"[sgaudit] written {OUT}")


if __name__ == "__main__":
    main()
