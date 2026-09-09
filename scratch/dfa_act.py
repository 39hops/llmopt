"""WRITER-DFA-1 ACT observable (PRE-REG L68321 item 6 (vii), S10 of
L68543, P3 of L68644): on the frozen 256-row probe batch, teacher-forced,
CPU float64, for each block l: H_l = mean over the 6 heads and the
unmasked input tokens with position 0 excluded of the entropy (nats) of
the explicitly recomputed causal softmax attention row (keys <= t,
1 / sqrt(d_head) scaling, key padding masked); r_l = effective rank of
the residual-stream covariance at the block output over the unmasked
input tokens (position 0 included): center by the feature mean,
C = (1/N) sum (x - mu)(x - mu)^T, lambda = eigvalsh(C) in float64,
negative eigenvalues with |lambda| <= 1e-10 lambda_max clamped to 0
(any more negative -> NOT-RESOLVABLE), p = lambda / sum lambda,
r = exp(-sum p log p) (0 log 0 = 0); zero mass -> NOT-RESOLVABLE.
ACT = (H_0..H_7, r_0..r_7) in R^16; distances are L2. The manual
forward is self-checked against the model's own forward in float64
(max abs logit diff recorded, must be <= 1e-8).

ACT_SET=envelope  A, B, N1, N2, N3, N4 finals -> logs/writerdfa1/act_envelope.json
                  (the OBSERVATION booked before any DFA snapshot is read)
ACT_SET=dfa       DFA and CTRL finals (from logs/writerdfa1/births.jsonl)
                  -> logs/writerdfa1/act.json with ACT-1 against the booked numbers
SMOKE=1           the newest smoke birth with a final.pt, 2 chunks only,
                  appends kind=act to logs/writerdfa1/smoke.jsonl
"""
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
os.environ["ARM"] = "off"
os.environ.setdefault("BIRTH_SEED", "0")

import torch  # noqa: E402

import train_mathnative as TM  # noqa: E402
from atomtraj_pins import state_digest  # noqa: E402
from dfa_credit import causal_mask  # noqa: E402
from dfa_probe import probe_rows, probe_tensors  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
ACT_SET = os.environ.get("ACT_SET", "envelope")
OUT = Path("logs/writerdfa1")
REPAIR = Path("/Users/artin/code/llmopt-repair")
CHUNK = 32
HEADS = 6
EIG_REL_TOL = 1e-10
SELFCHECK_TOL = 1e-8
ENVELOPE_SPECIMENS = {
    "A": "checkpoints/gallery19m_phase_s2.pt", "B": "checkpoints/gallery19m_backsched_s2.pt",
    "N1": "checkpoints/atomtraj1/stock_s6/step_15420.pt", "N2": str(REPAIR / "checkpoints/atomtraj1/stock_s6/step_15420.pt"),
    "N3": "checkpoints/atomtraj1/stock_s7/step_15420.pt", "N4": str(REPAIR / "checkpoints/atomtraj1/stock_s7/step_15420.pt"),
}


def load_sd(p):
    d = torch.load(p, map_location="cpu")
    return d["model"] if isinstance(d, dict) and "model" in d else d


def rope(q, k, pos0=0):
    """Verbatim formula of build_model's rope (float32 angle table, as the model computes it)."""
    B, H, T, D = q.shape
    half = D // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=q.device) / half)
    t = torch.arange(pos0, pos0 + T, device=q.device)
    ang = t[:, None] * freq[None, :]
    cos, sin = ang.cos(), ang.sin()

    def rot(v):
        v1, v2 = v[..., :half], v[..., half:]
        return torch.cat([v1 * cos - v2 * sin, v1 * sin + v2 * cos], -1)
    return rot(q), rot(k)


def block_manual(b, x, m):
    """Block.forward with the softmax written out; returns (x_out, entropy (B, H, T))."""
    B_, T, d = x.shape
    h = b.n1(x)
    q, k, v = b.qkv(h).chunk(3, -1)
    q = q.view(B_, T, HEADS, -1).transpose(1, 2)
    k = k.view(B_, T, HEADS, -1).transpose(1, 2)
    v = v.view(B_, T, HEADS, -1).transpose(1, 2)
    q, k = rope(q, k, 0)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~m, float("-inf"))
    p = scores.softmax(-1)
    ent = -(p * torch.log(p.clamp_min(1e-300))).sum(-1)
    a = (p @ v).transpose(1, 2).reshape(B_, T, d)
    x = x + b.o(a)
    h2 = b.n2(x)
    x = x + b.down(torch.nn.functional.silu(b.gate(h2)) * b.up(h2))
    return x, ent


def effective_rank(n, sx, sxx):
    if n == 0:
        return None, "no tokens"
    mu = sx / n
    Cov = sxx / n - torch.outer(mu, mu)
    Cov = 0.5 * (Cov + Cov.t())
    lam = torch.linalg.eigvalsh(Cov)
    lmax = float(lam.max())
    if lmax <= 0:
        return None, "zero mass"
    neg = lam[lam < 0]
    if len(neg) and float(neg.min()) < -EIG_REL_TOL * lmax:
        return None, f"negative eigenvalue {float(neg.min())} below tolerance"
    lam = lam.clamp_min(0)
    tot = float(lam.sum())
    if tot <= 0:
        return None, "zero mass"
    p = lam / tot
    p = p[p > 0]
    return float(torch.exp(-(p * torch.log(p)).sum())), None


def act_vector(sd, tok, rows, n_chunks=None):
    model = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
    model.load_state_dict(sd)
    model = model.double().eval()
    ent_sum, ent_n = [0.0] * 8, [0] * 8
    n = [0] * 8
    sx = [torch.zeros(384, dtype=torch.float64) for _ in range(8)]
    sxx = [torch.zeros(384, 384, dtype=torch.float64) for _ in range(8)]
    selfcheck = 0.0
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    if n_chunks:
        chunks = chunks[:n_chunks]
    with torch.no_grad():
        for ch in chunks:
            ids, mask, _ = probe_tensors(ch, tok)
            m = causal_mask(ids, mask)
            elig = mask.bool()
            elig_ent = elig.clone()
            elig_ent[:, 0] = False
            x = model.emb(ids)
            for l, b in enumerate(model.blocks):
                x, ent = block_manual(b, x, m)
                e_sel = ent[:, :, :][elig_ent.unsqueeze(1).expand(-1, HEADS, -1)]
                ent_sum[l] += float(e_sel.sum())
                ent_n[l] += int(e_sel.numel())
                xs = x[elig]
                n[l] += int(xs.shape[0])
                sx[l] += xs.sum(0)
                sxx[l] += xs.t() @ xs
            logits_manual = model.head(model.norm(x))
            logits_model = model(ids, mask)
            selfcheck = max(selfcheck, float((logits_manual - logits_model).abs().max()))
    H = [ent_sum[l] / ent_n[l] for l in range(8)]
    r, notes = [], {}
    for l in range(8):
        v, why = effective_rank(n[l], sx[l], sxx[l])
        r.append(v)
        if why:
            notes[str(l)] = why
    return {"H": H, "r": r, "act": H + r, "n_entropy_terms": ent_n, "n_rank_tokens": n, "selfcheck_max_abs_logit_diff": selfcheck,
            "selfcheck_ok": selfcheck <= SELFCHECK_TOL, "not_resolvable": notes, "n_chunks": len(chunks)}


def dist(u, v):
    if any(x is None for x in u["act"]) or any(x is None for x in v["act"]):
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(u["act"], v["act"])))


def main():
    tok = TM.MathTokenizer()
    row_ids, rows, digest, _ = probe_rows(tok)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    rec = {"prereg": "WRITER-DFA-1", "kind": "act", "act_set": ACT_SET, "smoke": SMOKE, "commit": commit, "probe_token_digest": digest,
           "eig_rel_tol": EIG_REL_TOL, "tree_dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "specimens": {}}
    if SMOKE:
        births = [json.loads(l) for l in Path("logs/writerdfa1/smoke.jsonl").open() if '"kind": "birth"' in l]
        births = [b for b in births if b.get("final")]
        b = births[-1]
        spec = {"SMOKE": str(Path(b["outdir"]) / "final.pt")}
        n_chunks = 2
    elif ACT_SET == "envelope":
        spec = ENVELOPE_SPECIMENS
        n_chunks = None
        target = OUT / "act_envelope.json"
        if target.exists():
            raise SystemExit(f"REFUSING: {target} exists")
    else:
        births = [json.loads(l) for l in Path("logs/writerdfa1/births.jsonl").open() if '"kind": "birth"' in l]
        by_mode = {b["mode"]: b for b in births if b["phase"] == "disc" and b.get("final")}
        assert set(by_mode) == {"bp", "dfa"}, f"discovery births incomplete: {sorted(by_mode)}"
        spec = {"DFA": str(Path(by_mode["dfa"]["outdir"]) / "final.pt"), "CTRL": str(Path(by_mode["bp"]["outdir"]) / "final.pt")}
        n_chunks = None
        target = OUT / "act.json"
        if target.exists():
            raise SystemExit(f"REFUSING: {target} exists")
    for name, p in spec.items():
        if not Path(p).exists():
            raise SystemExit(f"NOT-RUN: missing {p}")
        sd = load_sd(p)
        v = act_vector(sd, tok, rows, n_chunks)
        v.update({"path": p, "file_sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(), "state_digest": state_digest(sd)})
        assert v["selfcheck_ok"], f"{name}: manual forward v model forward differ by {v['selfcheck_max_abs_logit_diff']}"
        rec["specimens"][name] = v
        print(f"[act] {name}: H {[round(h, 4) for h in v['H']]} r {[round(x, 2) if x is not None else None for x in v['r']]}", flush=True)
    S = rec["specimens"]
    if SMOKE:
        rec["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        with (OUT / "smoke.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print("[act] smoke row appended")
        return
    if ACT_SET == "envelope":
        d12, d34, dAB = dist(S["N1"], S["N2"]), dist(S["N3"], S["N4"]), dist(S["A"], S["B"])
        rec["observation"] = {"ACT_dist_N1_N2": d12, "ACT_dist_N3_N4": d34, "ACT_null_envelope": max(d12, d34) if None not in (d12, d34) else None,
                              "ACT_dist_A_B": dAB}
        print("[act] envelope:", json.dumps(rec["observation"]))
    else:
        env = json.load((OUT / "act_envelope.json").open())["observation"]
        d = dist(S["DFA"], S["CTRL"])
        rec["bars"] = {"ACT_dist_DFA_CTRL": d, "ACT_null_envelope_booked": env["ACT_null_envelope"], "ACT_dist_A_B_booked": env["ACT_dist_A_B"],
                       "ACT-1": (d is not None and d > env["ACT_null_envelope"] and d > env["ACT_dist_A_B"]) if d is not None else "NOT-RESOLVABLE"}
        print("[act] bars:", json.dumps(rec["bars"]))
    rec["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    target.write_text(json.dumps(rec, indent=1))
    print(f"[act] written {target}")


if __name__ == "__main__":
    main()
