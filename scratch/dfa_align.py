"""WRITER-DFA-1 alignment diagnostic (PRE-REG L68321 item 3, S4 / S9 of
L68543, P2 of L68644): for the DFA arm at each of its 17 snapshots (and
the control arm alongside, descriptively, with the DFA arm's feedback
matrices), on the frozen 256-row probe batch, CPU float64:
delta^DFA_l = B_l e and delta^BP_l = dL / dx_{l+1} (torch.autograd.grad,
read-only). Eligible entries = probe positions carrying a label
(label != -100); masked positions are excluded. Registered number =
cosine of the two flattened eligible-entry vectors per block,
accumulated over 8-row chunks as exact dot products and squared norms;
zero norm of either -> NOT-RESOLVABLE (null), no epsilon. The true
gradient never updates any arm. Writes logs/writerdfa1/align.json
(refuses to overwrite). SMOKE=1: the newest smoke DFA birth, 2 chunks,
appends kind=align to logs/writerdfa1/smoke.jsonl.
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
os.environ["ARM"] = "off"
os.environ.setdefault("BIRTH_SEED", "0")

import torch  # noqa: E402

import train_mathnative as TM  # noqa: E402
from atomtraj_pins import state_digest  # noqa: E402
from dfa_credit import dfa_objective, bp_hidden_errors, feedback_digest  # noqa: E402
from dfa_probe import probe_rows, probe_tensors  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
OUT = Path("logs/writerdfa1")
CHUNK = 8


def load_sd(p):
    d = torch.load(p, map_location="cpu")
    return d["model"] if isinstance(d, dict) and "model" in d else d


def align_snapshot(sd, Bs, tok, rows, n_chunks=None):
    model = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
    model.load_state_dict(sd)
    model = model.double().eval()
    Bd = [B.double() for B in Bs]
    dot, na, nb = [0.0] * 8, [0.0] * 8, [0.0] * 8
    n_elig, loss_sum, n_chunk = 0, 0.0, 0
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    if n_chunks:
        chunks = chunks[:n_chunks]
    for ch in chunks:
        ids, mask, labels = probe_tensors(ch, tok)
        elig = labels != -100
        ob = dfa_objective(model, Bd, ids, mask, labels)
        loss_bp, dbp = bp_hidden_errors(model, ids, mask, labels)
        for l in range(8):
            a = ob["deltas"][l].detach()[elig]
            b = dbp[l][elig]
            dot[l] += float((a * b).sum())
            na[l] += float((a * a).sum())
            nb[l] += float((b * b).sum())
        n_elig += int(elig.sum())
        loss_sum += float(ob["loss"])
        n_chunk += 1
        model.zero_grad(set_to_none=True)
    cos = [(dot[l] / math.sqrt(na[l] * nb[l])) if na[l] > 0 and nb[l] > 0 else None for l in range(8)]
    return {"cos": cos, "norm_dfa": [math.sqrt(x) for x in na], "norm_bp": [math.sqrt(x) for x in nb], "n_eligible": n_elig,
            "probe_loss_mean_chunk": loss_sum / n_chunk, "n_chunks": n_chunk, "state_digest": state_digest(sd)}


def main():
    tok = TM.MathTokenizer()
    row_ids, rows, digest, _ = probe_rows(tok)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    rec = {"prereg": "WRITER-DFA-1", "kind": "align", "smoke": SMOKE, "commit": commit, "probe_token_digest": digest,
           "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "arms": {}}
    src = Path("logs/writerdfa1/smoke.jsonl" if SMOKE else "logs/writerdfa1/births.jsonl")
    births = [json.loads(l) for l in src.open() if '"kind": "birth"' in l]
    if SMOKE:
        dfa = [b for b in births if b["mode"] == "dfa" and b.get("final") and b["emit"]][-1]
        arms = {"DFA": dfa}
        n_chunks = 2
    else:
        target = OUT / "align.json"
        if target.exists():
            raise SystemExit(f"REFUSING: {target} exists")
        by_mode = {b["mode"]: b for b in births if b["phase"] == "disc" and b.get("final")}
        assert set(by_mode) == {"bp", "dfa"}, f"discovery births incomplete: {sorted(by_mode)}"
        arms = {"DFA": by_mode["dfa"], "CTRL": by_mode["bp"]}
        n_chunks = None
    fb = torch.load(Path(arms["DFA"]["outdir"]) / "feedback.pt", map_location="cpu")
    Bs = fb["B"]
    assert feedback_digest(Bs) == arms["DFA"]["feedback"]["digest"], "feedback digest v receipt"
    rec["feedback"] = {"s": fb["s"], "digest": feedback_digest(Bs), "from_arm": "DFA"}
    for name, b in arms.items():
        steps = sorted(int(s) for s in b["snapshots"])
        rec["arms"][name] = {"outdir": b["outdir"], "snapshots": {}}
        for s in steps:
            p = Path(b["outdir"]) / f"step_{s:05d}.pt"
            sd = load_sd(p)
            assert state_digest(sd) == b["snapshots"][str(s)]["state_digest"], f"{p}: digest v receipt"
            r = align_snapshot(sd, Bs, tok, rows, n_chunks)
            rec["arms"][name]["snapshots"][str(s)] = r
            print(f"[align] {name} step {s}: cos {[round(c, 4) if c is not None else None for c in r['cos']]}", flush=True)
    rec["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    if SMOKE:
        with (OUT / "smoke.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print("[align] smoke row appended")
    else:
        target.write_text(json.dumps(rec, indent=1))
        print(f"[align] written {target}")


if __name__ == "__main__":
    main()
