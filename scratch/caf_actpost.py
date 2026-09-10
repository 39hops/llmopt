"""DFA-LOWER-HARM-DESK-0, part 2: the frozen ACT observable (entropy H_l and
effective rank r_l, scratch/dfa_act.py act_vector) on the nine seed-23
frontier arms at six snapshots (0, 463, 1028, 3084, 8224, 15420), and the
sealed alignment A_l / magnitude ratio Q_l (scratch/dfa_align.py
align_snapshot) on the three lr 3e-4 hybrids at the same snapshots, defined
only for the DFA blocks 0..7-k (the BP segment's entries are reported as
null). Emphasis: the DFA -> BP boundary, i.e. r at the output of block 7-k
(the BP segment's input) and at the output of block 8-k. Zero training,
CPU float64, frozen 256-row probe. Rows stream to logs/dfaharm0/act_rows.jsonl,
tables to logs/dfaharm0/actpost.json (refuses to overwrite). SMOKE=1:
smoke hybrid k = 2 and zero k = 2 arms, steps 0 and 300, 2 probe chunks,
receipt logs/dfaharm0/smoke.jsonl.
"""
import datetime
import json
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
from dfa_act import act_vector, dist  # noqa: E402
from dfa_align import align_snapshot, load_sd  # noqa: E402
from dfa_credit import feedback_digest  # noqa: E402
from dfa_probe import probe_rows  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
OUT = Path("logs/dfaharm0")
STEPS = [0, 300] if SMOKE else [0, 463, 1028, 3084, 8224, 15420]
QUAL = Path("logs/writercaf1/smoke.jsonl" if SMOKE else "logs/writercaf1/qual.jsonl")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / ("smoke.jsonl" if SMOKE else "actpost.json")
    rows_p = OUT / ("smoke_act_rows.jsonl" if SMOKE else "act_rows.jsonl")
    if not SMOKE and (target.exists() or rows_p.exists()):
        raise SystemExit(f"REFUSING: {target} or {rows_p} exists")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    tok = TM.MathTokenizer()
    row_ids, rows, probe_digest, _ = probe_rows(tok)
    births = [r for r in (json.loads(l) for l in QUAL.open()) if r.get("kind") == "birth" and r.get("final")]
    if SMOKE:
        births = [b for b in births if b["cell"] in ("smoke_hybrid_k2_s11_S1_lr0.0003", "smoke_zero_k2_s11_lr0.0003")]
        n_chunks = 2
    else:
        births = [b for b in births if b["phase"] == "qual"]
        assert len(births) == 9, len(births)
        n_chunks = None
    rec = {"prereg": "DFA-LOWER-HARM-DESK-0", "kind": "actpost", "smoke": SMOKE, "commit": commit, "tree_dirty": dirty, "probe_token_digest": probe_digest,
           "steps": STEPS, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "arms": {}}
    w0 = None
    for b in births:
        k = b["k_bp"]
        arm = {"mode": b["mode"], "k_bp": k, "s": b["s"], "peak_lr": b["peak_lr"], "outdir": b["outdir"], "boundary_block": 7 - k, "snapshots": {}}
        Bs = None
        if b["mode"] == "hybrid":
            fb = torch.load(Path(b["outdir"]) / "feedback.pt", map_location="cpu")
            Bs = fb["B"]
            assert feedback_digest(Bs) == b["feedback"]["digest"]
        for step in STEPS:
            p = Path(b["outdir"]) / f"step_{step:05d}.pt"
            if not p.exists():
                continue
            sd = load_sd(p)
            assert state_digest(sd) == b["snapshots"][str(step)]["state_digest"], f"{p}: digest v receipt"
            if step == 0:
                d0 = state_digest(sd)
                assert w0 is None or w0 == d0, "shared W_0 law"
                w0 = d0
            av = act_vector(sd, tok, rows, n_chunks)
            assert av["selfcheck_ok"]
            row = {"cell": b["cell"], "mode": b["mode"], "k_bp": k, "step": step, "state_digest": state_digest(sd), "H": av["H"], "r": av["r"], "act": av["act"],
                   "r_boundary_in": av["r"][7 - k], "r_boundary_out": av["r"][8 - k] if 8 - k < 8 else None, "act_not_resolvable": av["not_resolvable"]}
            if Bs is not None and b["peak_lr"] == 3e-4:
                al = align_snapshot(sd, Bs, tok, rows, n_chunks)
                nd = 8 - k
                row["A"] = al["cos"][:nd] + [None] * k
                row["Q"] = [(al["norm_dfa"][l] / al["norm_bp"][l]) if al["norm_bp"][l] > 0 else None for l in range(nd)] + [None] * k
                row["norm_bp"] = al["norm_bp"]
                row["probe_loss_mean_chunk"] = al["probe_loss_mean_chunk"]
            with rows_p.open("a") as f:
                f.write(json.dumps({**row, "commit": commit}) + "\n")
            arm["snapshots"][str(step)] = row
            print(f"[actpost] {b['cell']} step {step}: r {[round(x, 1) for x in av['r']]} H {[round(h, 2) for h in av['H']]}", flush=True)
        rec["arms"][b["cell"]] = arm
    rec["w0_state_digest"] = w0
    # summaries: r at the boundary per arm and step; ACT distance hybrid v zero at matched k and step
    cells = list(rec["arms"])
    rec["boundary_rank"] = {c: {s: [rec["arms"][c]["snapshots"][s]["r_boundary_in"], rec["arms"][c]["snapshots"][s]["r_boundary_out"]] for s in rec["arms"][c]["snapshots"]} for c in cells}
    pairs = {}
    for c in cells:
        a = rec["arms"][c]
        if a["mode"] != "hybrid" or a["peak_lr"] != 3e-4:
            continue
        z = next((cz for cz in cells if rec["arms"][cz]["mode"] == "zero" and rec["arms"][cz]["k_bp"] == a["k_bp"]), None)
        if z:
            pairs[f"{c}|{z}"] = {s: dist({"act": a["snapshots"][s]["act"]}, {"act": rec["arms"][z]["snapshots"][s]["act"]}) for s in a["snapshots"] if s in rec["arms"][z]["snapshots"]}
    rec["act_dist_hybrid_v_zero"] = pairs
    rec["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    if SMOKE:
        with target.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print("[actpost] smoke row appended")
    else:
        target.write_text(json.dumps(rec, indent=1))
        print("[actpost] written", target)


if __name__ == "__main__":
    main()
