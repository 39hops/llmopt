"""WHY-DFA-FAILED-DESCRIPTIVE-0: zero-training postmortem of the four
WRITER-DFA-1 qualification arms (seed 21, checkpoints/writerdfa1/qual_*;
VERDICT WRITER-DFA-1 RESULTS L68802). CPU, frozen 256-row probe
(logs/writerdfa1/probe.json), existing snapshots only. For every cell,
every saved snapshot and every block:

  A_l(t) = cos(delta^DFA_l, delta^BP_l)   the sealed alignment reduction
           (scratch/dfa_align.py align_snapshot: float64, label
           positions only, 32-row chunks, zero norm -> null)
  Q_l(t) = ||delta^DFA_l||_2 / ||delta^BP_l||_2   descriptive magnitude
           ratio over the same flattened eligible entries; null
           (NOT-RESOLVABLE) when the BP norm is zero, no epsilon
  ACT(t)  the frozen R^16 observable (scratch/dfa_act.py act_vector);
           distance from the shared seed-21 W_0 (step 0) over time,
           pairwise distances among the four cells at matched steps,
           final vectors
  U_S(t)  learned-update L2 norm ||theta_S(t) - theta_S(0)|| per tensor
           set S in {GLOBAL, BLOCK0..7, OUTSIDE} (scratch/writertraj_census
           flat / SETS), plus the OUTSIDE share of the GLOBAL squared norm

Rows stream to logs/dfapost/rows.jsonl (one per cell x snapshot, kill-safe);
the assembled tables go to logs/dfapost/postmortem.json (refuses to
overwrite). Contextual anchors (the ACT null envelope 8.904 and ACT(A, B)
15.714 of L68742) are copied into the receipt under their own fences
(cross-init v same-init: not a null for these seed-21 cells). No bars.
SMOKE=1: the smoke DFA arm, 2 snapshots, 2 probe chunks, appends to
logs/dfapost/smoke.jsonl.
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
from dfa_align import align_snapshot, load_sd  # noqa: E402
from dfa_act import act_vector, dist  # noqa: E402
from dfa_credit import feedback_digest  # noqa: E402
from dfa_probe import probe_rows  # noqa: E402
from writertraj_census import flat, SETS  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
OUT = Path("logs/dfapost")
SRC = Path("logs/writerdfa1/smoke.jsonl" if SMOKE else "logs/writerdfa1/qual.jsonl")
NORM_SETS = ["GLOBAL", "OUTSIDE"] + [f"BLOCK{l}" for l in range(8)]
ANCHORS = {"ACT_null_envelope_L68742": 8.904018358847024, "ACT_dist_A_B_L68742": 15.714282867272743,
           "fence": "cross-init contextual anchors only; the seed-21 cells share W_0 with each other, not with A, B or the null pairs"}


def update_norms(sd, w0):
    out = {}
    for s in NORM_SETS:
        out[s] = float((flat(sd, SETS[s]) - flat(w0, SETS[s])).norm())
    g2 = out["GLOBAL"] ** 2
    out["OUTSIDE_share_of_global_sq"] = (out["OUTSIDE"] ** 2 / g2) if g2 > 0 else None
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / ("smoke.jsonl" if SMOKE else "postmortem.json")
    rows_p = OUT / ("smoke_rows.jsonl" if SMOKE else "rows.jsonl")
    if not SMOKE and (target.exists() or rows_p.exists()):
        raise SystemExit(f"REFUSING: {target} or {rows_p} exists")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    tok = TM.MathTokenizer()
    row_ids, rows, probe_digest, _ = probe_rows(tok)
    births = [json.loads(l) for l in SRC.open() if '"kind": "birth"' in l]
    if SMOKE:
        births = [b for b in births if b["mode"] == "dfa" and b.get("final") and b["emit"]][-1:]
        n_chunks = 2
    else:
        births = [b for b in births if b["phase"] == "qual" and b.get("final")]
        assert len(births) == 4, len(births)
        n_chunks = None
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rec = {"prereg": "WHY-DFA-FAILED-DESCRIPTIVE-0", "kind": "postmortem", "smoke": SMOKE, "commit": commit, "tree_dirty": dirty,
           "probe_token_digest": probe_digest, "started_utc": started, "anchors": ANCHORS, "cells": {}, "w0": {}}
    # shared W_0 across the cells
    w0_digests = {b["cell"]: b["init_state_digest"] for b in births}
    assert len(set(w0_digests.values())) == 1, w0_digests
    w0 = load_sd(Path(births[0]["outdir"]) / "step_00000.pt")
    assert state_digest(w0) == births[0]["init_state_digest"]
    rec["w0"] = {"state_digest": state_digest(w0), "seed": births[0]["seed"]}
    act_w0 = act_vector(w0, tok, rows, n_chunks)
    assert act_w0["selfcheck_ok"]
    rec["w0"]["act"] = act_w0
    acts = {}   # cell -> step -> act vector
    for b in births:
        cell = b["cell"]
        fb = torch.load(Path(b["outdir"]) / "feedback.pt", map_location="cpu")
        Bs = fb["B"]
        assert feedback_digest(Bs) == b["feedback"]["digest"], f"{cell}: feedback digest v receipt"
        crec = {"outdir": b["outdir"], "s": b["s"], "peak_lr": b["peak_lr"], "final_loss": b.get("final_loss"),
                "feedback_digest": b["feedback"]["digest"], "steps": sorted(int(s) for s in b["snapshots"]), "snapshots": {}}
        acts[cell] = {}
        for step in crec["steps"]:
            p = Path(b["outdir"]) / f"step_{step:05d}.pt"
            sd = load_sd(p)
            assert state_digest(sd) == b["snapshots"][str(step)]["state_digest"], f"{p}: digest v receipt"
            al = align_snapshot(sd, Bs, tok, rows, n_chunks)
            Q = [(al["norm_dfa"][l] / al["norm_bp"][l]) if al["norm_bp"][l] > 0 else None for l in range(8)]
            av = act_vector(sd, tok, rows, n_chunks)
            assert av["selfcheck_ok"], f"{cell} {step}: ACT self-check {av['selfcheck_max_abs_logit_diff']}"
            un = update_norms(sd, w0)
            row = {"cell": cell, "step": step, "state_digest": al["state_digest"], "A": al["cos"], "Q": Q,
                   "norm_dfa": al["norm_dfa"], "norm_bp": al["norm_bp"], "n_eligible": al["n_eligible"], "probe_loss_mean_chunk": al["probe_loss_mean_chunk"],
                   "act": av["act"], "H": av["H"], "r": av["r"], "act_not_resolvable": av["not_resolvable"],
                   "act_dist_from_w0": dist(av, act_w0), "update_norms": un, "commit": commit,
                   "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
            with rows_p.open("a") as f:
                f.write(json.dumps(row) + "\n")
            crec["snapshots"][str(step)] = row
            acts[cell][step] = av
            print(f"[post] {cell} step {step}: A {[round(a, 3) if a is not None else None for a in al['cos']]} "
                  f"Q {[round(q, 2) if q is not None else None for q in Q]} ACTd(W0) {row['act_dist_from_w0']:.3f} "
                  f"|dtheta| G {un['GLOBAL']:.3f} OUT {un['OUTSIDE']:.3f}", flush=True)
        rec["cells"][cell] = crec
    # pairwise ACT at matched steps
    cells = list(acts)
    common = sorted(set.intersection(*[set(acts[c]) for c in cells]))
    rec["pairwise_act"] = {str(s): {f"{a}|{b}": dist(acts[a][s], acts[b][s]) for i, a in enumerate(cells) for b in cells[i + 1:]} for s in common}
    rec["final_act"] = {c: acts[c][max(acts[c])]["act"] for c in cells}
    rec["tables"] = {"A": {c: {str(s): rec["cells"][c]["snapshots"][str(s)]["A"] for s in rec["cells"][c]["steps"]} for c in cells},
                     "Q": {c: {str(s): rec["cells"][c]["snapshots"][str(s)]["Q"] for s in rec["cells"][c]["steps"]} for c in cells},
                     "act_dist_from_w0": {c: {str(s): rec["cells"][c]["snapshots"][str(s)]["act_dist_from_w0"] for s in rec["cells"][c]["steps"]} for c in cells},
                     "update_norms": {c: {str(s): rec["cells"][c]["snapshots"][str(s)]["update_norms"] for s in rec["cells"][c]["steps"]} for c in cells}}
    rec["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rec["rows_sha256"] = hashlib.sha256(rows_p.read_bytes()).hexdigest()
    if SMOKE:
        with target.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print("[post] smoke row appended")
    else:
        target.write_text(json.dumps(rec, indent=1))
        print(f"[post] written {target}")


if __name__ == "__main__":
    main()
