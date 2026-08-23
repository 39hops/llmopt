"""MATH-CYBER-1 theta_0 CALIBRATION liveness qualification
(PRE-REG MATH-CYBER-1-THETA0-BIRTH-0). Frozen test: CALIBRATION
seeds 9100-9109 x L4-7 (40 episodes), 12-decision budget, 60 s
per-episode wall safety cap; policy = serial B=1 full
teacher-forced scoring of every legal candidate (score = sum
log p over child tokens incl. terminating newline), argmax,
tie-break (score, rule name, child key); overflow law: any
candidate sequence > 4096 tokens -> decision unscorable, episode
model_ctx_overflow. NO hce anywhere in the model arm. Bar:
>= 1/40 solved; full result reported either way.

World-snapshot discipline: fresh process, clean caches, idle
machine (operator-enforced); per decision the COMPLETE legal set
is enumerated ONCE through a state-keyed cache, its
(state_hash, ordered_legal_set_hash) recorded at first
materialization and immutable after; enumeration completes
before scoring begins at each decision (single-threaded driver —
no overlap by construction). SMOKE=1: 2 L4 episodes, smoke_
paths.

Receipts: logs/mathworld1/{PRE}liveness.jsonl (per-decision rows
+ per-episode rows + meta), {PRE}liveness_verdict.json.
Refuse-if-exists.

    .venv/bin/python scratch/mathworld1_liveness.py           (Mac)
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402
import torch  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.mathgen.problems import make_integrate  # noqa: E402
from llmopt.search.derivation import (State,  # noqa: E402
                                      is_solved, successors)
from llmopt.train.mathnative import build_model  # noqa: E402
from scratch.mathworld1_birth import GCTok  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
PRE = "smoke_" if SMOKE else ""
CKPT = Path(f"checkpoints/{PRE}mathnative_19m_mw1_theta0.pt")
ROWS = Path(f"logs/mathworld1/{PRE}liveness.jsonl")
VERDICT = Path(f"logs/mathworld1/{PRE}liveness_verdict.json")
LEVELS = [4] if SMOKE else [4, 5, 6, 7]
SEEDS = range(9100, 9102) if SMOKE else range(9100, 9110)
MAX_DECISIONS = 12
WALL_CAP_S = 60.0
CTX = 4096
X = sp.Symbol("x")


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def main():
    for p in (ROWS, VERDICT):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    if not CKPT.exists():
        raise SystemExit(f"MISSING theta_0: {CKPT}")
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/mathworld1_liveness.py",
         "scratch/mathworld1_birth.py",
         "llmopt/search/derivation.py",
         "llmopt/mathgen/problems.py",
         "llmopt/train/mathnative.py"])
    # theta_0 identity is the DERIVED checkpoint sha256 below
    # (artifacts= expects HF snapshot dirs, not .pt files)
    ck_sha = hashlib.sha256(CKPT.read_bytes()).hexdigest()
    tok = GCTok()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_model(tok.vocab_size, ctx=CTX).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location=dev))
    model.eval()

    snapshot = {}   # state_hash -> (aset_hash, [(name, child)])

    def legal(st):
        h = sha(st.key())
        if h in snapshot:
            return h, snapshot[h]
        acts = sorted(successors(st),
                      key=lambda nc: (nc[0], nc[1].key()))
        ah = sha("\n".join(f"{n}|{c.key()}" for n, c in acts))
        snapshot[h] = (ah, acts)
        return h, snapshot[h]

    def score(prefix_ids, child_ids):
        ids = torch.tensor([prefix_ids + child_ids], device=dev)
        with torch.no_grad():
            logits = model(ids)
        lp = torch.log_softmax(logits[0].float(), -1)
        return sum(lp[len(prefix_ids) + i - 1, t].item()
                   for i, t in enumerate(child_ids))

    solves = {}
    n_overflow_eps = 0
    outcomes = {}
    t_all = time.monotonic()
    with ROWS.open("a") as f:
        for lv in LEVELS:
            solves[f"L{lv}"] = 0
            for seed in SEEDS:
                eid = f"L{lv}-s{seed}"
                prob = make_integrate(lv, seed)
                st = State(sp.Integral(prob._expr, X))
                t_ep = time.monotonic()
                outcome = "budget_exhausted"
                for step_id in range(MAX_DECISIONS):
                    if is_solved(st):
                        outcome = "solved"
                        break
                    if time.monotonic() - t_ep > WALL_CAP_S:
                        outcome = "wall_cap"
                        break
                    sh, (ah, acts) = legal(st)
                    if not acts:
                        outcome = "dead_end"
                        break
                    parent = str(st.expr)
                    pre_ids = tok.encode(
                        f"Current: {parent}\nHints: none\nStep: ")
                    cand = []
                    overflow = False
                    for name, c in acts:
                        cids = tok.encode(str(c.expr) + "\n")
                        if len(pre_ids) + len(cids) > CTX:
                            overflow = True
                            break
                        cand.append((name, c, cids))
                    if overflow:
                        outcome = "model_ctx_overflow"
                        f.write(json.dumps({
                            "episode_id": eid, "step_id": step_id,
                            "state_hash": sh,
                            "legal_set_hash": ah,
                            "n_legal": len(acts),
                            "event": "model_ctx_overflow"}) + "\n")
                        break
                    scored = [(score(pre_ids, cids), name, c)
                              for name, c, cids in cand]
                    scored.sort(
                        key=lambda t: (-t[0], t[1], t[2].key()))
                    s_best, name, child = scored[0]
                    f.write(json.dumps({
                        "episode_id": eid, "step_id": step_id,
                        "state_hash": sh, "legal_set_hash": ah,
                        "n_legal": len(acts),
                        "chosen": f"{name}#{sha(child.key())}",
                        "score": round(s_best, 4)}) + "\n")
                    st = child
                    if is_solved(st):
                        outcome = "solved"
                        break
                else:
                    outcome = ("solved" if is_solved(st)
                               else "budget_exhausted")
                if outcome == "solved":
                    solves[f"L{lv}"] += 1
                if outcome == "model_ctx_overflow":
                    n_overflow_eps += 1
                outcomes[eid] = outcome
                f.write(json.dumps({"episode_id": eid,
                                    "outcome": outcome}) + "\n")
                f.flush()
                print(f"[live] {eid}: {outcome}", flush=True)
        total = sum(solves.values())
        verdict = {
            "solved_by_level": solves, "solved_total": total,
            "episodes": len(outcomes),
            "bar_pass": total >= 1,
            "model_ctx_overflow_episodes": n_overflow_eps,
            "outcomes": outcomes,
            "snapshot_states": len(snapshot),
            "snapshot_hashes": {h: a for h, (a, _)
                                in snapshot.items()},
            "theta0_sha256": ck_sha,
            "wall_s": round(time.monotonic() - t_all, 1),
            "start": START,
            "completion_commit": completion_commit()}
        f.write(json.dumps({"meta": {
            "start": START,
            "completion_commit": completion_commit()}}) + "\n")
    VERDICT.write_text(json.dumps(verdict, indent=1))
    print(f"[live] solved {total}/{len(outcomes)} "
          f"{solves}; bar_pass={verdict['bar_pass']}", flush=True)
    return 0 if verdict["bar_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
