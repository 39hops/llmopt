"""MATH-CYBER-1 RETRO-LABELER-QUAL-0 — failure-triggered
retrospective-credit labeler, mechanism qualification on
outcome-spent data (PRE-REG MATH-CYBER-1-RETRO-LABELER-QUAL-0,
booked at 61fdbc96).

Operator (frozen): controller = TERMINAL-FIRST + theta_0 — at
each state, if any legal child is terminal-solved (world
is_solved, no model, no hce) take the minimal terminal under
(name, child.key()); otherwise theta_0 argmax (serial B=1,
tie-break (-score, name, child.key()), ctx 4096, overflow law —
scorer-invoked states only). Budget 12, charged wall 60 s
(fresh paid set per walk, cap checked before each decision,
one-decision overshoot). LABELER: on episode FAILURE only,
visit each scorer-invoked state in order, force the episode's
own stored theta_0 RANK-2 child (outcome-blind; rank 3+ never
consulted), continue under the same frozen TERMINAL-FIRST +
theta_0 policy for 12-(t+1) decisions; emit a preference label
ONLY when that continuation SOLVES. Censored / ctx-overflow /
world-noncomparable forks emit NO label, never a negative (the
row schema carries no negative-label field at all).

Qualification episodes (all outcome-spent): L7-s9303 (B1
binding to the recorded FROZEN trajectory + B2 target
reproduction v the FRONTIER rank-2 rescue identity), L4-s9401
(B3 zero-label), L6-s9403 (B4 overflow), L6-s9300 (B6
TERMINAL-FIRST solves at step 0), plus a censor-injection
re-walk of L4-s9401 (B5, synthetic 61 s charge injected into
its step-0 fork continuation, separate stage so B3 stays
clean). One-shot; receipts logs/mathworld1/retrolabel_qual.jsonl
+ retrolabel_qual_verdict.json, refuse-if-exists. Label counts
here are qualification artifacts on spent data, never yield
evidence.

    .venv/bin/python scratch/mathworld1_retrolabel.py   (Mac)
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402
import torch  # noqa: E402

import llmopt.search.derivation as derivation  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.mathgen.problems import make_integrate  # noqa: E402
from llmopt.search.derivation import (State,  # noqa: E402
                                      is_solved, successors)
from llmopt.train.mathnative import build_model  # noqa: E402
from scratch.mathworld1_birth import GCTok  # noqa: E402

CKPT = Path("checkpoints/mathnative_19m_mw1_theta0.pt")
ROWS = Path("logs/mathworld1/retrolabel_qual.jsonl")
VERDICT = Path("logs/mathworld1/retrolabel_qual_verdict.json")
MAX_DECISIONS = 12
WALL_CAP_S = 60.0
CTX = 4096
X = sp.Symbol("x")


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


class World:
    def __init__(self):
        self.cache = {}
        self.walls = {}

    def legal(self, st: State):
        k = st.key()
        if k not in self.cache:
            derivation._RULE_CACHE.clear()
            t0 = time.monotonic()
            acts = sorted(successors(st),
                          key=lambda nc: (nc[0], nc[1].key()))
            self.walls[k] = time.monotonic() - t0
            self.cache[k] = [(n, c.expr) for n, c in acts]
        acts = [(n, State(e, st.plies + 1, st.history + (n,)))
                for n, e in self.cache[k]]
        ah = sha("\n".join(f"{n}|{c.key()}" for n, c in acts))
        return acts, ah


class Scorer:
    def __init__(self, model, tok, dev):
        self.model, self.tok, self.dev = model, tok, dev

    def score(self, prefix_ids, child_ids):
        ids = torch.tensor([prefix_ids + child_ids],
                           device=self.dev)
        with torch.no_grad():
            logits = self.model(ids)
        lp = torch.log_softmax(logits[0].float(), -1)
        s = sum(lp[len(prefix_ids) + i - 1, t].item()
                for i, t in enumerate(child_ids))
        if not math.isfinite(s):
            print("[FAIL] nonfinite score", flush=True)
            sys.exit(4)
        return s

    def rank_candidates(self, st, acts):
        pre_ids = self.tok.encode(
            f"Current: {str(st.expr)}\nHints: none\nStep: ")
        cand = []
        for name, c in acts:
            cids = self.tok.encode(str(c.expr) + "\n")
            if len(pre_ids) + len(cids) > CTX:
                return None
            cand.append((name, c, cids))
        scored = [(self.score(pre_ids, cids), name, c)
                  for name, c, cids in cand]
        scored.sort(key=lambda t: (-t[0], t[1], t[2].key()))
        return scored


def terminal_first_walk(root_state, world, scorer, budget,
                        inject_delay_s=0.0):
    """One frozen TERMINAL-FIRST + theta_0 walk. steps keep the
    rank-2 State object in memory (receipts carry ids only)."""
    st = root_state
    charged = inject_delay_s
    paid = set()
    steps = []
    for d in range(budget):
        if is_solved(st):
            return {"outcome": "solved", "depth": d,
                    "steps": steps, "charged": round(charged, 3)}
        if charged > WALL_CAP_S:
            return {"outcome": "wall_cap", "depth": d,
                    "steps": steps, "charged": round(charged, 3)}
        acts, ah = world.legal(st)
        k = st.key()
        if k not in paid:
            charged += world.walls.get(k, 0.0)
            paid.add(k)
        t0 = time.monotonic()
        if not acts:
            return {"outcome": "dead_end", "depth": d,
                    "steps": steps, "charged": round(charged, 3)}
        terminals = [(n, c) for n, c in acts if is_solved(c)]
        if terminals:
            name, child = min(
                terminals, key=lambda nc: (nc[0], nc[1].key()))
            steps.append({
                "step": d, "state_hash": sha(k),
                "legal_set_hash": ah,
                "scorer_invoked": False,
                "chosen": f"{name}#{sha(child.key())}",
                "rank2_id": None, "rank2_state": None})
            charged += time.monotonic() - t0
            st = child
            continue
        scored = scorer.rank_candidates(st, acts)
        charged += time.monotonic() - t0
        if scored is None:
            steps.append({
                "step": d, "state_hash": sha(k),
                "legal_set_hash": ah,
                "scorer_invoked": False, "chosen": None,
                "rank2_id": None, "rank2_state": None,
                "event": "model_ctx_overflow"})
            return {"outcome": "model_ctx_overflow", "depth": d,
                    "steps": steps, "charged": round(charged, 3)}
        s0, n0, c0 = scored[0]
        has2 = len(scored) > 1
        steps.append({
            "step": d, "state_hash": sha(k),
            "legal_set_hash": ah, "scorer_invoked": True,
            "chosen": f"{n0}#{sha(c0.key())}",
            "chosen_score": round(s0, 4),
            "rank2_id": (f"{scored[1][1]}"
                         f"#{sha(scored[1][2].key())}"
                         if has2 else None),
            "rank2_score": (round(scored[1][0], 4)
                            if has2 else None),
            "rank2_state": scored[1][2] if has2 else None})
        st = c0
    if is_solved(st):
        return {"outcome": "solved", "depth": budget,
                "steps": steps, "charged": round(charged, 3)}
    return {"outcome": "budget_exhausted", "depth": budget,
            "steps": steps, "charged": round(charged, 3)}


def label_episode(eid, stage, root, world, scorer, sink,
                  inject_fork_step=None):
    walk = terminal_first_walk(State(root), world, scorer,
                               MAX_DECISIONS)
    for s in walk["steps"]:
        row = {k: v for k, v in s.items() if k != "rank2_state"}
        row.update({"row": "decision", "episode_id": eid,
                    "stage": stage})
        sink.write(json.dumps(row) + "\n")
    labels = []
    n_forks = n_censored = n_overflow = 0
    if walk["outcome"] != "solved":
        for s in walk["steps"]:
            if (not s["scorer_invoked"]
                    or s["rank2_state"] is None):
                continue
            t = s["step"]
            horizon = MAX_DECISIONS - (t + 1)
            inj = (61.0 if inject_fork_step == t else 0.0)
            if is_solved(s["rank2_state"]) and inj == 0.0:
                cont = {"outcome": "solved", "depth": 0,
                        "charged": 0.0}
            else:
                cont = terminal_first_walk(
                    s["rank2_state"], world, scorer, horizon,
                    inject_delay_s=inj)
                cont = {k: cont[k]
                        for k in ("outcome", "depth", "charged")}
            n_forks += 1
            censored = cont["outcome"] == "wall_cap"
            n_censored += censored
            n_overflow += cont["outcome"] == "model_ctx_overflow"
            is_label = cont["outcome"] == "solved"
            frow = {"row": "fork", "episode_id": eid,
                    "stage": stage, "step_id": t,
                    "state_hash": s["state_hash"],
                    "chosen": s["chosen"],
                    "forced_rank2": s["rank2_id"],
                    "margin": round(s["chosen_score"]
                                    - s["rank2_score"], 4),
                    "horizon": horizon,
                    "continuation": cont,
                    "censored": censored,
                    "label": is_label}
            sink.write(json.dumps(frow) + "\n")
            if is_label:
                labels.append(frow)
    ep = {"row": "episode", "episode_id": eid, "stage": stage,
          "outcome": walk["outcome"],
          "charged_wall_s": walk["charged"],
          "scorer_invoked_states": sum(
              1 for s in walk["steps"] if s["scorer_invoked"]),
          "forks": n_forks, "labels": len(labels),
          "censored_forks": n_censored,
          "overflow_forks": n_overflow}
    sink.write(json.dumps(ep) + "\n")
    sink.flush()
    print(f"[retro] {eid}/{stage}: {walk['outcome']} "
          f"forks={n_forks} labels={len(labels)} "
          f"censored={n_censored}", flush=True)
    return walk, labels, ep


def main():
    for p in (ROWS, VERDICT):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    if not CKPT.exists():
        raise SystemExit(f"MISSING theta_0: {CKPT}")
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/mathworld1_retrolabel.py",
         "scratch/mathworld1_birth.py",
         "llmopt/search/derivation.py",
         "llmopt/mathgen/problems.py",
         "llmopt/train/mathnative.py"])
    ck_sha = hashlib.sha256(CKPT.read_bytes()).hexdigest()
    tok = GCTok()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_model(tok.vocab_size, ctx=CTX).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location=dev))
    model.eval()
    scorer = Scorer(model, tok, dev)
    world = World()

    def root(lv, seed):
        return sp.Integral(make_integrate(lv, seed)._expr, X)

    # recorded FROZEN trajectory of L7-s9303 for B1 binding
    rec = [json.loads(l) for l in
           open("logs/mathworld1/active_pair.jsonl")
           if l.strip() and not l.startswith('{"meta"')]
    rec9303 = sorted(
        [r for r in rec if r.get("arm") == "FROZEN"
         and r.get("row") == "decision"
         and r.get("stage") == "adapt"
         and r.get("episode_id") == "L7-s9303"],
        key=lambda r: r["step_id"])
    # FRONTIER step-0 rank-2 rescue identity for B2
    fr = [json.loads(l) for l in
          open("logs/mathworld1/frontier_desk.jsonl")
          if '"row": "fork"' in l]
    b2_target = [f["forced"] for f in fr
                 if f["episode_id"] == "L7-s9303"
                 and f["step_id"] == 0
                 and f["theta0_rank"] == 2]
    assert len(b2_target) == 1
    with ROWS.open("a") as f:
        w9303, lab9303, _ = label_episode(
            "L7-s9303", "qual", root(7, 9303), world, scorer, f)
        _, lab9401, ep9401 = label_episode(
            "L4-s9401", "qual", root(4, 9401), world, scorer, f)
        _, lab9403, ep9403 = label_episode(
            "L6-s9403", "qual", root(6, 9403), world, scorer, f)
        w9300, _, _ = label_episode(
            "L6-s9300", "qual", root(6, 9300), world, scorer, f)
        _, lab_cens, ep_cens = label_episode(
            "L4-s9401", "censor_qual", root(4, 9401), world,
            scorer, f, inject_fork_step=0)
        rows_all = [json.loads(l) for l in ROWS.open()
                    if l.strip()]
        bars = {
            "B1_binding_and_fail":
                w9303["outcome"] == "budget_exhausted"
                and len(w9303["steps"]) == len(rec9303) == 12
                and all(s["state_hash"] == r["state_hash"]
                        and s["legal_set_hash"]
                        == r["legal_set_hash"]
                        for s, r in zip(w9303["steps"],
                                        rec9303)),
            "B2_target_reproduced":
                len(lab9303) >= 1 and any(
                    l["step_id"] == 0
                    and l["forced_rank2"] == b2_target[0]
                    for l in lab9303),
            "B3_zero_label":
                ep9401["outcome"] != "solved"
                and len(lab9401) == 0,
            "B4_overflow_no_negative":
                ep9403["outcome"] == "model_ctx_overflow"
                and len(lab9403) == 0
                and not any("negative" in r for r in rows_all),
            "B5_censor_no_label":
                any(r.get("row") == "fork"
                    and r["stage"] == "censor_qual"
                    and r["step_id"] == 0 and r["censored"]
                    and not r["label"] for r in rows_all),
            "B6_terminal_first_solves_9300":
                w9300["outcome"] == "solved"
                and w9300["steps"][0]["scorer_invoked"]
                is False}
        f.write(json.dumps({"meta": {
            "theta0_sha256": ck_sha, "device": dev,
            "b2_target_identity": b2_target[0],
            "start": START,
            "completion_commit": completion_commit()}}) + "\n")
    verdict = {
        "device": dev, "theta0_sha256": ck_sha,
        "bars": bars, "pass": all(bars.values()),
        "labels_L7_s9303": [
            {"step": l["step_id"], "forced": l["forced_rank2"],
             "margin": l["margin"],
             "depth": l["continuation"]["depth"]}
            for l in lab9303],
        "world_states": len(world.cache),
        "start": START,
        "completion_commit": completion_commit()}
    VERDICT.write_text(json.dumps(verdict, indent=1))
    print(f"[retro] pass={verdict['pass']} bars={bars}",
          flush=True)
    return 0 if verdict["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
