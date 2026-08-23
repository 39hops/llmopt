"""MATH-CYBER-1 TERMINAL-DOMINANCE-0 terminal-miss census
(PRE-REG MATH-CYBER-1-TERMINAL-DOMINANCE-0, commit a3759d06).

Population: every recorded theta_0-scored decision state —
CALIBRATION (logs/mathworld1/liveness.jsonl, 40 episodes) +
ADAPT/HOLDOUT FROZEN-arm rows (logs/mathworld1/active_pair.jsonl,
80 episodes). ACTIVE-arm rows are EXCLUDED (mid-stream weights
unrecoverable); the exclusion count is reported.

Per episode: binding replay (state_hash + legal_set_hash +
n_legal + recorded-action existence asserted; advance by the
RECORDED child; mismatch books the episode WORLD-NONCOMPARABLE
with its bound prefix counted). Per bound state: terminal
children via the world predicate is_solved (no model); at states
with >=1 terminal child, theta_0 re-scores the FULL legal set
(serial B=1, tie-break (-score, name, child.key()), ctx 4096,
overflow law) to record every terminal child's rank, whether the
recorded choice was terminal, and the chosen-minus-best-terminal
margin. A TERMINAL MISS = >=1 legal terminal child, recorded
choice non-terminal. Recorded model_ctx_overflow states census
terminal children but carry no ranks (unscorable, structural).
Zero training; outcome-spent data only.

SMOKE=1: spent CALIBRATION episodes L4-s9100 (solved-in-1;
exercises scoring + the solved-episode invariant that the last
recorded decision chose a terminal child) + L4-s9104 (12-decision
failure) + a corrupted-hash tamper case; receipts on smoke_
paths. Real mode needs MW1_TERMINAL_GO=1.

    SMOKE=1 .venv/bin/python scratch/mathworld1_terminal.py  (Mac)
"""
import hashlib
import json
import math
import os
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

SMOKE = os.environ.get("SMOKE") == "1"
if not SMOKE and os.environ.get("MW1_TERMINAL_GO") != "1":
    raise SystemExit("REFUSING: real mode needs MW1_TERMINAL_GO=1;"
                     " SMOKE=1 for qualification")
PRE = (f"smoke{os.environ.get('SMOKE_TAG', '')}_"
       if SMOKE else "")
CKPT = Path("checkpoints/mathnative_19m_mw1_theta0.pt")
ROWS = Path(f"logs/mathworld1/{PRE}terminal_census.jsonl")
VERDICT = Path(
    f"logs/mathworld1/{PRE}terminal_census_verdict.json")
CTX = 4096
X = sp.Symbol("x")


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


class World:
    def __init__(self):
        self.cache = {}

    def legal(self, st: State):
        k = st.key()
        if k not in self.cache:
            derivation._RULE_CACHE.clear()
            acts = sorted(successors(st),
                          key=lambda nc: (nc[0], nc[1].key()))
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


def load_recorded(source, eid, stage):
    if source == "liveness":
        rows = [json.loads(l) for l in
                open("logs/mathworld1/liveness.jsonl")
                if l.strip()]
        dec = [r for r in rows if r.get("episode_id") == eid
               and r.get("step_id") is not None]
        out = [r for r in rows if r.get("episode_id") == eid
               and "outcome" in r]
    else:
        rows = [json.loads(l) for l in
                open("logs/mathworld1/active_pair.jsonl")
                if l.strip() and not l.startswith('{"meta"')]
        dec = [r for r in rows if r.get("arm") == "FROZEN"
               and r.get("row") == "decision"
               and r.get("stage") == stage
               and r.get("episode_id") == eid]
        out = [r for r in rows if r.get("arm") == "FROZEN"
               and r.get("row") == "episode"
               and r.get("stage") == stage
               and r.get("episode_id") == eid]
    dec.sort(key=lambda r: r["step_id"])
    assert len(out) == 1, (eid, stage, len(out))
    return dec, out[0]["outcome"]


def census_episode(eid, band, root, rec, outcome, world, scorer,
                   sink, tamper_step=None):
    """Replay + terminal census. Returns episode summary."""
    st = State(root)
    n_census = n_term_states = n_miss = 0
    mismatch = None
    for r in rec:
        if tamper_step is not None and r["step_id"] == tamper_step:
            r = dict(r)
            r["state_hash"] = "0" * 16
        if sha(st.key()) != r["state_hash"]:
            mismatch = f"state_hash mismatch at step {r['step_id']}"
            break
        acts, ah = world.legal(st)
        if (ah != r["legal_set_hash"]
                or len(acts) != r["n_legal"]):
            mismatch = f"legal_set mismatch at step {r['step_id']}"
            break
        terminals = [(n, c) for n, c in acts if is_solved(c)]
        row = {"row": "state", "episode_id": eid, "band": band,
               "step_id": r["step_id"],
               "state_hash": r["state_hash"],
               "n_legal": len(acts),
               "n_terminal": len(terminals),
               "episode_outcome": outcome}
        if r.get("event") == "model_ctx_overflow":
            sc = scorer.rank_candidates(st, acts)
            if sc is not None:
                mismatch = (f"overflow did not reproduce at step "
                            f"{r['step_id']}")
                break
            row.update({"scorable": False, "chosen": None,
                        "chosen_terminal": None,
                        "terminal_ranks": None,
                        "best_terminal_rank": None,
                        "margin_chosen_minus_best_terminal":
                            None})
            n_census += 1
            n_term_states += bool(terminals)
            sink.write(json.dumps(row) + "\n")
            break                        # trajectory ends here
        name, chash = r["chosen"].rsplit("#", 1)
        match = [(n, c) for n, c in acts
                 if n == name and sha(c.key()) == chash]
        if len(match) != 1:
            mismatch = (f"recorded action absent at step "
                        f"{r['step_id']}")
            break
        chosen_terminal = is_solved(match[0][1])
        row.update({"scorable": True, "chosen": r["chosen"],
                    "chosen_terminal": chosen_terminal})
        if terminals:
            n_term_states += 1
            scored = scorer.rank_candidates(st, acts)
            assert scored is not None, (eid, r["step_id"])
            id2rank = {f"{n}#{sha(c.key())}": i + 1
                       for i, (s, n, c) in enumerate(scored)}
            id2score = {f"{n}#{sha(c.key())}": s
                        for s, n, c in scored}
            tids = [f"{n}#{sha(c.key())}" for n, c in terminals]
            tranks = sorted(id2rank[t] for t in tids)
            best_t = min(tids, key=lambda t: id2rank[t])
            row.update({
                "terminal_ranks": tranks,
                "best_terminal_rank": tranks[0],
                "margin_chosen_minus_best_terminal": round(
                    id2score[r["chosen"]] - id2score[best_t], 4),
                "rescored_argmax_matches_recorded":
                    min(id2rank, key=lambda k: id2rank[k])
                    == r["chosen"]})
            if not chosen_terminal:
                n_miss += 1
                row["terminal_miss"] = True
        else:
            row.update({"terminal_ranks": None,
                        "best_terminal_rank": None,
                        "margin_chosen_minus_best_terminal":
                            None})
        n_census += 1
        sink.write(json.dumps(row) + "\n")
        st = match[0][1]
    summary = {"row": "episode", "episode_id": eid, "band": band,
               "episode_outcome": outcome,
               "states_censused": n_census,
               "terminal_child_states": n_term_states,
               "terminal_misses": n_miss,
               "world_noncomparable": mismatch}
    sink.write(json.dumps(summary) + "\n")
    sink.flush()
    print(f"[terminal] {eid}: {n_census} states, "
          f"{n_term_states} w/terminal, {n_miss} miss"
          + (f" MISMATCH {mismatch}" if mismatch else ""),
          flush=True)
    return summary


def main():
    for p in (ROWS, VERDICT):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    if not CKPT.exists():
        raise SystemExit(f"MISSING theta_0: {CKPT}")
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/mathworld1_terminal.py",
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

    def ep(lv, seed):
        return (f"L{lv}-s{seed}",
                sp.Integral(make_integrate(lv, seed)._expr, X))

    if SMOKE:
        plan = [(4, 9100, "calibration", "liveness"),
                (4, 9104, "calibration", "liveness")]
    else:
        plan = ([(lv, s, "calibration", "liveness")
                 for s in range(9100, 9110) for lv in (4, 5, 6, 7)]
                + [(lv, s, "adapt", "adapt")
                   for s in range(9300, 9310)
                   for lv in (4, 5, 6, 7)]
                + [(lv, s, "holdout", "holdout")
                   for s in range(9400, 9410)
                   for lv in (4, 5, 6, 7)])
    # ACTIVE-arm exclusion count (disclosed, never censused)
    apr = [json.loads(l) for l in
           open("logs/mathworld1/active_pair.jsonl")
           if l.strip() and not l.startswith('{"meta"')]
    active_excluded = sum(
        1 for r in apr if r.get("arm") == "ACTIVE"
        and r.get("row") == "decision"
        and r.get("stage") in ("adapt", "holdout"))
    summaries = []
    t0 = time.monotonic()
    with ROWS.open("a") as f:
        for lv, seed, band, source in plan:
            eid, root = ep(lv, seed)
            rec, outcome = load_recorded(source, eid,
                                         band if source
                                         != "liveness" else None)
            summaries.append(census_episode(
                eid, band, root, rec, outcome, world, scorer, f))
        checks = {}
        if SMOKE:
            eid, root = ep(4, 9104)
            rec, outcome = load_recorded("liveness", eid, None)
            tam = census_episode(f"{eid}-TAMPER", "calibration",
                                 root, rec, outcome, world,
                                 scorer, f, tamper_step=3)
            srows = [json.loads(l) for l in ROWS.open()
                     if '"row": "state"' in l]
            s9100 = [r for r in srows
                     if r["episode_id"] == "L4-s9100"]
            checks = {
                "replay_ok_both": all(
                    s["world_noncomparable"] is None
                    for s in summaries),
                "tamper_detected":
                    tam["world_noncomparable"] is not None,
                "solved_last_decision_chose_terminal":
                    s9100[-1]["chosen_terminal"] is True,
                "scoring_exercised": any(
                    r.get("best_terminal_rank") is not None
                    for r in srows),
                "rescored_argmax_reported": any(
                    "rescored_argmax_matches_recorded" in r
                    for r in srows),
                "fields_complete": all(
                    set(("n_terminal", "chosen_terminal",
                         "episode_outcome")) <= set(r)
                    for r in srows)}
        f.write(json.dumps({"meta": {
            "theta0_sha256": ck_sha, "device": dev,
            "smoke": SMOKE,
            "active_rows_excluded": active_excluded,
            "wall_s": round(time.monotonic() - t0, 1),
            "start": START,
            "completion_commit": completion_commit()}}) + "\n")
    tot = lambda k: sum(s[k] for s in summaries)  # noqa: E731
    misses = [s for s in summaries if s["terminal_misses"]]
    verdict = {
        "smoke": SMOKE, "device": dev, "theta0_sha256": ck_sha,
        "episodes": len(summaries),
        "world_noncomparable": [
            s["episode_id"] for s in summaries
            if s["world_noncomparable"]],
        "states_censused": tot("states_censused"),
        "terminal_child_states": tot("terminal_child_states"),
        "terminal_misses": tot("terminal_misses"),
        "miss_episodes": {s["episode_id"]: {
            "band": s["band"], "misses": s["terminal_misses"],
            "outcome": s["episode_outcome"]} for s in misses},
        "active_rows_excluded": active_excluded,
        "wall_s": round(time.monotonic() - t0, 1),
        "mechanism_checks": checks,
        "pass": (all(checks.values()) if SMOKE else None),
        "start": START,
        "completion_commit": completion_commit()}
    VERDICT.write_text(json.dumps(verdict, indent=1))
    print(f"[terminal] {'smoke ' if SMOKE else ''}states="
          f"{verdict['states_censused']} wterm="
          f"{verdict['terminal_child_states']} miss="
          f"{verdict['terminal_misses']}"
          + (f" pass={verdict['pass']}" if SMOKE else ""),
          flush=True)
    if SMOKE:
        return 0 if verdict["pass"] else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
