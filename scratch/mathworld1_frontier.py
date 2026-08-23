"""MATH-CYBER-1 FRONTIER-DESK-0 one-deviation repairability census
(PRE-REG MATH-CYBER-1-FRONTIER-DESK-0, commit 4b2ae5af).

For each of the six FROZEN theta_0 failed roots (ADAPT L6-s9300,
L7-s9303; HOLDOUT L4-s9400, L4-s9401, L6-s9403, L4-s9405):
reconstruct the recorded trajectory from its receipt
(logs/mathworld1/active_pair.jsonl, arm FROZEN) with binding
asserts (state_hash, legal_set_hash + n_legal, recorded
name#child_hash exists), advancing via the RECORDED child; then
fork every legal action except the recorded choice at every bound
decision state and continue with immutable theta_0 (serial B=1
teacher-forced scorer, tie-break (-score, name, child.key()),
ctx 4096, overflow law) for exactly 12-(t+1) further decisions.
Any world mismatch books that root WORLD-NONCOMPARABLE/UNDECIDED.

Continuations are DETERMINISTIC (frozen policy + one realized
desk-world snapshot), so each forced child's continuation is
computed once at horizon 11 and re-read as a truncation at
shorter horizons (memo cache of the same computation). Re-scored
candidate ranks/margins v the recorded rows are REPORTED
metadata, never reconstruction gates (mps forwards are run-level
nondeterministic across processes).

Classification: forks from scored decision states are
MODEL-ACTIONABLE; forced rescues from a recorded
model_ctx_overflow state are STRUCTURAL-ONLY/INTERFACE-BLOCKED.
Wall: per-continuation charged accounting (fresh paid set,
first-materialization charge + own scoring wall, 60 s cap checked
before each decision); wall-censored forks report CENSORED, never
"not repairable".

SMOKE=1: spent CALIBRATION liveness failures L4-s9104
(budget_exhausted, reconstruction + fork + memo-truncation) +
L6-s9100 (model_ctx_overflow, structural-fork path) from
logs/mathworld1/liveness.jsonl, forks capped at 2 per site
(mechanism coverage only; real mode is exhaustive), plus a
deliberate corrupted-hash reconstruction proving the mismatch
branch. Receipts on smoke_ paths. Real mode needs
MW1_FRONTIER_GO=1.

    SMOKE=1 .venv/bin/python scratch/mathworld1_frontier.py   (Mac)
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
if not SMOKE and os.environ.get("MW1_FRONTIER_GO") != "1":
    raise SystemExit("REFUSING: real mode needs MW1_FRONTIER_GO=1;"
                     " SMOKE=1 for qualification")
PRE = (f"smoke{os.environ.get('SMOKE_TAG', '')}_"
       if SMOKE else "")
CKPT = Path("checkpoints/mathnative_19m_mw1_theta0.pt")
ROWS = Path(f"logs/mathworld1/{PRE}frontier_desk.jsonl")
VERDICT = Path(f"logs/mathworld1/{PRE}frontier_desk_verdict.json")
MAX_DECISIONS = 12
WALL_CAP_S = 60.0
CTX = 4096
X = sp.Symbol("x")

REAL_ROOTS = [("L6-s9300", "adapt"), ("L7-s9303", "adapt"),
              ("L4-s9400", "holdout"), ("L4-s9401", "holdout"),
              ("L6-s9403", "holdout"), ("L4-s9405", "holdout")]
SMOKE_ROOTS = [("L4-s9104", "liveness"), ("L6-s9100", "liveness")]


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


class World:
    """Complete legal-set cache, FULL-key-keyed, semantic values;
    records each state's first-materialization wall."""

    def __init__(self):
        self.cache = {}
        self.walls = {}

    def materialize(self, keys_states):
        for k in sorted(set(keys_states) - set(self.cache)):
            derivation._RULE_CACHE.clear()
            st = keys_states[k]
            t0 = time.monotonic()
            acts = sorted(successors(st),
                          key=lambda nc: (nc[0], nc[1].key()))
            self.walls[k] = time.monotonic() - t0
            self.cache[k] = [(n, c.expr) for n, c in acts]

    def legal(self, st: State):
        acts = [(n, State(e, st.plies + 1, st.history + (n,)))
                for n, e in self.cache[st.key()]]
        ah = sha("\n".join(f"{n}|{c.key()}" for n, c in acts))
        return acts, ah


class Scorer:
    def __init__(self, model, tok, dev):
        self.model = model
        self.tok = tok
        self.dev = dev

    def score(self, prefix_ids, child_ids):
        ids = torch.tensor([prefix_ids + child_ids],
                           device=self.dev)
        with torch.no_grad():
            logits = self.model(ids)
        lp = torch.log_softmax(logits[0].float(), -1)
        s = sum(lp[len(prefix_ids) + i - 1, t].item()
                for i, t in enumerate(child_ids))
        if not math.isfinite(s):
            print("[FAIL] nonfinite candidate score", flush=True)
            sys.exit(4)
        return s

    def rank_candidates(self, st, acts):
        """Score every legal candidate; return sorted
        [(score, name, child, cids)] under the standing tie-break,
        or None on ctx overflow (any candidate beyond CTX)."""
        pre_ids = self.tok.encode(
            f"Current: {str(st.expr)}\nHints: none\nStep: ")
        cand = []
        for name, c in acts:
            cids = self.tok.encode(str(c.expr) + "\n")
            if len(pre_ids) + len(cids) > CTX:
                return None
            cand.append((name, c, cids))
        scored = [(self.score(pre_ids, cids), name, c, cids)
                  for name, c, cids in cand]
        scored.sort(key=lambda t: (-t[0], t[1], t[2].key()))
        return scored


def run_continuation(start_state, world, scorer):
    """Frozen theta_0 greedy walk from start_state at the MAXIMUM
    horizon (11), per-continuation charged wall. Returns
    {terminal, depth, charged_at_event, steps}: terminal in
    solved/dead_end/model_ctx_overflow/wall_cap/budget_exhausted;
    depth = continuation decisions consumed when terminal fired
    (solved-on-entry is the caller's depth-0 case)."""
    charged = 0.0
    paid = set()
    st = start_state
    steps = []
    for d in range(MAX_DECISIONS - 1):
        if is_solved(st):
            return {"terminal": "solved", "depth": d,
                    "charged": round(charged, 3), "steps": steps}
        if charged > WALL_CAP_S:
            return {"terminal": "wall_cap", "depth": d,
                    "charged": round(charged, 3), "steps": steps}
        k = st.key()
        world.materialize({k: st})
        if k not in paid:
            charged += world.walls.get(k, 0.0)
            paid.add(k)
        t0 = time.monotonic()
        acts, ah = world.legal(st)
        if not acts:
            charged += time.monotonic() - t0
            return {"terminal": "dead_end", "depth": d,
                    "charged": round(charged, 3), "steps": steps}
        scored = scorer.rank_candidates(st, acts)
        if scored is None:
            charged += time.monotonic() - t0
            return {"terminal": "model_ctx_overflow", "depth": d,
                    "charged": round(charged, 3), "steps": steps}
        _, name, child, _ = scored[0]
        charged += time.monotonic() - t0
        steps.append(f"{name}#{sha(child.key())}")
        st = child
        if is_solved(st):
            return {"terminal": "solved", "depth": d + 1,
                    "charged": round(charged, 3), "steps": steps}
    return {"terminal": "budget_exhausted",
            "depth": MAX_DECISIONS - 1,
            "charged": round(charged, 3), "steps": steps}


def truncate(trace, horizon):
    """Outcome of the memoized max-horizon continuation when only
    `horizon` decisions remain (deterministic truncation)."""
    if trace["terminal"] == "solved" and trace["depth"] <= horizon:
        return "solved", trace["depth"]
    if (trace["terminal"] in ("dead_end", "model_ctx_overflow",
                              "wall_cap")
            and trace["depth"] <= horizon):
        return trace["terminal"], trace["depth"]
    return "budget_exhausted", horizon


def load_recorded(eid, stage):
    """Recorded FROZEN decision rows for one root, step order."""
    if stage == "liveness":
        src = Path("logs/mathworld1/liveness.jsonl")
        rows = [json.loads(l) for l in src.open() if l.strip()]
        rec = [r for r in rows if r.get("episode_id") == eid
               and r.get("step_id") is not None]
    else:
        src = Path("logs/mathworld1/active_pair.jsonl")
        rows = [json.loads(l) for l in src.open()
                if l.strip() and not l.startswith('{"meta"')]
        rec = [r for r in rows if r.get("arm") == "FROZEN"
               and r.get("row") == "decision"
               and r.get("stage") == stage
               and r.get("episode_id") == eid]
    rec.sort(key=lambda r: r["step_id"])
    return rec


def reconstruct(eid, root, rec, world, scorer):
    """Binding replay of the recorded trajectory. Returns
    (sites, None) or (partial_sites, mismatch_reason)."""
    st = State(root)
    sites = []
    for r in rec:
        k = st.key()
        if sha(k) != r["state_hash"]:
            return sites, (f"state_hash mismatch at step "
                           f"{r['step_id']}: {sha(k)} v "
                           f"{r['state_hash']}")
        world.materialize({k: st})
        acts, ah = world.legal(st)
        if ah != r["legal_set_hash"] or len(acts) != r["n_legal"]:
            return sites, (f"legal_set mismatch at step "
                           f"{r['step_id']}: {ah}/{len(acts)} v "
                           f"{r['legal_set_hash']}/{r['n_legal']}")
        if r.get("event") == "model_ctx_overflow":
            scored = scorer.rank_candidates(st, acts)
            if scored is not None:
                return sites, (f"overflow did not reproduce at "
                               f"step {r['step_id']}")
            sites.append({"step": r["step_id"], "state": st,
                          "acts": acts, "ah": ah,
                          "kind": "structural", "scored": None,
                          "rec_chosen": None, "rec_score": None})
            return sites, None      # trajectory ends here
        name, chash = r["chosen"].rsplit("#", 1)
        match = [(n, c) for n, c in acts
                 if n == name and sha(c.key()) == chash]
        if len(match) != 1:
            return sites, (f"recorded action absent at step "
                           f"{r['step_id']}: {r['chosen'][:60]}")
        scored = scorer.rank_candidates(st, acts)
        if scored is None:
            return sites, (f"unexpected overflow re-scoring step "
                           f"{r['step_id']}")
        sites.append({"step": r["step_id"], "state": st,
                      "acts": acts, "ah": ah, "kind": "scored",
                      "scored": scored,
                      "rec_chosen": r["chosen"],
                      "rec_score": r.get("score")})
        st = match[0][1]
    return sites, None


def fork_root(eid, root, rec, world, scorer, sink,
              max_forks_per_site=None):
    sites, mismatch = reconstruct(eid, root, rec, world, scorer)
    if mismatch is not None:
        sink.write(json.dumps({
            "row": "root", "episode_id": eid,
            "classification": "UNDECIDED",
            "reason": f"WORLD-NONCOMPARABLE: {mismatch}",
            "bound_sites": len(sites)}) + "\n")
        sink.flush()
        return {"episode_id": eid, "classification": "UNDECIDED",
                "reason": mismatch}
    memo = {}
    forks = []
    argmax_agree = 0
    n_scored_sites = 0
    for site in sites:
        t = site["step"]
        horizon = MAX_DECISIONS - (t + 1)
        if site["kind"] == "scored":
            n_scored_sites += 1
            id2rank = {f"{n}#{sha(c.key())}": i + 1
                       for i, (s, n, c, _) in
                       enumerate(site["scored"])}
            top1 = (f"{site['scored'][0][1]}"
                    f"#{sha(site['scored'][0][2].key())}")
            if top1 == site["rec_chosen"]:
                argmax_agree += 1
            chosen_rank = id2rank[site["rec_chosen"]]
            chosen_score = next(
                s for s, n, c, _ in site["scored"]
                if f"{n}#{sha(c.key())}" == site["rec_chosen"])
            alts = [(s, n, c, cids) for s, n, c, cids
                    in site["scored"]
                    if f"{n}#{sha(c.key())}" != site["rec_chosen"]]
        else:
            chosen_rank = None
            chosen_score = None
            alts = [(None, n, c,
                     scorer.tok.encode(str(c.expr) + "\n"))
                    for n, c in site["acts"]]
        if max_forks_per_site is not None:
            alts = alts[:max_forks_per_site]
        for s_alt, name, child, cids in alts:
            fid = f"{name}#{sha(child.key())}"
            if is_solved(child):
                outcome, depth, charged, memo_hit = (
                    "solved", 0, 0.0, False)
            else:
                ck = child.key()
                memo_hit = ck in memo
                if not memo_hit:
                    memo[ck] = run_continuation(child, world,
                                                scorer)
                tr = memo[ck]
                outcome, depth = truncate(tr, horizon)
                charged = tr["charged"]
            fork = {
                "row": "fork", "episode_id": eid, "step_id": t,
                "state_hash": sha(site["state"].key()),
                "legal_set_hash": site["ah"],
                "forced": fid,
                "fork_class": ("MODEL-ACTIONABLE"
                               if site["kind"] == "scored"
                               else "STRUCTURAL-ONLY"),
                "theta0_score": (round(s_alt, 4)
                                 if s_alt is not None else None),
                "theta0_rank": (id2rank[fid]
                                if site["kind"] == "scored"
                                else None),
                "chosen_rank": chosen_rank,
                "margin_chosen_minus_forced": (
                    round(chosen_score - s_alt, 4)
                    if s_alt is not None else None),
                "cand_tokens": (len(cids) if cids is not None
                                else None),
                "horizon": horizon,
                "outcome": outcome,
                "continuation_depth": depth,
                "continuation_charged_s": charged,
                "censored": outcome == "wall_cap",
                "memo_hit": memo_hit}
            sink.write(json.dumps(fork) + "\n")
            forks.append(fork)
        sink.flush()
        print(f"[frontier] {eid} step {t}: {len(alts)} forks "
              f"done", flush=True)
    act = [f for f in forks
           if f["fork_class"] == "MODEL-ACTIONABLE"]
    struct = [f for f in forks
              if f["fork_class"] == "STRUCTURAL-ONLY"]
    act_rescues = [f for f in act if f["outcome"] == "solved"]
    struct_rescues = [f for f in struct
                      if f["outcome"] == "solved"]
    censored = [f for f in forks if f["censored"]]
    if act_rescues:
        cls = "ONE-DEV-REPAIRABLE"
    elif censored or max_forks_per_site is not None:
        cls = "UNDECIDED"
    else:
        cls = "ONE-DEV-NOT-REPAIRABLE"
    summary = {
        "row": "root", "episode_id": eid,
        "classification": cls,
        "bound_sites": len(sites),
        "scored_sites": n_scored_sites,
        "structural_sites": len(sites) - n_scored_sites,
        "alternatives_exhausted": len(forks),
        "actionable_forks": len(act),
        "structural_forks": len(struct),
        "actionable_rescues": len(act_rescues),
        "structural_rescues": len(struct_rescues),
        "rescue_fraction_actionable": (
            round(len(act_rescues) / len(act), 4) if act else None),
        "earliest_rescue_step": (
            min(f["step_id"] for f in act_rescues)
            if act_rescues else None),
        "min_rescue_rank": (
            min(f["theta0_rank"] for f in act_rescues)
            if act_rescues else None),
        "rank2_rescue_exists": any(
            f["theta0_rank"] == 2 for f in act_rescues),
        "censored_forks": len(censored),
        "argmax_agreement": f"{argmax_agree}/{n_scored_sites}",
        "smoke_fork_cap": max_forks_per_site}
    sink.write(json.dumps(summary) + "\n")
    sink.flush()
    print(f"[frontier] {eid}: {cls} rescues="
          f"{len(act_rescues)}+{len(struct_rescues)}s "
          f"censored={len(censored)}", flush=True)
    return summary


def main():
    for p in (ROWS, VERDICT):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    if not CKPT.exists():
        raise SystemExit(f"MISSING theta_0: {CKPT}")
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/mathworld1_frontier.py",
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
    roots = SMOKE_ROOTS if SMOKE else REAL_ROOTS
    cap = 2 if SMOKE else None
    summaries = {}
    with ROWS.open("a") as f:
        for eid, stage in roots:
            lv, seed = int(eid[1]), int(eid.split("-s")[1])
            root = sp.Integral(make_integrate(lv, seed)._expr, X)
            rec = load_recorded(eid, stage)
            assert rec, f"no recorded rows for {eid}"
            summaries[eid] = fork_root(eid, root, rec, world,
                                       scorer, f,
                                       max_forks_per_site=cap)
        checks = {}
        if SMOKE:
            # mismatch branch: corrupted copy of L4-s9104 rows
            rec = load_recorded("L4-s9104", "liveness")
            bad = [dict(r) for r in rec]
            bad[3]["state_hash"] = "0" * 16
            tam = fork_root("L4-s9104-TAMPER",
                            sp.Integral(make_integrate(
                                4, 9104)._expr, X),
                            bad, world, scorer, f,
                            max_forks_per_site=cap)
            frows = [json.loads(l) for l in ROWS.open()
                     if '"row": "fork"' in l]
            checks = {
                "recon_ok_both": all(
                    summaries[e]["classification"] != "UNDECIDED"
                    or summaries[e].get("reason") is None
                    for e in ("L4-s9104", "L6-s9100")),
                "tamper_detected":
                    tam["classification"] == "UNDECIDED"
                    and "state_hash mismatch"
                    in tam.get("reason", ""),
                "structural_site_found":
                    summaries["L6-s9100"]["structural_sites"] >= 1
                    and summaries["L6-s9100"]
                    ["structural_forks"] >= 1,
                "memo_hit_seen": any(r["memo_hit"]
                                     for r in frows),
                "truncation_horizons_vary": len(
                    {r["horizon"] for r in frows}) > 1,
                "fields_complete": all(
                    set(("forced", "theta0_rank", "horizon",
                         "outcome", "continuation_depth",
                         "censored")) <= set(r) for r in frows),
                "actionable_ranked": all(
                    r["theta0_rank"] is not None
                    for r in frows
                    if r["fork_class"] == "MODEL-ACTIONABLE"),
                "structural_unranked": all(
                    r["theta0_rank"] is None for r in frows
                    if r["fork_class"] == "STRUCTURAL-ONLY")}
        f.write(json.dumps({"meta": {
            "theta0_sha256": ck_sha, "device": dev,
            "smoke": SMOKE, "start": START,
            "completion_commit": completion_commit()}}) + "\n")
    verdict = {
        "smoke": SMOKE, "device": dev,
        "theta0_sha256": ck_sha,
        "world_states_materialized": len(world.cache),
        "roots": summaries,
        "mechanism_checks": checks,
        "pass": (all(checks.values()) if SMOKE else None),
        "start": START,
        "completion_commit": completion_commit()}
    VERDICT.write_text(json.dumps(verdict, indent=1))
    if SMOKE:
        print(f"[frontier] smoke pass={verdict['pass']} "
              f"checks={checks}", flush=True)
        return 0 if verdict["pass"] else 3
    print("[frontier] real census complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
