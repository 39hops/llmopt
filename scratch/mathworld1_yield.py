"""MATH-CYBER-1 LABEL-YIELD-0 — sacrificial fresh-band census of
retrospective-credit label supply (PRE-REG
MATH-CYBER-1-LABEL-YIELD-0 as AMENDED by -DEDUP, 020a0d25).

Band: L4-L7 x seeds 9500-9519, 80 episodes, seed-major
interleaved, SACRIFICIAL (permanently outcome-spent after this
census; future treatment bands >= 9600). Operator: the
RETRO-LABELER-QUAL-0-qualified frozen TERMINAL-FIRST + theta_0
controller + failure-triggered rank-2 labeler (top-2 only,
outcome-blind, no negatives, censored/overflow/noncomparable =
no label, hce nowhere). Zero training.

GO/NO-GO (amended -DEDUP): GO iff BOTH (1) >=3 DISTINCT failed
episodes each with >=1 usable outcome-differing rank-2 label AND
(2) >=5 DISTINCT corrective preference facts, globally deduped
by exact (state_hash, chosen, forced_rank2) — repeated visits
count once. Raw emitted-label rows = secondary workload color.
Diagnostic horizon_sensitive_fact: a deduped fact whose label
status differs across remaining horizons (reported, never split
into multiple facts). Contamination law: exact-match audit of
the 80 roots v birth diet + all prior bands + within-band
duplicates, disclosed, fixed denominator, never reseeded.

SMOKE=1 (spent data only, deterministic bars from the qual
receipts): L7-s9303 must yield 11 raw label rows collapsing to
3 distinct corrective facts with exactly 1 horizon-sensitive
fact; L6-s9300 solves (no forks); the contamination auditor
must reproduce the 11 known flagged ADAPT/HOLDOUT roots.
Receipts on smoke_ paths. Real mode needs MW1_YIELD_GO=1
(granted: Artin GO 2026-08-23 after -DEDUP amendment green).

    SMOKE=1 .venv/bin/python scratch/mathworld1_yield.py   (Mac)
"""
import glob
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
if not SMOKE and os.environ.get("MW1_YIELD_GO") != "1":
    raise SystemExit("REFUSING: real mode needs MW1_YIELD_GO=1;"
                     " SMOKE=1 for qualification")
PRE = (f"smoke{os.environ.get('SMOKE_TAG', '')}_"
       if SMOKE else "")
CKPT = Path("checkpoints/mathnative_19m_mw1_theta0.pt")
ROWS = Path(f"logs/mathworld1/{PRE}yield_census.jsonl")
VERDICT = Path(f"logs/mathworld1/{PRE}yield_census_verdict.json")
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


def terminal_first_walk(root_state, world, scorer, budget):
    st = root_state
    charged = 0.0
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
                "legal_set_hash": ah, "scorer_invoked": False,
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
                "legal_set_hash": ah, "scorer_invoked": False,
                "chosen": None, "rank2_id": None,
                "rank2_state": None,
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


def label_episode(eid, root, world, scorer, sink):
    """Walk + labeler; per-fork rows carry the dedup fact key."""
    walk = terminal_first_walk(State(root), world, scorer,
                               MAX_DECISIONS)
    for s in walk["steps"]:
        row = {k: v for k, v in s.items() if k != "rank2_state"}
        row.update({"row": "decision", "episode_id": eid})
        sink.write(json.dumps(row) + "\n")
    n_forks = n_labels = n_censored = n_overflow = 0
    facts = {}   # fact_key -> {"labeled": set(horizons),
    #                           "unlabeled": set(horizons)}
    if walk["outcome"] != "solved":
        for s in walk["steps"]:
            if (not s["scorer_invoked"]
                    or s["rank2_state"] is None):
                continue
            t = s["step"]
            horizon = MAX_DECISIONS - (t + 1)
            if is_solved(s["rank2_state"]):
                cont = {"outcome": "solved", "depth": 0,
                        "charged": 0.0}
            else:
                c = terminal_first_walk(
                    s["rank2_state"], world, scorer, horizon)
                cont = {k: c[k]
                        for k in ("outcome", "depth", "charged")}
            n_forks += 1
            censored = cont["outcome"] == "wall_cap"
            n_censored += censored
            n_overflow += cont["outcome"] == "model_ctx_overflow"
            is_label = cont["outcome"] == "solved"
            n_labels += is_label
            # -DEDUP-B identity pin: 16-hex child-hash suffixes,
            # never the printed rule@target string
            fact = (s["state_hash"],
                    s["chosen"].rsplit("#", 1)[1],
                    s["rank2_id"].rsplit("#", 1)[1])
            fk = "|".join(fact)
            facts.setdefault(fk, {"labeled": [],
                                  "unlabeled": []})
            facts[fk]["labeled" if is_label
                      else "unlabeled"].append(horizon)
            sink.write(json.dumps({
                "row": "fork", "episode_id": eid, "step_id": t,
                "state_hash": s["state_hash"],
                "chosen": s["chosen"],
                "forced_rank2": s["rank2_id"],
                "fact_key_sha": sha(fk),
                "margin": round(s["chosen_score"]
                                - s["rank2_score"], 4),
                "horizon": horizon, "continuation": cont,
                "censored": censored,
                "label": is_label}) + "\n")
    labeled_facts = {k: v for k, v in facts.items()
                     if v["labeled"]}
    hsens = {k: v for k, v in facts.items()
             if v["labeled"] and v["unlabeled"]}
    ep = {"row": "episode", "episode_id": eid,
          "outcome": walk["outcome"],
          "charged_wall_s": walk["charged"],
          "scorer_invoked_states": sum(
              1 for s in walk["steps"] if s["scorer_invoked"]),
          "forks": n_forks, "raw_label_rows": n_labels,
          "distinct_facts_encountered": len(facts),
          "distinct_corrective_facts": len(labeled_facts),
          "horizon_sensitive_facts": [
              {"fact_key_sha": sha(k),
               "labeled_horizons": sorted(v["labeled"]),
               "unlabeled_horizons": sorted(v["unlabeled"])}
              for k, v in sorted(hsens.items())],
          "censored_forks": n_censored,
          "overflow_forks": n_overflow}
    sink.write(json.dumps(ep) + "\n")
    sink.flush()
    print(f"[yield] {eid}: {walk['outcome']} forks={n_forks} "
          f"rawlabels={n_labels} facts={len(labeled_facts)} "
          f"hsens={len(hsens)}", flush=True)
    return ep, labeled_facts


def contamination_audit(episodes, exclude_bands=()):
    """Exact-match of root strings v birth diet + prior bands +
    within-band duplicates. Frozen law: disclose, never fix.
    exclude_bands: SMOKE-only — when the audited population IS a
    prior band, that band is excluded from the prior dict so the
    check measures cross-band hits, not self-identity. Real mode
    passes nothing here."""
    diet = set()
    for fp in (sorted(glob.glob(
            "data/micromodel_chains_shard*.jsonl"))
            + ["data/step_chains.jsonl"]):
        for line in open(fp):
            r = json.loads(line)
            diet.add(r["cur"])
            diet.add(r["nxt"])
    prior = {}
    for band, lo, hi in (("calibration", 9100, 9110),
                         ("train", 9200, 9250),
                         ("adapt", 9300, 9310),
                         ("holdout", 9400, 9410)):
        if band in exclude_bands:
            continue
        for lv in (4, 5, 6, 7):
            for s in range(lo, hi):
                prior.setdefault(str(sp.Integral(
                    make_integrate(lv, s)._expr, X)),
                    f"{band}:L{lv}-s{s}")
    flagged = {}
    seen = {}
    for eid, root in episodes:
        rs = str(root)
        hits = []
        if rs in diet:
            hits.append("birth_diet")
        if rs in prior:
            hits.append(prior[rs])
        if rs in seen:
            hits.append(f"duplicate_of:{seen[rs]}")
        seen.setdefault(rs, eid)
        if hits:
            flagged[eid] = hits
    return flagged


def main():
    for p in (ROWS, VERDICT):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    if not CKPT.exists():
        raise SystemExit(f"MISSING theta_0: {CKPT}")
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/mathworld1_yield.py",
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
        episodes = [ep(7, 9303), ep(6, 9300)]
    else:
        episodes = [ep(lv, s) for s in range(9500, 9520)
                    for lv in (4, 5, 6, 7)]
    audit_pop = (episodes if not SMOKE
                 else [ep(lv, s) for s in range(9300, 9310)
                       for lv in (4, 5, 6, 7)]
                 + [ep(lv, s) for s in range(9400, 9410)
                    for lv in (4, 5, 6, 7)])
    contamination = contamination_audit(
        audit_pop,
        exclude_bands=(("adapt", "holdout") if SMOKE else ()))
    print(f"[yield] contamination: {len(contamination)} flagged",
          flush=True)
    ep_rows = []
    global_facts = {}
    with ROWS.open("a") as f:
        f.write(json.dumps({"row": "contamination",
                            "denominator": len(audit_pop),
                            "flagged": contamination}) + "\n")
        for eid, root in episodes:
            row, lf = label_episode(eid, root, world, scorer, f)
            ep_rows.append(row)
            for k in lf:
                global_facts.setdefault(k, []).append(eid)
        f.write(json.dumps({"meta": {
            "theta0_sha256": ck_sha, "device": dev,
            "smoke": SMOKE, "start": START,
            "completion_commit": completion_commit()}}) + "\n")
    fails = [r for r in ep_rows if r["outcome"] != "solved"]
    label_eps = [r for r in ep_rows
                 if r["distinct_corrective_facts"] >= 1]
    go_a = len(label_eps) >= 3
    go_b = len(global_facts) >= 5
    checks = {}
    if SMOKE:
        r9303 = [r for r in ep_rows
                 if r["episode_id"] == "L7-s9303"][0]
        checks = {
            "dedup_3_facts":
                r9303["distinct_corrective_facts"] == 3,
            "raw_rows_11": r9303["raw_label_rows"] == 11,
            "one_horizon_sensitive":
                len(r9303["horizon_sensitive_facts"]) == 1,
            "solved_episode_no_forks": [
                r for r in ep_rows
                if r["episode_id"] == "L6-s9300"][0]["forks"]
                == 0,
            "contamination_reproduces_11":
                len(contamination) == 11}
    verdict = {
        "smoke": SMOKE, "device": dev, "theta0_sha256": ck_sha,
        "episodes": len(ep_rows),
        "contamination": {"denominator": len(audit_pop),
                          "flagged": contamination},
        "baseline": {
            "solves": len(ep_rows) - len(fails),
            "failures": len(fails),
            "by_level_solves": {
                f"L{lv}": sum(
                    1 for r in ep_rows
                    if r["episode_id"].startswith(f"L{lv}-")
                    and r["outcome"] == "solved")
                for lv in (4, 5, 6, 7)},
            "failure_outcomes": {
                r["episode_id"]: r["outcome"] for r in fails}},
        "failed_eps_with_eligible_fork": sum(
            1 for r in fails if r["forks"] >= 1),
        "label_bearing_failed_episodes": len(label_eps),
        "distinct_corrective_facts_global": len(global_facts),
        "raw_label_rows_total": sum(
            r["raw_label_rows"] for r in ep_rows),
        "censored_forks_total": sum(
            r["censored_forks"] for r in ep_rows),
        "overflow_forks_total": sum(
            r["overflow_forks"] for r in ep_rows),
        "horizon_sensitive_facts_total": sum(
            len(r["horizon_sensitive_facts"]) for r in ep_rows),
        "go_bars": {"a_label_bearing_eps_ge3": go_a,
                    "b_distinct_facts_ge5": go_b,
                    "GO": go_a and go_b},
        "mechanism_checks": checks,
        "pass": (all(checks.values()) if SMOKE else None),
        "world_states": len(world.cache),
        "start": START,
        "completion_commit": completion_commit()}
    VERDICT.write_text(json.dumps(verdict, indent=1))
    print(f"[yield] {'smoke ' if SMOKE else ''}solves="
          f"{verdict['baseline']['solves']}/{len(ep_rows)} "
          f"label_eps={len(label_eps)} facts={len(global_facts)}"
          f" GO={go_a and go_b}"
          + (f" pass={verdict['pass']}" if SMOKE else ""),
          flush=True)
    if SMOKE:
        return 0 if verdict["pass"] else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
