"""MATH-CYBER-1 ACTIVE-EPISODIC-0 lockstep paired driver
(PRE-REG MATH-CYBER-1-ACTIVE-EPISODIC-0, commit 73c063af).
Two arms — ACTIVE-EPISODIC (success-gated episode-boundary
updates) and FROZEN (theta_0, zero updates) — share theta_0,
episode order, roots, budgets, overflow law, and ONE world
snapshot, in ONE driver.

World-snapshot discipline (hardened): the complete-world cache
keys on FULL State.key() and stores semantic (rule label, child
expr) pairs; 16-hex hashes are receipt fields only. At each
logical step the driver collects both arms' needed current
states, dedups and sorts by full key, and materializes every
missing state BEFORE scoring either arm; before each NEW state's
first successors() call it clears derivation._RULE_CACHE so a
prior state's load-induced rule timeout cannot become sticky
missing-moves (RULE_WALL itself unchanged). Arm-local State
metadata is reconstructed on cache hits.

Dose (frozen, -DESIGN (c) + -SUBSTRATE-DESK-0-SCOPE (d) +
ACTIVE-EPISODIC-0): after each SOLVED episode exactly one AdamW
step (lr 1e-4, betas 0.9/0.95, wd 0, persistent state), loss =
child-token mean CE incl. terminating newline per decision,
decisions averaged equally; failed episodes = zero optimizer
activity; training sequences up to ctx 4096 (birth SEQ_CAP does
NOT apply); nonfinite loss/grad or backward OOM = TREATMENT
INSTRUMENT FAILURE, exit 4, never fallback/truncate. No grad
clipping (none is registered in the dose).

SMOKE=1 (the only mode runnable before the launch GO):
mechanism-complete on SPENT CALIBRATION seeds only —
  ep1 L4-s9100 (theta_0 solves it: exercises ACTIVE success +
      one real update),
  ep2 L4-s9104 (theta_0 fails it: zero-update path),
  ep3 L4-s9101 with a 150-token ctx override (synthetic
      overflow-path exercise; both arms book model_ctx_overflow),
then a final FROZEN-evaluation pass over ep1+ep2 for both final
policies (ACTIVE-final differs from theta_0 after the update —
divergent states exercise the shared cache). Receipts on smoke_
paths. The REAL mode (contamination audit -> ADAPT 9300-9309
seed-major interleaved with episode-boundary updates -> HOLDOUT
9400-9409 zero-update evaluation of both final policies ->
MINLEN static diagnostic) runs only with MW1_ACTIVE_GO=1 —
granted by Artin + outside GO after the persistent charged-wall
re-smoke (10/10).

Receipts: logs/mathworld1/{PRE}active_pair.jsonl (per-decision
rows both arms + update rows + episode rows),
{PRE}active_pair_verdict.json. Refuse-if-exists.

    SMOKE=1 .venv/bin/python scratch/mathworld1_active.py    (Mac)
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
if not SMOKE and os.environ.get("MW1_ACTIVE_GO") != "1":
    raise SystemExit("REFUSING: real mode needs MW1_ACTIVE_GO=1 "
                     "(launch GO gate); SMOKE=1 for qualification")
PRE = (f"smoke{os.environ.get('SMOKE_TAG', '')}_"
       if SMOKE else "")   # SMOKE_TAG: new receipt path per requalification
CKPT = Path("checkpoints/mathnative_19m_mw1_theta0.pt")
ROWS = Path(f"logs/mathworld1/{PRE}active_pair.jsonl")
VERDICT = Path(f"logs/mathworld1/{PRE}active_pair_verdict.json")
ACTIVE_CKPT = Path(
    f"checkpoints/{PRE}mathnative_19m_mw1_active_final.pt")
MAX_DECISIONS = 12
WALL_CAP_S = 60.0
CTX = 4096
X = sp.Symbol("x")


def _finite(x) -> bool:
    """Nonfinite guard predicate (NaN, +/-inf)."""
    return math.isfinite(x)


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


class World:
    """Complete legal-set cache, FULL-key-keyed, semantic values."""

    def __init__(self):
        self.cache = {}   # State.key() -> [(name, child_expr)]
        self.walls = {}   # State.key() -> first-materialization s

    def materialize(self, keys_states):
        """keys_states: {full_key: representative State} for every
        state both arms need this step; enumerate missing ones.
        Records each state's first-materialization wall — the
        CHARGED-WALL unit each arm pays once on first use."""
        for k in sorted(set(keys_states) - set(self.cache)):
            derivation._RULE_CACHE.clear()
            st = keys_states[k]
            t0 = time.monotonic()
            acts = sorted(successors(st),
                          key=lambda nc: (nc[0], nc[1].key()))
            self.walls[k] = time.monotonic() - t0
            self.cache[k] = [(n, c.expr) for n, c in acts]

    def legal(self, st: State):
        """Arm-local States reconstructed from semantic values."""
        acts = [(n, State(e, st.plies + 1, st.history + (n,)))
                for n, e in self.cache[st.key()]]
        ah = sha("\n".join(f"{n}|{c.key()}" for n, c in acts))
        return acts, ah


class Arm:
    def __init__(self, name, model, tok, dev, sink, trainable):
        self.name = name
        self.model = model
        self.tok = tok
        self.dev = dev
        self.sink = sink
        self.trainable = trainable
        self.opt = (torch.optim.AdamW(
            model.parameters(), lr=1e-4, betas=(0.9, 0.95),
            weight_decay=0.0) if trainable else None)
        self.updates = 0
        self.inject_delay_s = 0.0   # smoke-only synthetic charge
        # PERSISTENT across all run_pair calls (WALL amendment
        # conformance): an arm pays a state's materialization
        # wall once per EXPERIMENT, not once per episode
        self.world_keys_paid = set()

    def score(self, prefix_ids, child_ids):
        ids = torch.tensor([prefix_ids + child_ids],
                           device=self.dev)
        with torch.no_grad():
            logits = self.model(ids)
        lp = torch.log_softmax(logits[0].float(), -1)
        s = sum(lp[len(prefix_ids) + i - 1, t].item()
                for i, t in enumerate(child_ids))
        if not _finite(s):                # nonfinite score guard
            print("[FAIL] nonfinite candidate score", flush=True)
            sys.exit(4)
        return s

    def update_on(self, trajectory, eid):
        """trajectory: [(prefix_ids, child_ids)] of the SOLVED
        episode's chosen transitions. One step, frozen dose."""
        self.model.train()
        losses = []
        n = len(trajectory)
        for prefix_ids, child_ids in trajectory:
            ids = torch.tensor([prefix_ids + child_ids],
                               device=self.dev)
            logits = self.model(ids)
            lp = logits[0, len(prefix_ids) - 1:-1]
            tgt = torch.tensor(child_ids, device=self.dev)
            loss = torch.nn.functional.cross_entropy(
                lp.float(), tgt)          # child-token mean CE
            try:
                (loss / n).backward()     # equal decision weights
            except RuntimeError as e:     # backward OOM class
                print(f"[FAIL] backward error at {eid}: "
                      f"{str(e)[:120]}", flush=True)
                sys.exit(4)
            losses.append(float(loss.detach()))
        gn = torch.sqrt(sum(
            p.grad.detach().float().pow(2).sum()
            for p in self.model.parameters()
            if p.grad is not None))
        if not torch.isfinite(gn) or not all(
                map(lambda x: x == x and abs(x) != float("inf"),
                    losses)):
            print(f"[FAIL] nonfinite loss/grad at {eid}",
                  flush=True)
            sys.exit(4)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.model.eval()
        self.updates += 1
        self.sink.write(json.dumps({
            "arm": self.name, "row": "update", "episode_id": eid,
            "update_index": self.updates,
            "loss_mean": round(sum(losses) / n, 4),
            "grad_norm": round(float(gn), 4),
            "trajectory_edges": n}) + "\n")
        self.sink.flush()


def run_pair(episodes, world, arms, tok, ctx_overrides, sink,
             stage, update_active=True):
    """Lockstep: both arms walk each episode in parallel steps."""
    results = {a.name: {} for a in arms}
    divergent_steps = 0
    for eid, root in episodes:
        ctx = ctx_overrides.get(eid, CTX)
        state = {a.name: State(root) for a in arms}
        alive = {a.name: True for a in arms}
        outcome = {a.name: "budget_exhausted" for a in arms}
        traj = {a.name: [] for a in arms}
        # CHARGED WALL, per arm per episode: each arm pays a
        # state's first-materialization wall ONCE on its own
        # first use of that state, plus only its OWN candidate
        # scoring wall; shared states charge the same recorded
        # duration to both arms independently; one arm's private
        # cost never reaches the other; updates (post-episode)
        # excluded. Cap checked BEFORE the next decision — the
        # registered one-decision overshoot semantics.
        charged = {a.name: a.inject_delay_s for a in arms}
        for a in arms:
            a.inject_delay_s = 0.0   # consumed once
        for step_id in range(MAX_DECISIONS):
            need = {}
            for a in arms:
                if not alive[a.name]:
                    continue
                st = state[a.name]
                if is_solved(st):
                    outcome[a.name] = "solved"
                    alive[a.name] = False
                    continue
                if charged[a.name] > WALL_CAP_S:
                    outcome[a.name] = "wall_cap"
                    alive[a.name] = False
                    continue
                need[st.key()] = st
            if not need:
                break
            if len(need) > 1:
                divergent_steps += 1
            world.materialize(need)   # BEFORE scoring either arm
            for a in arms:
                if not alive[a.name]:
                    continue
                st = state[a.name]
                k = st.key()
                if k not in a.world_keys_paid:
                    charged[a.name] += world.walls.get(k, 0.0)
                    a.world_keys_paid.add(k)
                t_score = time.monotonic()
                acts, ah = world.legal(st)
                if not acts:
                    outcome[a.name] = "dead_end"
                    alive[a.name] = False
                    charged[a.name] += time.monotonic() - t_score
                    continue
                parent = str(st.expr)
                pre_ids = tok.encode(
                    f"Current: {parent}\nHints: none\nStep: ")
                cand, over = [], False
                for name, c in acts:
                    cids = tok.encode(str(c.expr) + "\n")
                    if len(pre_ids) + len(cids) > ctx:
                        over = True
                        break
                    cand.append((name, c, cids))
                if over:
                    outcome[a.name] = "model_ctx_overflow"
                    alive[a.name] = False
                    charged[a.name] += time.monotonic() - t_score
                    sink.write(json.dumps({
                        "arm": a.name, "row": "decision",
                        "stage": stage, "ctx": ctx,
                        "episode_id": eid, "step_id": step_id,
                        "state_hash": sha(st.key()),
                        "legal_set_hash": ah,
                        "n_legal": len(acts),
                        "event": "model_ctx_overflow"}) + "\n")
                    continue
                scored = [(a.score(pre_ids, cids), name, c, cids)
                          for name, c, cids in cand]
                scored.sort(
                    key=lambda t: (-t[0], t[1], t[2].key()))
                s_best, name, child, cids = scored[0]
                sink.write(json.dumps({
                    "arm": a.name, "row": "decision",
                    "stage": stage, "ctx": ctx,
                    "episode_id": eid, "step_id": step_id,
                    "state_hash": sha(st.key()),
                    "legal_set_hash": ah, "n_legal": len(acts),
                    "chosen": f"{name}#{sha(child.key())}",
                    "score": round(s_best, 4)}) + "\n")
                traj[a.name].append((pre_ids, cids))
                state[a.name] = child
                charged[a.name] += time.monotonic() - t_score
                if is_solved(child):
                    outcome[a.name] = "solved"
                    alive[a.name] = False
        for a in arms:
            results[a.name][eid] = outcome[a.name]
            sink.write(json.dumps({
                "arm": a.name, "row": "episode", "stage": stage,
                "episode_id": eid,
                "outcome": outcome[a.name],
                "charged_wall_s": round(charged[a.name], 3)})
                + "\n")
            if (update_active and a.trainable
                    and outcome[a.name] == "solved"):
                a.update_on(traj[a.name], eid)
        sink.flush()
        print(f"[pair] {eid}: " + " ".join(
            f"{a.name}={outcome[a.name]}" for a in arms),
            flush=True)
    return results, divergent_steps


def main():
    for p in (ROWS, VERDICT, ACTIVE_CKPT):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    if not CKPT.exists():
        raise SystemExit(f"MISSING theta_0: {CKPT}")
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/mathworld1_active.py",
         "scratch/mathworld1_birth.py",
         "llmopt/search/derivation.py",
         "llmopt/mathgen/problems.py",
         "llmopt/train/mathnative.py"])
    ck_sha = hashlib.sha256(CKPT.read_bytes()).hexdigest()
    tok = GCTok()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    def load():
        m = build_model(tok.vocab_size, ctx=CTX).to(dev)
        m.load_state_dict(torch.load(CKPT, map_location=dev))
        m.eval()
        return m

    def ep(lv, seed):
        p = make_integrate(lv, seed)
        return (f"L{lv}-s{seed}", sp.Integral(p._expr, X))

    if not SMOKE:
        return real_main(START, ck_sha, tok, dev, load, ep)
    # ---- SMOKE (spent CALIBRATION seeds only) ----

    episodes = [ep(4, 9100), ep(4, 9104)]
    over_ep = ep(4, 9101)
    world = World()
    with ROWS.open("a") as f:
        active = Arm("ACTIVE", load(), tok, dev, f,
                     trainable=True)
        frozen = Arm("FROZEN", load(), tok, dev, f,
                     trainable=False)
        arms = [active, frozen]
        res1, div1 = run_pair(episodes, world, arms, tok, {}, f,
                              stage="adapt_smoke")
        res2, _ = run_pair([over_ep], world, arms, tok,
                           {over_ep[0]: 150}, f,
                           stage="overflow_smoke")
        # DIVERGENCE stage: distinct-weight pair (theta_0 v the
        # smoke-birth checkpoint) walks a 12-decision episode so
        # the arms genuinely branch and the shared cache serves
        # two different states in one materialize() call
        div_model = build_model(tok.vocab_size, ctx=CTX).to(dev)
        div_model.load_state_dict(torch.load(
            "checkpoints/smoke_mathnative_19m_mw1_theta0.pt",
            map_location=dev))
        div_model.eval()
        divarm = Arm("DIVPROBE", div_model, tok, dev, f,
                     trainable=False)
        _, div2 = run_pair([ep(4, 9104)], world,
                           [frozen, divarm], tok, {}, f,
                           stage="divergence_smoke",
                           update_active=False)
        # WALL-ISOLATION stage: inject a synthetic 61 s charge
        # into DIVPROBE only; it must wall_cap while FROZEN
        # completes — one arm's cost can never cap the other
        divarm.inject_delay_s = 61.0
        res4, _ = run_pair([ep(4, 9100)], world,
                           [frozen, divarm], tok, {}, f,
                           stage="wall_isolation_smoke",
                           update_active=False)
        # final frozen evaluation of BOTH final policies
        for a in arms:
            a.trainable = False
        res3, _ = run_pair(episodes, world, arms, tok, {}, f,
                           stage="frozen_eval_smoke",
                           update_active=False)
        f.write(json.dumps({"meta": {
            "start": START,
            "completion_commit": completion_commit()}}) + "\n")
    torch.save(active.model.state_dict(), ACTIVE_CKPT)
    a_sha = hashlib.sha256(ACTIVE_CKPT.read_bytes()).hexdigest()
    upd_rows = [json.loads(l) for l in ROWS.open()
                if '"row": "update"' in l]
    ep_rows = [json.loads(l) for l in ROWS.open()
               if '"row": "episode"' in l]

    def wall_of(arm, stage, eid):
        r = [x for x in ep_rows if x["arm"] == arm
             and x["stage"] == stage and x["episode_id"] == eid]
        assert len(r) == 1, (arm, stage, eid)
        return r[0]["charged_wall_s"]

    root_key_9100 = State(episodes[0][1]).key()
    root_key_9104 = State(episodes[1][1]).key()
    checks = {
        "active_update_count_exact": active.updates == 1,
        "no_update_row_on_failed_episode": not any(
            u["episode_id"] == "L4-s9104" for u in upd_rows),
        "overflow_both_arms": all(
            res2[n]["L4-s9101"] == "model_ctx_overflow"
            for n in ("ACTIVE", "FROZEN")),
        "final_ckpt_differs_theta0": a_sha != ck_sha,
        "divergent_states_seen": div2 >= 1,
        "wall_isolation": (
            res4["DIVPROBE"]["L4-s9100"] == "wall_cap"
            and res4["FROZEN"]["L4-s9100"] == "solved"),
        "nonfinite_guard_static": not any(
            _finite(x) for x in (float("nan"), float("inf"),
                                 -float("inf"))),
        "frozen_eval_matches_stage1_frozen":
            res3["FROZEN"] == res1["FROZEN"],
        "revisit_zero_recharge":
            wall_of("FROZEN", "frozen_eval_smoke", "L4-s9100")
            < world.walls[root_key_9100] + 0.5
            and wall_of("FROZEN", "adapt_smoke", "L4-s9100")
            >= world.walls[root_key_9100] - 1e-9,
        "other_arm_first_use_pays_once":
            wall_of("DIVPROBE", "divergence_smoke", "L4-s9104")
            >= world.walls[root_key_9104] - 1e-9}
    verdict = {
        "smoke": True, "device": dev,
        "theta0_sha256": ck_sha,
        "active_final_sha256": a_sha,
        "optimizer_steps": active.updates,
        "stage1": res1, "stage2_overflow": res2,
        "stage3_frozen_eval": res3,
        "stage4_wall_isolation": res4,
        "divergent_lockstep_steps": div2,
        "stage1_divergent_steps": div1,
        "world_states_materialized": len(world.cache),
        "mechanism_checks": checks,
        "pass": all(checks.values()),
        "start": START,
        "completion_commit": completion_commit()}
    VERDICT.write_text(json.dumps(verdict, indent=1))
    print(f"[pair] smoke pass={verdict['pass']} checks={checks}",
          flush=True)
    return 0 if verdict["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())


def real_main(START, ck_sha, tok, dev, load, ep):
    """REAL one-shot: contamination audit -> ADAPT (seed-major
    interleaved 9300-9309) -> HOLDOUT (9400-9409, zero updates,
    evaluated once per arm) -> MINLEN static diagnostic. Frozen
    per PRE-REG MATH-CYBER-1-ACTIVE-EPISODIC-0 + -WALL."""
    import glob
    adapt = [ep(lv, s) for s in range(9300, 9310)
             for lv in (4, 5, 6, 7)]
    holdout = [ep(lv, s) for s in range(9400, 9410)
               for lv in (4, 5, 6, 7)]
    # --- contamination audit (frozen law; disclose, never fix) ---
    diet_cur, diet_nxt = set(), set()
    for fpath in (sorted(glob.glob(
            "data/micromodel_chains_shard*.jsonl"))
            + ["data/step_chains.jsonl"]):
        for line in open(fpath):
            r = json.loads(line)
            diet_cur.add(r["cur"])
            diet_nxt.add(r["nxt"])
    calib = {str(sp.Integral(make_integrate(lv, s)._expr, X))
             for lv in (4, 5, 6, 7) for s in range(9100, 9110)}
    train_band = {str(sp.Integral(make_integrate(lv, s)._expr, X))
                  for lv in (4, 5, 6, 7)
                  for s in range(9200, 9250)}
    contamination = {}
    seen_roots = {}
    for eid, root in adapt + holdout:
        rs = str(root)
        hits = []
        if rs in diet_cur or rs in diet_nxt:
            hits.append("birth_diet")
        if rs in calib:
            hits.append("calibration")
        if rs in train_band:
            hits.append("train_band")
        if rs in seen_roots:
            hits.append(f"duplicate_of:{seen_roots[rs]}")
        seen_roots.setdefault(rs, eid)
        if hits:
            contamination[eid] = hits
    print(f"[real] contamination: {len(contamination)} flagged "
          f"root(s): {contamination}", flush=True)
    world = World()
    with ROWS.open("a") as f:
        active = Arm("ACTIVE", load(), tok, dev, f,
                     trainable=True)
        frozen = Arm("FROZEN", load(), tok, dev, f,
                     trainable=False)
        arms = [active, frozen]
        res_adapt, div_a = run_pair(adapt, world, arms, tok, {},
                                    f, stage="adapt")
        active.trainable = False   # theta_final frozen
        res_hold, div_h = run_pair(holdout, world, arms, tok, {},
                                   f, stage="holdout",
                                   update_active=False)
        # MINLEN static diagnostic on HOLDOUT (no model; same
        # snapshot; chooses the shortest scoring sequence,
        # tie-break (len, name, child key)); overflow law applies
        minlen_out = {}
        for eid, root in holdout:
            st = State(root)
            outcome = "budget_exhausted"
            for step_id in range(MAX_DECISIONS):
                if is_solved(st):
                    outcome = "solved"
                    break
                world.materialize({st.key(): st})
                acts, ah = world.legal(st)
                if not acts:
                    outcome = "dead_end"
                    break
                pre = tok.encode(f"Current: {str(st.expr)}\n"
                                 f"Hints: none\nStep: ")
                cand = []
                over = False
                for name, c in acts:
                    cids = tok.encode(str(c.expr) + "\n")
                    if len(pre) + len(cids) > CTX:
                        over = True
                        break
                    cand.append((len(cids), name, c))
                if over:
                    outcome = "model_ctx_overflow"
                    break
                cand.sort(key=lambda t: (t[0], t[1], t[2].key()))
                _, name, child = cand[0]
                f.write(json.dumps({
                    "arm": "MINLEN", "row": "decision",
                    "stage": "holdout_diag", "ctx": CTX,
                    "episode_id": eid, "step_id": step_id,
                    "state_hash": sha(st.key()),
                    "legal_set_hash": ah, "n_legal": len(acts),
                    "chosen": f"{name}#{sha(child.key())}"})
                    + "\n")
                st = child
                if is_solved(st):
                    outcome = "solved"
                    break
            minlen_out[eid] = outcome
            f.write(json.dumps({
                "arm": "MINLEN", "row": "episode",
                "stage": "holdout_diag", "episode_id": eid,
                "outcome": outcome}) + "\n")
        f.write(json.dumps({"meta": {
            "start": START,
            "completion_commit": completion_commit()}}) + "\n")
    torch.save(active.model.state_dict(), ACTIVE_CKPT)
    a_sha = hashlib.sha256(ACTIVE_CKPT.read_bytes()).hexdigest()

    def solves(res, arm):
        return sum(1 for o in res[arm].values() if o == "solved")

    verdict = {
        "smoke": False, "device": dev,
        "theta0_sha256": ck_sha,
        "active_final_sha256": a_sha,
        "optimizer_steps": active.updates,
        "contamination": contamination,
        "adapt": {"ACTIVE": res_adapt["ACTIVE"],
                  "FROZEN": res_adapt["FROZEN"],
                  "acquisition_solves_active":
                      solves(res_adapt, "ACTIVE"),
                  "frozen_solves": solves(res_adapt, "FROZEN"),
                  "divergent_lockstep_steps": div_a},
        "holdout": {"ACTIVE": res_hold["ACTIVE"],
                    "FROZEN": res_hold["FROZEN"],
                    "active_final_solves":
                        solves(res_hold, "ACTIVE"),
                    "frozen_solves": solves(res_hold, "FROZEN"),
                    "divergent_lockstep_steps": div_h},
        "minlen_diag": {"outcomes": minlen_out,
                        "solves": sum(1 for o in minlen_out
                                      .values()
                                      if o == "solved")},
        "adjudication": {
            "perfect_bar_active_holdout_40":
                solves(res_hold, "ACTIVE") == 40,
            "feedback_direction_active_gt_frozen":
                solves(res_hold, "ACTIVE")
                > solves(res_hold, "FROZEN")},
        "world_states_materialized": len(world.cache),
        "start": START,
        "completion_commit": completion_commit()}
    VERDICT.write_text(json.dumps(verdict, indent=1))
    print(f"[real] ADAPT ACTIVE {solves(res_adapt, 'ACTIVE')}/40 "
          f"FROZEN {solves(res_adapt, 'FROZEN')}/40 | HOLDOUT "
          f"ACTIVE {solves(res_hold, 'ACTIVE')}/40 FROZEN "
          f"{solves(res_hold, 'FROZEN')}/40 MINLEN "
          f"{verdict['minlen_diag']['solves']}/40 | "
          f"updates {active.updates}", flush=True)
    return 0
