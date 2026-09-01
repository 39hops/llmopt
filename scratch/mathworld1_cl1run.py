"""MATH-CYBER-1 CLOSED-LOOP-1 JOINT-THREE-ARM-RUN-0 — execute the
frozen CLOSED-LOOP-1 experiment (prereg 082a.. entry
MATH-CYBER-1-CLOSED-LOOP-1-PREREG-0, prereg commit
082d4bc6ed32207a0d0b1fd5d85a56af0a9e4caf) on the sealed 96-root
population (manifest sha 50c05794...).

Three arms per root, identical root bytes, isolated state:
  A ENGINE-HCE      min over legal actions by (hce(child), name,
                    child.key()) — the standing episode policy.
  B MODEL-LEGAL-    frozen seed-19001 CANONICAL checkpoint scores
    RANKING         EVERY stable legal action: canonical
                    ActionProgram (derive_program law) -> factor
                    code + EOS (T=9) under the standing prompt;
                    total log-probability; argmax; exact ties by
                    lexicographically smallest factor code, then
                    engine (name, child-key) order. B never sees
                    hce, teacher labels, or future information.
  C RANDOM-LEGAL    index = int.from_bytes(sha256("closed-loop-1-
                    random-{root_full_sha}-{depth}").digest()[:8],
                    "big") % K over the engine-sorted legal set.
                    Descriptive only.

Episode law: MAX_DECISIONS=12; WALL_CAP_S=60 metered on
ENGINE-SIDE wall only (B's derive/encode/score wall excluded,
persisted as rider); stable_legal_set double-enumeration (unstable
-> legal_set_unstable); repeated full state key -> cycle;
CTX=4096 on prompt+9 (B only) -> context_overflow; B program
derivation/domain/tokenization failure -> action_encoding_failure.
All non-solved classes = NOT SOLVED. Arm order rotates per root:
row_index % 3 -> (A,B,C)/(B,C,A)/(C,A,B).

RAW-FIRST: per-decision trajectory receipts streamed to
logs/mathworld1/cl1/run/trajectories.jsonl (refuse-if-exists),
hashed BEFORE any endpoint. ONE inferential contrast: B v A
paired episode-level exact min-likelihood McNemar, alpha .05.
C descriptive. Encoding-confound gate co-reported.

    .venv/bin/python scratch/mathworld1_cl1run.py             (Mac)
"""
import hashlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402
import torch  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.search.derivation import State, hce, is_solved  # noqa: E402
from llmopt.train.mathnative import build_model  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpcode import (factor_symbols,  # noqa: E402
                                        in_domain)
from scratch.mathworld1_svpeval import (TOK, derive_program,  # noqa: E402
                                        stable_legal_set)
from scratch.mathworld1_svpfohrepl import (binom_minlik_p,  # noqa: E402
                                           token_lps)

MAX_DECISIONS = 12
WALL_CAP_S = 60.0
CTX = 4096
VOCAB = 340
CODE_BASE = 332
ALPHA = 0.05
X = sp.Symbol("x")
OUTDIR = Path("logs/mathworld1/cl1/run")
POPDIR = Path("logs/mathworld1/cl1/pop")
LOCK = Path("docs/receipts.lock.json")
BIRTH_RECEIPT = Path("logs/mathworld1/svpforder_s19001_receipt.json")
CKPT = Path("checkpoints/svp_forder_canonical_s19001.pt")
PAIRED = Path("data/matsub_paired.jsonl")
PAIRED_SHA = ("a943ba7fc581db743b07192e5d951fadd"
              "dd2ba19bca3225b75d8402351d468e8")
MANIFEST_SHA = ("50c05794d9773142c55b932f69685ad1c0124d168c12"
                "dd1f830e36a2944d1846")
PREREG_COMMIT = "082d4bc6ed32207a0d0b1fd5d85a56af0a9e4caf"
ARMS = ["A", "B", "C"]
ROTATION = [("A", "B", "C"), ("B", "C", "A"), ("C", "A", "B")]


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ssha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def rebuild_root(row):
    """Root from sealed manifest bytes: sympify(root_cur) with
    sstr-roundtrip + full-sha identity gates."""
    root = sp.sympify(row["root_cur"])
    cur = sp.sstr(root)
    gate(cur == row["root_cur"], "ROOT SSTR ROUNDTRIP")
    gate(ssha(cur) == row["root_sha"], "ROOT SHA")
    return root


def rng_choice(root_sha, depth, k):
    """Frozen C RNG law (execution-order-independent)."""
    seed = f"closed-loop-1-random-{root_sha}-{depth}"
    return int.from_bytes(
        hashlib.sha256(seed.encode("utf-8")).digest()[:8],
        "big") % k


def b_score_decision(model, dev, parent, acts):
    """Arm B one decision: programs + factor codes + total-lp
    scores for the COMPLETE engine-sorted legal set. Returns
    (choice_index, cand_rows, prompt_len, None) or
    (None, cand_rows_so_far, prompt_len, failure_class)."""
    cur = sp.sstr(parent)
    accepted = defaultdict(set)
    for n, c in acts:
        r = n.split("@", 1)[0] if "@" in n else n
        accepted[r].add(c.key())
    cands = []
    for n, c in acts:
        rule = n.split("@", 1)[0] if "@" in n else n
        try:
            prog, why = derive_program(parent, rule, c.key(),
                                       accepted)
            if prog is None:
                return None, cands, None, "action_encoding_failure"
            fc = factor_symbols(prog["rule"], prog["site_kind"],
                                prog["site_ordinal"],
                                prog["param_kind"],
                                prog["param_index"])
        except BaseException:
            return None, cands, None, "action_encoding_failure"
        cands.append({"name": n, "prog": prog, "fc": fc,
                      "in_domain": bool(in_domain(
                          prog["rule"], prog["site_kind"],
                          prog["site_ordinal"], prog["param_kind"],
                          prog["param_index"]))})
    pre_len = len(TOK.encode(f"Current: {cur}\nHints: none\nStep: "))
    if pre_len + 9 > CTX:
        return None, cands, pre_len, "context_overflow"
    conts = [[CODE_BASE + s for s in c["fc"]] + [TOK.eos_id]
             for c in cands]
    try:
        lps = token_lps(model, dev, cur, conts)
    except BaseException:
        return None, cands, pre_len, "action_encoding_failure"
    for c, v in zip(cands, lps):
        c["token_lps"] = v
        c["score"] = sum(v)
    best = max(c["score"] for c in cands)
    tied = [i for i, c in enumerate(cands)
            if c["score"] == best]
    # exact-tie law: smallest factor code, then engine order
    # (engine order = enumeration order; min() keeps first index)
    ci = min(tied, key=lambda i: (cands[i]["fc"], i))
    return ci, cands, pre_len, None


def run_arm(arm, root, root_sha, model, dev, sink, row_index,
            level):
    """One episode under one arm. Streams the full trajectory
    receipt; returns the terminal summary row."""
    state = State(root)
    visited = {state.key()}
    engine_wall = model_wall = 0.0
    engine_calls = model_calls = 0
    t_total = time.monotonic()
    outcome = "budget_exhausted"
    n_dec = 0
    dec_rows = []
    for depth in range(MAX_DECISIONS):
        t0 = time.monotonic()
        if is_solved(state):
            outcome = "solved"
            engine_wall += time.monotonic() - t0
            break
        if engine_wall > WALL_CAP_S:
            outcome = "wall_cap"
            break
        acts, stable = stable_legal_set(state)
        engine_calls += 1
        engine_wall += time.monotonic() - t0
        parent_cur = sp.sstr(state.expr)
        if not stable:
            outcome = "legal_set_unstable"
            dec_rows.append({"depth": depth, "parent": parent_cur,
                             "event": "legal_set_unstable"})
            break
        if not acts:
            outcome = "dead_end"
            dec_rows.append({"depth": depth, "parent": parent_cur,
                             "event": "dead_end"})
            break
        legal = [{"name": n, "child_sstr": sp.sstr(c.expr)}
                 for n, c in acts]
        row = {"depth": depth, "parent": parent_cur,
               "n_legal": len(acts), "legal": legal}
        if arm == "A":
            t0 = time.monotonic()
            name, child = min(acts, key=lambda nc: (
                hce(nc[1]), nc[0], nc[1].key()))
            ci = next(i for i, (n, c) in enumerate(acts)
                      if n == name and c.key() == child.key())
            engine_wall += time.monotonic() - t0
        elif arm == "C":
            t0 = time.monotonic()
            ci = rng_choice(root_sha, depth, len(acts))
            name, child = acts[ci]
            engine_wall += time.monotonic() - t0
        else:
            tm = time.monotonic()
            ci, cands, pre_len, fail = b_score_decision(
                model, dev, state.expr, acts)
            model_wall += time.monotonic() - tm
            model_calls += 1
            row["prompt_tokens"] = pre_len
            row["candidates"] = [
                {"name": c["name"], "factor_code": c["fc"],
                 "in_domain": c["in_domain"],
                 "score": c.get("score"),
                 "token_lps": c.get("token_lps")}
                for c in cands]
            if fail is not None:
                outcome = fail
                row["event"] = fail
                dec_rows.append(row)
                break
            name, child = acts[ci]
        row["chosen_index"] = ci
        row["chosen_name"] = name
        row["chosen_child_sstr"] = sp.sstr(child.expr)
        dec_rows.append(row)
        n_dec += 1
        t0 = time.monotonic()
        ck = child.key()
        state = child
        if ck in visited:
            outcome = "cycle"
            engine_wall += time.monotonic() - t0
            break
        visited.add(ck)
        solved_now = is_solved(state)
        engine_wall += time.monotonic() - t0
        if solved_now:
            outcome = "solved"
            break
    summary = {"row_index": row_index, "level": level,
               "root_sha": root_sha, "arm": arm,
               "outcome": outcome, "solved": outcome == "solved",
               "n_decisions": n_dec,
               "engine_calls": engine_calls,
               "model_calls": model_calls,
               "engine_wall_s": round(engine_wall, 3),
               "model_wall_s": round(model_wall, 3),
               "total_wall_s": round(
                   time.monotonic() - t_total, 3)}
    sink.write(json.dumps({**summary, "decisions": dec_rows})
               + "\n")
    sink.flush()
    return summary


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    # pins: population (booked manifest sha), training artifact,
    # checkpoint sha DERIVED from the lock-pinned birth receipt
    gate(fsha(POPDIR / "manifest.jsonl") == MANIFEST_SHA,
         "MANIFEST PIN")
    pop = json.loads((POPDIR / "receipt.json").read_text())
    gate(pop["manifest_sha"] == MANIFEST_SHA, "POP RECEIPT SHA")
    gate(pop["prereg_commit"] == PREREG_COMMIT, "PREREG PIN")
    gate(pop["verdict"] == "POPULATION MATERIALIZED + QUALIFIED",
         "POP AUTHORITY")
    gate(fsha(PAIRED) == PAIRED_SHA, "PAIRED PIN")
    lock = json.loads(LOCK.read_text())["receipts"]
    birth_lock_sha = lock[str(BIRTH_RECEIPT)]["sha256"]
    gate(fsha(BIRTH_RECEIPT) == birth_lock_sha, "BIRTH RECEIPT PIN")
    birth = json.loads(BIRTH_RECEIPT.read_text())
    ckpt_sha_expected = birth["checkpoints"][str(CKPT)]
    gate(fsha(CKPT) == ckpt_sha_expected, "CKPT PIN")

    START = start_provenance(
        ["scratch/mathworld1_cl1run.py",
         "scratch/mathworld1_cl1pop.py",
         "scratch/mathworld1_svpeval.py",
         "scratch/mathworld1_svpfohrepl.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_actionsem.py",
         "scratch/mathworld1_actionfinal.py",
         "scratch/mathworld1_axfixture.py",
         "scratch/mathworld0.py",
         "llmopt/search/derivation.py", "llmopt/search/rules.py",
         "llmopt/train/mathnative.py",
         "llmopt/lab/provenance.py"])

    rows = [json.loads(l)
            for l in open(POPDIR / "manifest.jsonl")]
    gate(len(rows) == 96, "N ROOTS")
    gate([r["row_index"] for r in rows] == list(range(96)),
         "ROW ORDER")
    gate(Counter(r["level"] for r in rows)
         == Counter({4: 24, 5: 24, 6: 24, 7: 24}), "LEVELS")

    # training-parent set (descriptive overlap rider only)
    train_par = set()
    for l in open(PAIRED):
        train_par.add(json.loads(l)["cur"])
    gate(len(train_par) == 58988, "TRAIN PARENT COUNT")

    gate(torch.backends.mps.is_available(), "MPS")
    dev = torch.device("mps")
    model = build_model(VOCAB, ctx=CTX)
    model.load_state_dict(torch.load(CKPT, weights_only=True))
    gate(sum(p.numel() for p in model.parameters()) == 19142016,
         "PARAM COUNT")
    model.eval()
    model = model.to(dev)

    OUTDIR.mkdir(parents=True)
    t_run = time.time()
    summaries = []
    with open(OUTDIR / "trajectories.jsonl", "w") as sink:
        for r in rows:
            order = ROTATION[r["row_index"] % 3]
            got = {}
            for arm in order:
                got[arm] = run_arm(
                    arm, rebuild_root(r), r["root_sha"], model,
                    dev, sink, r["row_index"], r["level"])
            summaries.extend(got[a] for a in ARMS)
            print(f"[cl1run] root {r['row_index']:2d} L{r['level']}"
                  + "".join(f" {a}:{got[a]['outcome']}"
                            for a in ARMS), flush=True)
    raw_sha = fsha(OUTDIR / "trajectories.jsonl")
    gate(len(summaries) == 288, "EPISODE COUNT")
    print(f"[cl1run] RAW SEALED sha256={raw_sha}", flush=True)

    # ---- endpoints (raw hashed above; derive from summaries) ----
    by = {a: {s["row_index"]: s for s in summaries
              if s["arm"] == a} for a in ARMS}
    solved = {a: sum(1 for s in by[a].values() if s["solved"])
              for a in ARMS}
    b_only = sum(1 for i in range(96)
                 if by["B"][i]["solved"]
                 and not by["A"][i]["solved"])
    a_only = sum(1 for i in range(96)
                 if by["A"][i]["solved"]
                 and not by["B"][i]["solved"])
    n_disc = a_only + b_only
    p = binom_minlik_p(b_only, n_disc)
    if b_only > a_only and p < ALPHA:
        verdict = "MODEL-BETTER"
    elif a_only > b_only and p < ALPHA:
        verdict = "MODEL-WORSE"
    else:
        verdict = "NO-DIRECTIONAL-SEPARATION"

    # encoding-confound gate (mandatory co-report)
    a_only_roots = [i for i in range(96)
                    if by["A"][i]["solved"]
                    and not by["B"][i]["solved"]]
    enc_fail_among = sum(
        1 for i in a_only_roots
        if by["B"][i]["outcome"] == "action_encoding_failure")
    gate_reading = None
    if verdict == "MODEL-WORSE":
        other = len(a_only_roots) - enc_fail_among
        gate_reading = (
            "preference/transport attribution PERMITTED"
            if other > len(a_only_roots) / 2 else
            "driven by encoder coverage, not measured preference")

    receipt = {
        "prereg": "MATH-CYBER-1-CLOSED-LOOP-1-PREREG-0",
        "prereg_commit": PREREG_COMMIT,
        "manifest_sha": MANIFEST_SHA,
        "raw_trajectories_sha": raw_sha,
        "checkpoint": {"path": str(CKPT),
                       "sha256_derived_from_birth_receipt":
                           ckpt_sha_expected,
                       "sha256_observed": fsha(CKPT)},
        "primary": {
            "verdict": verdict,
            "A_solved": solved["A"], "B_solved": solved["B"],
            "C_solved": solved["C"],
            "B_only": b_only, "A_only": a_only,
            "n_discordant": n_disc,
            "mcnemar_minlik_p_two_sided": p,
            "alpha": ALPHA,
            "B_minus_A_pp": round(
                100.0 * (solved["B"] - solved["A"]) / 96, 2)},
        "encoding_confound_gate": {
            "A_only_roots": len(a_only_roots),
            "B_action_encoding_failure_among_A_only":
                enc_fail_among,
            "reading_if_model_worse": gate_reading},
        "outcome_table": {a: dict(Counter(
            s["outcome"] for s in by[a].values()))
            for a in ARMS},
        "solved_by_level": {a: dict(Counter(
            f"L{s['level']}" for s in by[a].values()
            if s["solved"])) for a in ARMS},
        "wall": {a: {
            "engine_wall_s": round(sum(
                s["engine_wall_s"] for s in by[a].values()), 1),
            "model_wall_s": round(sum(
                s["model_wall_s"] for s in by[a].values()), 1),
            "total_wall_s": round(sum(
                s["total_wall_s"] for s in by[a].values()), 1),
            "engine_calls": sum(
                s["engine_calls"] for s in by[a].values()),
            "model_calls": sum(
                s["model_calls"] for s in by[a].values()),
            "decisions": sum(
                s["n_decisions"] for s in by[a].values())}
            for a in ARMS},
        "run_wall_s": round(time.time() - t_run, 1),
        "device": "mps", "torch": torch.__version__,
        "env": {"platform": platform.platform()},
        "start": START, "completion_commit": completion_commit()}
    gate(fsha(POPDIR / "manifest.jsonl") == MANIFEST_SHA,
         "POST MANIFEST PIN")
    gate(fsha(CKPT) == ckpt_sha_expected, "POST CKPT PIN")
    (OUTDIR / "cl1run_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in
                      ("primary", "encoding_confound_gate",
                       "outcome_table", "solved_by_level")},
                     indent=1), flush=True)

    # ---- descriptive riders (from the sealed raw) ----
    traj = defaultdict(dict)
    for l in open(OUTDIR / "trajectories.jsonl"):
        t = json.loads(l)
        traj[t["arm"]][t["row_index"]] = t
    legal_by_depth = {a: defaultdict(list) for a in ARMS}
    overlap_by_depth = {a: defaultdict(lambda: [0, 0])
                        for a in ARMS}
    dom = Counter()
    prompt_lens = []
    for a in ARMS:
        for t in traj[a].values():
            for d in t["decisions"]:
                if "n_legal" not in d:
                    continue
                legal_by_depth[a][d["depth"]].append(d["n_legal"])
                ob = overlap_by_depth[a][d["depth"]]
                ob[1] += 1
                if d["parent"] in train_par:
                    ob[0] += 1
                if a == "B":
                    if d.get("prompt_tokens") is not None:
                        prompt_lens.append(d["prompt_tokens"])
                    for c in d.get("candidates", []):
                        dom["in" if c["in_domain"] else "out"] += 1

    def dist(xs):
        xs = sorted(xs)
        if not xs:
            return None
        return {"n": len(xs), "p50": xs[len(xs) // 2],
                "max": xs[-1]}

    joint = [i for i in range(96) if by["A"][i]["solved"]
             and by["B"][i]["solved"]]
    diverge = []
    for i in range(96):
        da = traj["A"][i]["decisions"]
        db = traj["B"][i]["decisions"]
        dd = None
        for k in range(min(len(da), len(db))):
            if (da[k].get("chosen_child_sstr")
                    != db[k].get("chosen_child_sstr")):
                dd = k
                break
        diverge.append(
            {"row_index": i, "divergence_depth": dd,
             "A_outcome": by["A"][i]["outcome"],
             "B_outcome": by["B"][i]["outcome"],
             "A_solved": by["A"][i]["solved"],
             "B_solved": by["B"][i]["solved"]})
    disagree = [d for d in diverge
                if d["A_solved"] != d["B_solved"]]
    riders = {
        "legal_set_size_by_arm_depth": {
            a: {str(d): dist(v)
                for d, v in sorted(legal_by_depth[a].items())}
            for a in ARMS},
        "training_parent_overlap_by_arm_depth": {
            a: {str(d): {"overlap": v[0], "n": v[1]}
                for d, v in sorted(overlap_by_depth[a].items())}
            for a in ARMS},
        "B_actionprogram_domain": dict(dom),
        "B_prompt_tokens": dist(prompt_lens),
        "cycle_by_arm": {a: sum(
            1 for s in by[a].values() if s["outcome"] == "cycle")
            for a in ARMS},
        "steps_to_solve_joint_AB": {
            "n_joint": len(joint),
            "A": dist([by["A"][i]["n_decisions"]
                       for i in joint]),
            "B": dist([by["B"][i]["n_decisions"]
                       for i in joint]),
            "B_minus_A_per_root": dict(Counter(
                by["B"][i]["n_decisions"]
                - by["A"][i]["n_decisions"] for i in joint))},
        "AB_divergence_all": diverge,
        "AB_disagreement_anatomy": disagree}
    (OUTDIR / "riders.json").write_text(
        json.dumps(riders, indent=1))
    print("[cl1run] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
