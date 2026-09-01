"""MATH-CYBER-1 CLOSED-LOOP-1 POLICY-SPLICE-DESK-0 — execute the
frozen splice prereg (96940b89d699c0478bab3e5d50358c99c36d0770,
RESULTS L59878) with the two record amendments carried as
factual/provenance corrections only (HEAD 62058c7e).

16 frozen roots x {AA, BB, BA, AB}: AA/BB = the verbatim
CLOSED-LOOP-1 arm policies rerun as in-process anchors; BA =
FORCE the booked MODEL depth-0 action then engine-hce tail;
AB = FORCE the booked ENGINE-HCE depth-0 action then model
tail. Forced identities come ONLY from the booked raw
(sha-gated); membership in the freshly enumerated stable legal
set is a hard gate — absent => SPLICE-INFEASIBLE, never
substituted. Budget: 12 decisions total (forced = decision 1),
60 s engine-side wall from the root (between-decision check),
model wall excluded; Latin-square cell order by root ordinal
k%4; per-segment wall emission is the sole instrumentation
change. Anchor comparison (action + outcome, four-class
partition); causal labels only on reproduced AA-solved/
BB-failed roots; no p-values. BA runs model-free; B-side
score/rank receipts join from BB after all four cells.

Raw first: logs/mathworld1/cl1/splice/ (refuse-if-exists),
trajectories streamed + hashed BEFORE any classification.

    .venv/bin/python scratch/mathworld1_cl1splice.py          (Mac)
"""
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402
import torch  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.search.derivation import State, hce, is_solved  # noqa: E402
from llmopt.train.mathnative import build_model  # noqa: E402
from scratch.mathworld1_cl1pop import materialize  # noqa: E402
from scratch.mathworld1_cl1run import (CKPT, CTX,  # noqa: E402
                                       LOCK, MANIFEST_SHA,
                                       MAX_DECISIONS, POPDIR,
                                       VOCAB, WALL_CAP_S,
                                       BIRTH_RECEIPT,
                                       b_score_decision,
                                       fsha, rebuild_root)
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpeval import stable_legal_set  # noqa: E402

PREREG_COMMIT = "96940b89d699c0478bab3e5d50358c99c36d0770"
RAW_CL1 = Path("logs/mathworld1/cl1/run/trajectories.jsonl")
RAW_CL1_SHA = ("22198ddf8d1e857a55bf98293d7aa101a5b0c29f48f474"
               "d30999ef4e872c1309")
CL1_RECEIPT = Path("logs/mathworld1/cl1/run/cl1run_receipt.json")
OUTDIR = Path("logs/mathworld1/cl1/splice")
ELIGIBLE = [12, 14, 15, 20, 23, 35, 37, 47, 54, 68, 72, 76, 78,
            88, 89, 94]
LATIN = [("AA", "BB", "BA", "AB"), ("BB", "BA", "AB", "AA"),
         ("BA", "AB", "AA", "BB"), ("AB", "AA", "BB", "BA")]
CELLS = ["AA", "BB", "BA", "AB"]
MAT_TIMEOUT_S = 120.0


def run_cell(cell, root, row, model, dev, sink, forced0):
    """One splice cell. forced0 = None for anchors (natural
    depth-0 policy) or the booked (name, child_sstr) pair for
    hybrids. Tail policy: 'A' for AA/BA, 'B' for BB/AB."""
    tail = "A" if cell in ("AA", "BA") else "B"
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
        parent_cur = sp.sstr(state.expr)
        legal = [{"name": n, "child_sstr": sp.sstr(c.expr)}
                 for n, c in acts]
        seg_legal = time.monotonic() - t0
        engine_wall += seg_legal
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
        row_d = {"depth": depth, "parent": parent_cur,
                 "n_legal": len(acts), "legal": legal,
                 "seg_legal_s": round(seg_legal, 4)}
        seg_policy_engine = seg_model = 0.0
        if depth == 0 and forced0 is not None:
            t0 = time.monotonic()
            hits = [i for i, (n, c) in enumerate(acts)
                    if n == forced0[0]
                    and legal[i]["child_sstr"] == forced0[1]]
            seg_policy_engine = time.monotonic() - t0
            row_d["forced"] = True
            row_d["forced_action"] = {"name": forced0[0],
                                      "child_sstr": forced0[1]}
            row_d["membership_gate"] = bool(hits)
            if not hits:
                outcome = "SPLICE-INFEASIBLE"
                row_d["event"] = "splice_infeasible"
                engine_wall += seg_policy_engine
                dec_rows.append(row_d)
                break
            ci = hits[0]
            name, child = acts[ci]
        elif (depth == 0 and cell == "AA") or (depth > 0
                                               and tail == "A"):
            t0 = time.monotonic()
            name, child = min(acts, key=lambda nc: (
                hce(nc[1]), nc[0], nc[1].key()))
            ci = next(i for i, (n, c) in enumerate(acts)
                      if n == name and c.key() == child.key())
            seg_policy_engine = time.monotonic() - t0
        else:
            tm = time.monotonic()
            ci, cands, pre_len, fail = b_score_decision(
                model, dev, state.expr, acts)
            seg_model = time.monotonic() - tm
            model_wall += seg_model
            model_calls += 1
            row_d["prompt_tokens"] = pre_len
            row_d["candidates"] = [
                {"name": c["name"],
                 "factor_code": c.get("fc"),
                 "in_domain": c.get("in_domain"),
                 "failed": c.get("failed"),
                 "score": c.get("score"),
                 "token_lps": c.get("token_lps")}
                for c in cands]
            if fail is not None:
                outcome = fail
                row_d["event"] = fail
                row_d["seg_model_s"] = round(seg_model, 4)
                dec_rows.append(row_d)
                break
            name, child = acts[ci]
        engine_wall += seg_policy_engine
        row_d["seg_policy_engine_s"] = round(seg_policy_engine, 4)
        row_d["seg_model_s"] = round(seg_model, 4)
        row_d["chosen_index"] = ci
        row_d["chosen_name"] = name
        row_d["chosen_child_sstr"] = sp.sstr(child.expr)
        n_dec += 1
        t0 = time.monotonic()
        ck = child.key()
        state = child
        if ck in visited:
            outcome = "cycle"
            row_d["seg_apply_s"] = round(time.monotonic() - t0, 4)
            engine_wall += time.monotonic() - t0
            dec_rows.append(row_d)
            break
        visited.add(ck)
        solved_now = is_solved(state)
        seg_apply = time.monotonic() - t0
        engine_wall += seg_apply
        row_d["seg_apply_s"] = round(seg_apply, 4)
        dec_rows.append(row_d)
        if solved_now:
            outcome = "solved"
            break
    summary = {"row_index": row["row_index"],
               "level": row["level"],
               "root_sha": row["root_sha"], "cell": cell,
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
    gate(fsha(RAW_CL1) == RAW_CL1_SHA, "CL1 RAW PIN")
    gate(fsha(POPDIR / "manifest.jsonl") == MANIFEST_SHA,
         "MANIFEST PIN")
    cl1r = json.loads(CL1_RECEIPT.read_text())
    gate(cl1r["raw_trajectories_sha"] == RAW_CL1_SHA,
         "RECEIPT RAW PIN")
    lock = json.loads(LOCK.read_text())["receipts"]
    birth_lock_sha = lock[str(BIRTH_RECEIPT)]["sha256"]
    gate(fsha(BIRTH_RECEIPT) == birth_lock_sha, "BIRTH LOCK PIN")
    birth = json.loads(BIRTH_RECEIPT.read_text())
    ckpt_sha = birth["checkpoints"][str(CKPT)]
    gate(fsha(CKPT) == ckpt_sha, "CKPT PIN")
    START = start_provenance(
        ["scratch/mathworld1_cl1splice.py",
         "scratch/mathworld1_cl1run.py",
         "scratch/mathworld1_cl1pop.py",
         "scratch/mathworld1_svpeval.py",
         "scratch/mathworld1_svpfohrepl.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld0.py",
         "llmopt/search/derivation.py", "llmopt/search/rules.py",
         "llmopt/train/mathnative.py",
         "llmopt/lab/provenance.py"])

    # booked depth-0 actions + anchor references from sealed raw
    booked = {}
    for l in open(RAW_CL1):
        t = json.loads(l)
        if t["row_index"] in ELIGIBLE and t["arm"] in ("A", "B"):
            d0 = t["decisions"][0]
            gate(d0.get("chosen_name") is not None,
                 f"NO BOOKED D0 {t['arm']} {t['row_index']}")
            booked[(t["arm"], t["row_index"])] = {
                "d0_name": d0["chosen_name"],
                "d0_child": d0["chosen_child_sstr"],
                "outcome": t["outcome"],
                "solved": t["solved"]}
    gate(len(booked) == 32, "BOOKED PAIRS")

    rows = {r["row_index"]: r for r in
            (json.loads(l)
             for l in open(POPDIR / "manifest.jsonl"))
            if r["row_index"] in ELIGIBLE}
    gate(sorted(rows) == ELIGIBLE, "ELIGIBLE ROWS")

    # roots rebuilt + byte-gated BEFORE namespace creation
    roots = {}
    for i in ELIGIBLE:
        res, fail = materialize(rows[i]["level"],
                                rows[i]["generator_seed"],
                                timeout=MAT_TIMEOUT_S)
        gate(fail is None, f"FORK PROBE {i}: {fail}")
        gate(res["cur"] == rows[i]["root_cur"],
             f"FORK PROBE BYTES {i}")
        roots[i] = rebuild_root(rows[i])

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
        for k, i in enumerate(ELIGIBLE):
            order = LATIN[k % 4]
            got = {}
            for cell in order:
                forced0 = None
                if cell == "BA":
                    b = booked[("B", i)]
                    forced0 = (b["d0_name"], b["d0_child"])
                elif cell == "AB":
                    a = booked[("A", i)]
                    forced0 = (a["d0_name"], a["d0_child"])
                got[cell] = run_cell(cell, roots[i], rows[i],
                                     model, dev, sink, forced0)
            summaries.extend(got[c] for c in CELLS)
            print(f"[cl1splice] root {i:2d} "
                  + " ".join(f"{c}:{got[c]['outcome']}"
                             for c in CELLS), flush=True)
    raw_sha = fsha(OUTDIR / "trajectories.jsonl")
    gate(len(summaries) == 64, "CELL COUNT")
    print(f"[cl1splice] RAW SEALED sha256={raw_sha}", flush=True)

    # ---- mechanical classification (post-hash) ----
    S = {(s["cell"], s["row_index"]): s for s in summaries}
    anchor_class = {}
    # re-read sealed raw for depth-0 actions (endpoints from raw)
    traj = {}
    for l in open(OUTDIR / "trajectories.jsonl"):
        t = json.loads(l)
        traj[(t["cell"], t["row_index"])] = t
    for i in ELIGIBLE:
        cls = {}
        for cell, arm in (("AA", "A"), ("BB", "B")):
            t = traj[(cell, i)]
            d0 = t["decisions"][0] if t["decisions"] else {}
            fresh_act = (d0.get("chosen_name"),
                         d0.get("chosen_child_sstr"))
            bk = booked[(arm, i)]
            act_ok = fresh_act == (bk["d0_name"], bk["d0_child"])
            out_ok = (t["outcome"] == bk["outcome"]
                      and t["solved"] == bk["solved"])
            cls[cell] = ("ANCHOR-REPRODUCED" if act_ok and out_ok
                         else "ANCHOR-ACTION-NONREPRODUCED"
                         if not act_ok and out_ok
                         else "ANCHOR-OUTCOME-NONREPRODUCED"
                         if act_ok else
                         "ANCHOR-BOTH-NONREPRODUCED")
        anchor_class[i] = cls

    labels = {}
    for i in ELIGIBLE:
        aa, bb = S[("AA", i)], S[("BB", i)]
        ba, ab = S[("BA", i)], S[("AB", i)]
        if (ba["outcome"] == "SPLICE-INFEASIBLE"
                or ab["outcome"] == "SPLICE-INFEASIBLE"):
            labels[i] = "SPLICE-INFEASIBLE"
            continue
        bkA, bkB = booked[("A", i)], booked[("B", i)]
        reproduced = (anchor_class[i]["AA"] == "ANCHOR-REPRODUCED"
                      and anchor_class[i]["BB"]
                      == "ANCHOR-REPRODUCED")
        if (bkA["solved"] and not bkB["solved"] and reproduced
                and aa["solved"] and not bb["solved"]):
            if not ba["solved"] and ab["solved"]:
                labels[i] = "FIRST-ACTION-SUFFICIENT"
            elif ba["solved"] and not ab["solved"]:
                labels[i] = "TAIL-POLICY-SUFFICIENT"
            elif not ba["solved"] and not ab["solved"]:
                labels[i] = "OVERDETERMINED"
            else:
                labels[i] = "CONJUNCTION-REQUIRED"
        elif not reproduced:
            labels[i] = "ANCHOR-SHIFTED"
        elif aa["solved"] and bb["solved"]:
            labels[i] = ("ROBUST-BOTH-SOLVED"
                         if ba["solved"] and ab["solved"]
                         else "SPLICE-FRAGILITY")
        elif not aa["solved"] and not bb["solved"]:
            labels[i] = ("HYBRID-WIN"
                         if ba["solved"] or ab["solved"]
                         else "ROBUST-BOTH-FAILED")
        else:
            labels[i] = "DESCRIPTIVE-OTHER"

    table = {i: {c: {"outcome": S[(c, i)]["outcome"],
                     "solved": S[(c, i)]["solved"],
                     "n_decisions": S[(c, i)]["n_decisions"],
                     "engine_wall_s": S[(c, i)]["engine_wall_s"],
                     "model_wall_s": S[(c, i)]["model_wall_s"]}
                 for c in CELLS} for i in ELIGIBLE}

    receipt = {
        "prereg": "MATH-CYBER-1-CLOSED-LOOP-1-POLICY-SPLICE-"
                  "PREREG-0",
        "prereg_commit": PREREG_COMMIT,
        "cl1_raw_pin": RAW_CL1_SHA,
        "manifest_sha": MANIFEST_SHA,
        "raw_splice_sha": raw_sha,
        "checkpoint_sha_derived": ckpt_sha,
        "anchor_census": {str(i): anchor_class[i]
                          for i in ELIGIBLE},
        "labels": {str(i): labels[i] for i in ELIGIBLE},
        "table": {str(i): table[i] for i in ELIGIBLE},
        "run_wall_s": round(time.time() - t_run, 1),
        "device": "mps", "torch": torch.__version__,
        "env": {"platform": platform.platform()},
        "start": START, "completion_commit": completion_commit()}
    gate(fsha(RAW_CL1) == RAW_CL1_SHA, "POST CL1 RAW PIN")
    gate(fsha(CKPT) == ckpt_sha, "POST CKPT PIN")
    gate(fsha(POPDIR / "manifest.jsonl") == MANIFEST_SHA,
         "POST MANIFEST PIN")
    (OUTDIR / "cl1splice_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in
                      ("anchor_census", "labels")}, indent=1),
          flush=True)

    # ---- search-externality riders (descriptive, post-receipt) --
    anat = {}
    for i in ELIGIBLE:
        aa = traj[("AA", i)]
        bb = traj[("BB", i)]
        d0 = aa["decisions"][0]
        # hce over the depth-0 legal set (A-side anatomy) needs
        # engine states: re-enumerate from the already-built root
        acts, _ = stable_legal_set(State(roots[i]))
        hvals = {}
        for n, c in acts:
            hvals[(n, sp.sstr(c.expr))] = float(hce(c))
        ent = {}
        for arm, t in (("A", aa), ("B", bb)):
            td = t["decisions"][0]
            ch = td.get("chosen_child_sstr")
            nm = td.get("chosen_name")
            child_legal = (t["decisions"][1]["n_legal"]
                           if len(t["decisions"]) > 1
                           and "n_legal" in t["decisions"][1]
                           else None)
            hv = hvals.get((nm, ch))
            rank = (1 + sum(1 for v in hvals.values()
                            if hv is not None and v < hv)
                    if hv is not None else None)
            bs = None
            brank = None
            c0 = bb["decisions"][0]
            if "candidates" in c0:
                sc = {x["name"]: x["score"]
                      for x in c0["candidates"]
                      if x.get("score") is not None}
                bs = sc.get(nm)
                if bs is not None:
                    brank = 1 + sum(1 for v in sc.values()
                                    if v > bs)
            ent[arm] = {
                "rule": (nm or "").split("@", 1)[0],
                "parent_len": len(td.get("parent", "")),
                "child_len": len(ch) if ch else None,
                "parent_n_legal": td.get("n_legal"),
                "child_n_legal": child_legal,
                "branching_ratio": (round(
                    child_legal / td["n_legal"], 2)
                    if child_legal and td.get("n_legal")
                    else None),
                "len_ratio": (round(len(ch)
                                    / len(td["parent"]), 2)
                              if ch and td.get("parent")
                              else None),
                "hce_value": hv, "hce_rank": rank,
                "model_score": bs, "model_rank": brank}
        anat[str(i)] = ent
    (OUTDIR / "riders.json").write_text(
        json.dumps({"depth0_anatomy": anat}, indent=1))
    print("[cl1splice] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
