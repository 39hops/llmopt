"""QWEN-MODEL1-TREE observations builder + mechanical tree walker.

Consumes the three scorer receipts (scratch/qwen_model1_score.py),
emits the observations document for docs/preregs/qwen-model1-tree.json,
runs the deterministic adjudicator, and maps bar outcomes to the
registered branch (-LOGIC evaluation order) with zero discretion:

  gates:  bar 1 ALIGN-SANITY and bar 2 UNIFORM-DAMAGE-CLEAN must
          FIRE, else INSTRUMENT-ALARM; bar 3 ANOMALY-CLEAN must
          FIRE, else NONMONOTONIC.
  steps:  a material-improvement step fires iff its X-relative,
          K-relative, and X-floor bars ALL fire; any NO-FIRE makes
          the step definitively not material; otherwise any
          UNRESOLVED bar leaves the step (and possibly the tree)
          UNRESOLVED (the registered degenerate/stop-for-review
          path rides in as contrast inadmissibility).
  branch: T1 iff step(B>A) and not step(C>B); T2 iff step(C>B);
          T4 iff neither step but step(C>A); else T3.

    .venv/bin/python scratch/qwen_tree_adjudicate.py

Writes logs/qwenmodel1/tree_observations.json and
logs/qwenmodel1/tree_verdict.txt.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

OUT = "logs/qwenmodel1"
PAIRS = [("A", "B", 4), ("B", "C", 7), ("A", "C", 10)]


def _m(value, metric, population, aggregation, provenance=""):
    return {"value": float(value), "metric": metric,
            "population": population, "aggregation": aggregation,
            "provenance": provenance}


def build_observations(rc: dict) -> dict:
    """rc = {"A": receipt, "B": receipt, "C": receipt} -> observations.

    Pure: no I/O. Degenerate X_Z <= 0 / K_Z <= 0 books the step's
    bars inadmissible via contrasts (registered UNRESOLVED path)."""
    arms = {}
    for a, r in rc.items():
        reasons = []
        if r.get("smoke"):
            reasons.append("smoke receipt")
        if r.get("device_actual") != "cpu":
            reasons.append(f"device {r.get('device_actual')}")
        tv = r.get("traversal", {})
        if (tv.get("linear_attn"), tv.get("full_attn")) != (48, 16):
            reasons.append(f"traversal {tv}")
        if not r.get("qualification"):
            reasons.append("no qualification report")
        arms[a] = {"admissible": not reasons}
        if reasons:
            arms[a]["reason"] = "; ".join(reasons)

    valid, vreason = True, ""
    base = rc["A"]["teacher"]
    for a in ("B", "C"):
        if rc[a]["teacher"] != base:
            valid, vreason = False, f"teacher identity differs in arm {a}"
    # ce_teacher/f_X are teacher-record-only quantities and must agree
    # bit-for-bit across receipts; f_K is arm-DEPENDENT by construction
    # (K's perturbation response runs through each arm's logits), so it
    # is consumed as max-over-arms, never equality-checked.
    for k in ("ce_teacher_nats", "f_X", "v_live"):
        vals = {a: rc[a][k] for a in rc}
        if len(set(vals.values())) != 1:
            valid, vreason = False, f"{k} differs across receipts {vals}"

    X = {a: rc[a]["X"] for a in rc}
    K = {a: rc[a]["K"] for a in rc}
    f_X = max(rc[a]["f_X"] for a in rc)
    f_K = max(rc[a]["f_K"] for a in rc)

    meas = {
        "1": _m(rc["A"]["ce_teacher_nats"], "ce_teacher_recomputed_nats",
                "positions:corpus", "mean", "locked records, P-1"),
        "2": _m(X["A"], "excess_ce_nats", "positions:corpus", "mean",
                "score_A.json"),
    }
    contrasts = {}
    for z, y, bid in PAIRS:
        prov = f"score_{z}.json v score_{y}.json"
        if X[z] <= 0.0:
            for b in (bid, bid + 2):
                contrasts[str(b)] = {
                    "admissible": False,
                    "reason": f"degenerate: X_{z} = {X[z]:.6f} <= 0 "
                              "(registered stop-for-review)"}
        else:
            meas[str(bid)] = _m((X[z] - X[y]) / X[z],
                                "rel_excess_ce_improvement",
                                "positions:corpus", "mean", prov)
            meas[str(bid + 2)] = _m(
                (X[z] - X[y]) / f_X if f_X > 0 else float("inf"),
                "excess_ce_delta_floor_multiple",
                "positions:corpus", "ratio", prov)
        if K[z] <= 0.0:
            contrasts[str(bid + 1)] = {
                "admissible": False,
                "reason": f"degenerate: K_{z} = {K[z]:.6f} <= 0 "
                          "(registered stop-for-review)"}
        else:
            meas[str(bid + 1)] = _m((K[z] - K[y]) / K[z],
                                    "rel_kl_improvement",
                                    "positions:prefixes", "mean", prov)

    # anomaly gate: material degradation up-rate, either metric,
    # each condition past its own 5x floor
    count = 0
    details = []
    for z, y in (("A", "B"), ("B", "C")):
        if X[y] > 0 and (X[y] - X[z]) >= 0.2 * X[y] \
                and (X[y] - X[z]) > 5 * f_X:
            count += 1
            details.append(f"X_{y} materially worse than X_{z}")
        if K[y] > 0 and (K[y] - K[z]) >= 0.2 * K[y] \
                and (K[y] - K[z]) > 5 * f_K:
            count += 1
            details.append(f"K_{y} materially worse than K_{z}")
    meas["3"] = _m(count, "anomaly_condition_count", "arm_pairs:up_rate",
                   "count", "; ".join(details) or "clean")

    obs = {"measurement_valid": valid, "arms": arms,
           "measurements": meas, "contrasts": contrasts,
           "X": X, "K": K, "f_X": f_X, "f_K": f_K}
    if not valid:
        obs["measurement_reason"] = vreason
    return obs


def step_state(outcomes: dict, ids) -> str:
    """'Y' all fire / 'N' any no-fire / 'U' otherwise-unresolved."""
    st = [outcomes[i] for i in ids]
    if any(s == "NO-FIRE" for s in st):
        return "N"
    if any(s == "UNRESOLVED" for s in st):
        return "U"
    return "Y"


def walk(outcomes: dict) -> tuple:
    """outcomes: {bar_id: 'FIRE'|'NO-FIRE'|'UNRESOLVED'} -> (branch,
    reason). Branch in {INSTRUMENT-ALARM, NONMONOTONIC, T1, T2, T3,
    T4, UNRESOLVED}."""
    for gate, label in ((1, "ALIGN-SANITY"), (2, "UNIFORM-DAMAGE")):
        if outcomes[gate] != "FIRE":
            return ("INSTRUMENT-ALARM",
                    f"gate {label} {outcomes[gate]} (fail-closed)")
    if outcomes[3] == "NO-FIRE":
        return ("NONMONOTONIC",
                "anomaly gate: material degradation up-rate")
    if outcomes[3] == "UNRESOLVED":
        return ("UNRESOLVED", "anomaly gate unresolved")
    s_ba = step_state(outcomes, (4, 5, 6))
    s_cb = step_state(outcomes, (7, 8, 9))
    s_ca = step_state(outcomes, (10, 11, 12))
    if s_cb == "U":
        return ("UNRESOLVED", "step C-over-B unresolved")
    if s_cb == "Y":
        return ("T2", "C materially improves over B")
    if s_ba == "U":
        return ("UNRESOLVED", "step B-over-A unresolved")
    if s_ba == "Y":
        return ("T1", "B materially improves over A, C does not over B")
    if s_ca == "U":
        return ("UNRESOLVED", "cumulative C-over-A unresolved")
    if s_ca == "Y":
        return ("T4", "cumulative C-over-A fires, no single step did")
    return ("T3", "functional parity: A is the runtime target")


def main():
    from llmopt.lab.prereg import (adjudicate_prereg,
                                   adjudicate_refutation, load)
    prereg = load("docs/preregs/qwen-model1-tree.json")
    rc = {}
    for a in ("A", "B", "C"):
        p = os.path.join(OUT, f"score_{a}.json")
        if not os.path.exists(p):
            raise SystemExit(f"REFUSING: missing receipt {p}")
        rc[a] = json.load(open(p))
        if rc[a].get("smoke"):
            raise SystemExit(f"REFUSING: smoke receipt at real path {p}")
    obs = build_observations(rc)
    # provenance: the walker's own executable identity + the exact
    # receipt bytes it consumed (receipt-audit adoption 2026-08-17)
    import hashlib
    import subprocess
    obs["provenance"] = {
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
        "tree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"]).decode().strip()),
        "receipt_sha256": {
            a: hashlib.sha256(open(os.path.join(
                OUT, f"score_{a}.json"), "rb").read()).hexdigest()
            for a in ("A", "B", "C")}}
    outcomes = {}
    lines = []
    for o in adjudicate_prereg(prereg, obs):
        outcomes[o.bar_id] = o.outcome
        line = f"BAR {o.bar_id} {o.bar_name}: {o.outcome}"
        if o.reasons:
            line += " [" + "; ".join(o.reasons) + "]"
        lines.append(line)
        print(line, flush=True)
    branch, reason = walk(outcomes)
    # the refutation clause reads ONLY on clean gates ("REFUTED if the
    # walker lands any branch other than T1 ON CLEAN GATES; a gate
    # alarm books no allocation claim" — prereg-auditor blocker,
    # 2026-08-17): under alarm/nonmonotonic/unresolved the prior is
    # UNADJUDICATED and the predicate measurement is withheld.
    if branch in ("T1", "T2", "T3", "T4"):
        obs["measurements"]["refuted:t1_prior_fired"] = _m(
            1.0 if branch == "T1" else 0.0, "t1_prior_fired",
            "tree:branch", "indicator", "walker")
        ref = adjudicate_refutation(prereg, obs)
    else:
        ref = "UNADJUDICATED (gate not clean — no allocation claim)"
    lines.append(f"BRANCH {branch}: {reason}")
    lines.append(f"REGISTERED-PRIOR(T1): {ref}")
    lines.append(f"PRODUCER {obs['provenance']['code_commit']}"
                 f" dirty={obs['provenance']['tree_dirty']}")
    print(lines[-2], flush=True)
    print(lines[-1], flush=True)
    with open(os.path.join(OUT, "tree_observations.json"), "w") as f:
        f.write(json.dumps(obs, indent=1) + "\n")
    with open(os.path.join(OUT, "tree_verdict.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[tree] -> {OUT}/tree_verdict.txt", flush=True)


if __name__ == "__main__":
    main()
