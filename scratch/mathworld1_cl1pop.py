"""MATH-CYBER-1 CLOSED-LOOP-1 population builder (prereg
082d4bc6, materialization stage ONLY). Materializes, qualifies,
freezes, and hashes the exact 96-root population (24 per level
L4-L7; per-level deterministic seed scan from 500000 with the
frozen skip-and-continue law). OUTCOME-BLIND by construction:
this module imports NO checkpoint, NO scoring/ranking function,
NO run_episode, NO hce, NO random policy — only sympy,
make_integrate (fork-isolated: L4+ construction is the named
hang class), and the engine's State/is_solved for the frozen
depth-0 structural filter.

Order of operations (raw-first):
  1 build the EXCLUSION UNIVERSE from authoritative artifacts
    (per-category counts + union + pairwise intersections),
    persist + hash the receipt BEFORE any seed is scanned;
    category B eval ROOTS are re-materialized fork-isolated
    from the 240 (level, seed) pairs in the episodes artifacts
    (their root curs are not stored) plus all decision parents;
  2 per level, scan seeds 500000.. ; per candidate emit ONE
    raw attempt row (level, seed, status, canonical root cur,
    full sha, depth0_is_solved, exclusion category on match,
    accepted/rejected + single reason under the frozen
    precedence: materialization_failure/timeout ->
    depth0_solved -> prior_exposure_overlap ->
    duplicate_within_new_manifest -> accepted) until 24
    accepted; no borrowing across levels;
  3 hash the raw attempt stream, then derive the manifest (96
    rows: index, level, seed, canonical root cur, full sha,
    depth0_is_solved=false, exclusion_pass=true) and the
    qualification + overlap receipts. No outcome, score, hce,
    trajectory, or controller field exists anywhere.

Outputs under logs/mathworld1/cl1/pop/ (refuse-if-exists):
exclusion_universe.json, raw_attempts.jsonl, manifest.jsonl,
receipt.json.

    .venv/bin/python scratch/mathworld1_cl1pop.py       (Mac)
"""
import hashlib
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

PREREG_COMMIT = "082d4bc6ed32207a0d0b1fd5d85a56af0a9e4caf"
BAND_START = 500000
PER_LEVEL = 24
LEVELS = [4, 5, 6, 7]
MAT_TIMEOUT_S = 30.0
SOURCES = {
    "data/matsub_paired.jsonl":
        ("a943ba7fc581db743b07192e5d951fadddd2ba19bca3225b75"
         "d8402351d468e8"),
    "logs/mathworld1/svpdiet3/covered_calibration.jsonl":
        ("af1a4aa1df7bf3224745e91a90e1a77c36e5c54f7ff9b08509"
         "794d0fb7978db3"),
    "logs/mathworld1/svpdiet3/heldout_test16.jsonl":
        ("a3f6103b3733d909281849dcb3fd6ba9fba3891f2014bec138"
         "81b4509df46ddb"),
}
EVAL_DIRS = {"svpeval": "logs/mathworld1/svpeval",
             "svpeval2": "logs/mathworld1/svpeval2",
             "svpeval3": "logs/mathworld1/svpeval3"}
CHAL = {"svpchal_blocks": "logs/mathworld1/svpchal/blocks.jsonl",
        "svpchal2_blocks":
            "logs/mathworld1/svpchal2/blocks.jsonl",
        "svpchal2_decisions":
            "logs/mathworld1/svpchal2/decisions.jsonl"}
OUTDIR = Path("logs/mathworld1/cl1/pop")
EXCL = OUTDIR / "exclusion_universe.json"
RAWA = OUTDIR / "raw_attempts.jsonl"
MANIFEST = OUTDIR / "manifest.jsonl"
RECEIPT = OUTDIR / "receipt.json"
SRC = ["scratch/mathworld1_cl1pop.py",
       "llmopt/mathgen/problems.py",
       "llmopt/search/derivation.py",
       "llmopt/lab/provenance.py"]


def gate(cond, msg):
    if not cond:
        raise SystemExit(f"GATE FAILED: {msg}")


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ssha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _worker(level, seed, q):
    """Fork target: materialize root, canonical cur, depth-0
    is_solved. Structural only — no controller, no scoring."""
    import sympy as sp
    from llmopt.mathgen.problems import make_integrate
    from llmopt.search.derivation import State, is_solved
    X = sp.Symbol("x")
    p = make_integrate(level, seed)
    root = sp.Integral(p._expr, X)
    q.put({"cur": sp.sstr(root),
           "depth0_solved": bool(is_solved(State(root)))})


def materialize(level, seed):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(level, seed, q))
    proc.start()
    proc.join(MAT_TIMEOUT_S)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return None
    if proc.exitcode != 0:
        return None
    try:
        return q.get_nowait()
    except Exception:
        return None


def build_exclusion():
    cats = {}
    # A: training parents
    for p, h in SOURCES.items():
        gate(fsha(p) == h, f"PIN {p}")
    curs = set()
    raw = 0
    for line in open("data/matsub_paired.jsonl"):
        raw += 1
        curs.add(json.loads(line)["cur"])
    gate(raw == 73324, "TRAIN ROWS")
    gate(len(curs) == 58988, f"TRAIN PARENTS {len(curs)}")
    cats["A_training_parents"] = (curs, raw)
    # B: eval roots (re-materialized) + decision parents
    ev = set()
    raw_b = 0
    n_roots = 0
    for name, d in EVAL_DIRS.items():
        for line in open(Path(d) / "episodes.jsonl"):
            r = json.loads(line)
            res = materialize(r["level"], r["seed"])
            gate(res is not None,
                 f"EVAL ROOT REBUILD {name} {r['seed']}")
            ev.add(res["cur"])
            n_roots += 1
            raw_b += 1
        for line in open(Path(d) / "decisions.jsonl"):
            r = json.loads(line)
            if "cur" in r:
                ev.add(r["cur"])
                raw_b += 1
    gate(n_roots == 240, f"EVAL ROOTS {n_roots}")
    cats["B_eval_roots_and_parents"] = (ev, raw_b)
    # C: SVP state blocks + challenge material
    sb = set()
    raw_c = 0
    for p in ("logs/mathworld1/svpdiet3/"
              "covered_calibration.jsonl",
              "logs/mathworld1/svpdiet3/heldout_test16.jsonl"):
        for line in open(p):
            sb.add(json.loads(line)["cur"])
            raw_c += 1
    for name, p in CHAL.items():
        if not Path(p).exists():
            continue
        for line in open(p):
            r = json.loads(line)
            for k in ("cur", "target_integrand"):
                if k in r and isinstance(r[k], str):
                    sb.add(r[k])
                    raw_c += 1
    cats["C_state_blocks_and_challenge"] = (sb, raw_c)
    # D: prior closed-loop populations — none may exist
    gate(not any(Path("logs/mathworld1/cl1").glob(
        "**/manifest.jsonl")), "PRIOR CL POPULATION EXISTS")
    cats["D_prior_closed_loop"] = (set(), 0)

    union = set()
    for s, _ in cats.values():
        union |= s
    names = list(cats)
    inter = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter[f"{names[i]}&{names[j]}"] = len(
                cats[names[i]][0] & cats[names[j]][0])
    receipt = {
        "sources": {p: fsha(p) for p in SOURCES},
        "eval_dirs": {n: {"episodes": fsha(
            Path(d) / "episodes.jsonl"), "decisions": fsha(
            Path(d) / "decisions.jsonl")}
            for n, d in EVAL_DIRS.items()},
        "challenge": {n: fsha(p) for n, p in CHAL.items()
                      if Path(p).exists()},
        "categories": {n: {"raw_rows": raw,
                           "distinct_states": len(s)}
                       for n, (s, raw) in cats.items()},
        "union_cardinality": len(union),
        "pairwise_intersections": inter}
    EXCL.write_text(json.dumps(receipt, indent=1))
    return union, cats, receipt


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    OUTDIR.mkdir(parents=True)
    START = start_provenance(SRC)
    t0 = time.time()
    union, cats, excl_receipt = build_exclusion()
    excl_sha = fsha(EXCL)
    print(f"[cl1pop] exclusion universe {len(union)} states "
          f"({time.time() - t0:.0f}s)", flush=True)

    stats = {lv: Counter() for lv in LEVELS}
    scan = {lv: [] for lv in LEVELS}
    accepted = {lv: [] for lv in LEVELS}
    seen_new = set()
    with open(RAWA, "w") as fo:
        for lv in LEVELS:
            seed = BAND_START
            while len(accepted[lv]) < PER_LEVEL:
                res = materialize(lv, seed)
                row = {"level": lv, "seed": seed}
                if res is None:
                    row.update({
                        "status": "materialization_failure",
                        "accepted": False,
                        "reason": "materialization_failure"})
                    stats[lv]["materialization_failure"] += 1
                else:
                    cur = res["cur"]
                    sha = ssha(cur)
                    row.update({"status": "materialized",
                                "cur": cur, "root_sha": sha,
                                "depth0_is_solved":
                                    res["depth0_solved"]})
                    if res["depth0_solved"]:
                        row.update({"accepted": False,
                                    "reason": "depth0_solved"})
                        stats[lv]["depth0_solved"] += 1
                    elif cur in union:
                        src = [n for n, (s, _) in cats.items()
                               if cur in s]
                        row.update({
                            "accepted": False,
                            "reason": "prior_exposure_overlap",
                            "exclusion_categories": src})
                        stats[lv]["prior_exposure_overlap"] \
                            += 1
                    elif cur in seen_new:
                        row.update({
                            "accepted": False,
                            "reason":
                                "duplicate_within_new_manifest"
                        })
                        stats[lv][
                            "duplicate_within_new_manifest"] \
                            += 1
                    else:
                        row.update({"accepted": True,
                                    "reason": "accepted"})
                        stats[lv]["accepted"] += 1
                        seen_new.add(cur)
                        accepted[lv].append(
                            {"level": lv, "seed": seed,
                             "cur": cur, "root_sha": sha})
                fo.write(json.dumps(row) + "\n")
                scan[lv].append(seed)
                seed += 1
            print(f"[cl1pop] L{lv}: 24 accepted, "
                  f"seeds {scan[lv][0]}..{scan[lv][-1]} "
                  f"({dict(stats[lv])})", flush=True)
    raw_sha = fsha(RAWA)

    rows = []
    idx = 0
    for lv in LEVELS:
        gate(len(accepted[lv]) == PER_LEVEL, f"BALANCE L{lv}")
        for a in accepted[lv]:
            rows.append({
                "population": "CLOSED-LOOP-1",
                "prereg_commit": PREREG_COMMIT,
                "row_index": idx, "level": a["level"],
                "generator_seed": a["seed"],
                "root_cur": a["cur"],
                "root_sha": a["root_sha"],
                "depth0_is_solved": False,
                "exclusion_pass": True,
                "qualification": "accepted"})
            idx += 1
    gate(len(rows) == 96, "N=96")
    gate(len({r["root_sha"] for r in rows}) == 96,
         "UNIQUE SHAS")
    with open(MANIFEST, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    man_sha = fsha(MANIFEST)

    # overlap receipt: independent recompute per category
    overlap = {}
    for n, (s, _) in cats.items():
        overlap[n] = sum(1 for r in rows if r["root_cur"] in s)
        gate(overlap[n] == 0, f"OVERLAP {n}")
    receipt = {
        "prereg": "MATH-CYBER-1-CLOSED-LOOP-1-PREREG-0",
        "prereg_commit": PREREG_COMMIT,
        "verdict": "POPULATION MATERIALIZED + QUALIFIED",
        "manifest_sha": man_sha,
        "raw_attempts_sha": raw_sha,
        "exclusion_universe_sha": excl_sha,
        "exclusion_summary": {
            "union": excl_receipt["union_cardinality"],
            "categories": excl_receipt["categories"]},
        "per_level": {str(lv): {
            "attempts": len(scan[lv]),
            "first_seed": scan[lv][0],
            "last_seed": scan[lv][-1],
            **{k: v for k, v in sorted(stats[lv].items())}}
            for lv in LEVELS},
        "totals": {
            "attempts": sum(len(scan[lv]) for lv in LEVELS),
            "accepted": 96,
            **{k: sum(stats[lv][k] for lv in LEVELS)
               for k in sorted({k for lv in LEVELS
                                for k in stats[lv]})}},
        "overlap_receipt": {**{n: f"0/96" for n in overlap},
                            "duplicate_shas_within_96": 0,
                            "depth0_solved_within_96": 0},
        "sealed_unscored": ("POPULATION SEALED, UNSCORED, "
                            "NO EPISODES RUN — no checkpoint "
                            "loaded, no model forward, no "
                            "controller policy executed"),
        "mat_timeout_s": MAT_TIMEOUT_S,
        "wall_s": round(time.time() - t0, 1),
        "start": START,
        "completion_commit": completion_commit()}
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in
                      ("verdict", "manifest_sha", "per_level",
                       "totals", "overlap_receipt")},
                     indent=1), flush=True)
    print("[cl1pop] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
