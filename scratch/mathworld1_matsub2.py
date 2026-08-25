"""MATH-CYBER-1 MATCHED-SUBSET-MATERIALIZE-0-CENSOR — amended
re-materialization under the general engine-irreproducible-emission
censoring law and a DETERMINISTIC hash-seed ladder.

Ontology (frozen in the amendment):
    historical PDC-matched population = all 73,324 source rows
    paired-eligible = FINAL canonical ActionProgram reconstructed
        and exactly replayed by the frozen current engine under
        the deterministic reproducibility law below
    engine_irreproducible_emission = historical row whose exact
        transition cannot be exposed/replayed because the required
        current legal edge is absent/dedup-suppressed in EVERY
        registered environment, with NO alternative canonical
        program reproducing the same nxt, and cur/nxt intact.
        Mathematical equivalence / alternate spelling NEVER
        upgrades to eligibility. All other error classes
        (collision, ambiguous decode, tokenizer failure, wrong
        child, malformed program, source/hash mismatch) remain
        BLOCKERS and cannot be censored.

Deterministic reproducibility law: the main pass runs under
PYTHONHASHSEED=0 (asserted); failures walk the registered ladder
PYTHONHASHSEED = 1..15 in order (subprocess per seed, first
success wins, finite stop). The prior run's 0..15 diagnostic is
DISCLOSED PRIOR INFORMATION motivating this ladder. No OS-random
interpreter seed influences the frozen artifact.

Outputs (refuse-if-exists):
- data/matsub_master.jsonl — ALL 73,324 historical rows in source
  order: row_id, source, cur, state_target, paired_eligible,
  exclusion_class/reason, successful_hash_seed/attempt,
  ActionProgram fields + program_text where eligible, overlap
  flags.
- data/matsub_paired.jsonl — the neutral paired-training view,
  paired_eligible rows only (symmetric for both future arms).
- logs/mathworld1/matsub2_receipt.json.

Reuses the frozen MSM-0 driver's derive_programs/--pair verbatim
via import/subprocess (scratch/mathworld1_matsub.py stays frozen
as booked evidence).

    PYTHONHASHSEED=0 .venv/bin/python scratch/mathworld1_matsub2.py
"""
import json
import multiprocessing as mp
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from scratch.mathworld1_matsub import (MANIFEST, PINS,  # noqa: E402
                                       TOK, derive_programs, fsha,
                                       load_population,
                                       overlap_sets, sha)

LADDER = list(range(1, 16))     # main pass = seed 0, then 1..15
MASTER = Path("data/matsub_master.jsonl")
PAIRED = Path("data/matsub_paired.jsonl")
RECEIPT = Path("logs/mathworld1/matsub2_receipt.json")
MATSUB = "scratch/mathworld1_matsub.py"
# error classes that MAY feed the censoring evaluation (edge
# absence); anything else observed in ANY environment = blocker
ABSENCE = {"rule_absent", "unaddressable"}


def pair_at_seed(cur, want, seed):
    p = subprocess.run(
        [sys.executable, MATSUB, "--pair"],
        input=json.dumps([cur, want]),
        capture_output=True, text=True,
        env={**os.environ, "PYTHONHASHSEED": str(seed)})
    if p.returncode != 0:
        return None
    return json.loads(p.stdout)


def alt_edge_exists(cur, nxt, seed):
    """ANY current legal edge (any rule) whose child prints nxt,
    in a fresh seed environment."""
    code = (
        "import sys, json; sys.path.insert(0, '.')\n"
        "import sympy as sp\n"
        "from llmopt.search.derivation import State, successors\n"
        "cur, nxt = json.loads(sys.stdin.read())\n"
        "p = sp.sympify(cur)\n"
        "hit = [n for n, ch in successors(State(p))"
        " if sp.sstr(ch.expr) == nxt]\n"
        "print(json.dumps(hit))\n")
    p = subprocess.run(
        [sys.executable, "-c", code],
        input=json.dumps([cur, nxt]),
        capture_output=True, text=True,
        env={**os.environ, "PYTHONHASHSEED": str(seed)})
    if p.returncode != 0:
        return ["__probe_error__"]
    return json.loads(p.stdout)


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("LAUNCH WITH PYTHONHASHSEED=0")
    for p in (MASTER, PAIRED, RECEIPT):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    START = start_provenance(
        ["scratch/mathworld1_matsub2.py",
         "scratch/mathworld1_matsub.py",
         "scratch/mathworld1_axfixture.py",
         "scratch/mathworld1_actionfinal.py",
         "scratch/mathworld1_unprodsem.py",
         "scratch/mathworld1_actionsem.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_srepr_export.py",
         "scratch/mathworld1_birth.py",
         "llmopt/lab/provenance.py",
         "llmopt/search/derivation.py", "llmopt/search/rules.py"])
    man = json.load(open(MANIFEST))
    dm = man["diet_manifest"]
    files = list(dm)
    for f in files:
        if fsha(f) != dm[f]["sha256"]:
            raise SystemExit(f"MANIFEST SHA MISMATCH {f}")
    pins = {p: fsha(p) for p in PINS}

    up = load_population()
    src_rows = []
    total_scanned = 0
    for f in files:
        for i, l in enumerate(open(f)):
            total_scanned += 1
            r = json.loads(l)
            cur, nxt = r["cur"], r["nxt"]
            h = sha(cur)
            if h in up and nxt in up[h]:
                meta = {k: r[k] for k in ("source", "hints", "think")
                        if k in r}
                src_rows.append(
                    (f, i, r.get("level"), meta, cur, nxt))
    if total_scanned != man["rows"]:
        raise SystemExit(
            f"ROW COUNT {total_scanned} != {man['rows']}")
    if len(src_rows) != 73324:
        raise SystemExit(f"POPULATION {len(src_rows)} != 73324")
    need = defaultdict(dict)
    for (_, _, _, _, cur, nxt) in src_rows:
        need[cur][nxt] = up[sha(cur)][nxt]
    items = sorted(need.items())
    print(f"[msm2] rows={len(src_rows)} parents={len(items)} "
          f"seed=0 main pass", flush=True)

    ctx = mp.get_context("fork")
    programs = {}
    with ctx.Pool(9) as pool:
        for (cur, _), res in zip(
                items, pool.imap(derive_programs, items,
                                 chunksize=16)):
            for nxt, row in res.items():
                if "err" not in row:
                    row["successful_hash_seed"] = 0
                    row["attempt"] = 0
                programs[(cur, nxt)] = row
    fail0 = {cur for (cur, nxt), r in programs.items()
             if "err" in r}
    print(f"[msm2] seed-0 failures parents={len(fail0)} classes="
          f"{dict(Counter(r['err'] for r in programs.values() if 'err' in r))}",
          flush=True)

    # deterministic ladder, first success
    seed_errs = defaultdict(dict)   # cur -> seed -> err classes
    for cur in sorted(fail0):
        for (c2, n2), r2 in programs.items():
            if c2 == cur and "err" in r2:
                seed_errs[cur][0] = seed_errs[cur].get(0, set()) \
                    | {r2["err"]}
        for k, seed in enumerate(LADDER, start=1):
            res = pair_at_seed(cur, need[cur], seed)
            if res is None:
                seed_errs[cur][seed] = {"__subprocess_error__"}
                continue
            errs_here = {r["err"] for r in res.values()
                         if "err" in r}
            if not errs_here:
                for nxt, row in res.items():
                    row["successful_hash_seed"] = seed
                    row["attempt"] = k
                    programs[(cur, nxt)] = row
                break
            seed_errs[cur][seed] = errs_here
        else:
            pass  # exhausted; censoring evaluation below

    # censoring evaluation for exhausted rows
    exclusions = []          # (cur, nxt, class, reason)
    blockers = []
    for (cur, nxt), r in sorted(programs.items()):
        if "err" not in r:
            continue
        classes = set()
        for s, es in seed_errs.get(cur, {}).items():
            classes |= es
        classes.add(r["err"])
        if not classes <= ABSENCE:
            blockers.append((cur, nxt, sorted(classes)))
            continue
        # condition 4: no alternative canonical program (any rule)
        alts = set()
        for seed in [0] + LADDER:
            alts |= set(alt_edge_exists(cur, nxt, seed))
        if alts:
            blockers.append(
                (cur, nxt, ["alternative_edge_exists"]
                 + sorted(alts)[:3]))
            continue
        reason = (
            f"required {need[cur][nxt]} edge absent in all 16 "
            f"registered environments (classes {sorted(classes)}); "
            f"no edge of any rule prints the historical nxt; "
            f"cur/nxt intact per source-scan hash pin")
        exclusions.append((cur, nxt,
                           "engine_irreproducible_emission", reason))
    if blockers:
        print(f"[msm2] BLOCKERS {len(blockers)}: "
              f"{blockers[:5]}", flush=True)

    corpus_states, corpus_edges, bands = overlap_sets()
    excl_of = {(c, n): (cl, why) for (c, n, cl, why) in exclusions}
    blk_of = {(c, n): cls for (c, n, cls) in blockers}
    per_src_all = Counter()
    per_src_excl = Counter()
    per_rule = Counter()
    per_pk = Counter()
    ov = Counter()
    st_lens, pg_lens = [], []
    collide = {}
    n_lt = 0
    n_elig = 0
    tok_fail = 0
    with open(MASTER, "w") as fm, open(PAIRED, "w") as fp:
        for (f, i, level, meta, cur, nxt) in src_rows:
            per_src_all[f] += 1
            pr = programs[(cur, nxt)]
            base = {
                "row_id": sha(f"{f}:{i}:{cur}:{nxt}"),
                "source_file": f, "source_line": i,
                "level": level, **({"meta": meta} if meta else {}),
                "cur": cur, "state_target": nxt}
            if (cur, nxt) in excl_of:
                cl, why = excl_of[(cur, nxt)]
                fm.write(json.dumps({
                    **base, "paired_eligible": False,
                    "exclusion_class": cl,
                    "exclusion_reason": why}) + "\n")
                per_src_excl[f] += 1
                continue
            if (cur, nxt) in blk_of:
                fm.write(json.dumps({
                    **base, "paired_eligible": False,
                    "exclusion_class": "BLOCKER",
                    "exclusion_reason": str(blk_of[(cur, nxt)])})
                    + "\n")
                continue
            if "<" in cur or "<" in nxt:
                n_lt += 1
            if TOK.decode(TOK.encode(pr["program_text"])) \
                    != pr["program_text"]:
                tok_fail += 1
            ckey = (cur, pr["program_text"])
            if ckey in collide and collide[ckey] != nxt:
                blockers.append((cur, nxt, ["collision"]))
                continue
            collide[ckey] = nxt
            flags = {
                "overlap_corpus_state": cur in corpus_states,
                "overlap_corpus_edge": (cur, nxt) in corpus_edges}
            for b, hs in bands.items():
                flags[f"overlap_band_{b}"] = pr["band_hash"] in hs
            row = {
                **base, "paired_eligible": True,
                "successful_hash_seed": pr["successful_hash_seed"],
                "attempt": pr["attempt"],
                "rule": pr["rule"], "site_kind": pr["site_kind"],
                "site_ordinal": pr["site_ordinal"],
                "param_kind": pr["param_kind"],
                "param_index": pr["param_index"],
                "program_text": pr["program_text"], **flags}
            fm.write(json.dumps(row) + "\n")
            fp.write(json.dumps(row) + "\n")
            n_elig += 1
            per_rule[pr["rule"]] += 1
            per_pk[pr["param_kind"]] += 1
            for k, v in flags.items():
                if v:
                    ov[k] += 1
            st_lens.append(len(TOK.encode(nxt)))
            pg_lens.append(len(TOK.encode(pr["program_text"])))

    def dist(xs):
        xs = sorted(xs)
        q = lambda p: xs[min(len(xs) - 1,
                             int(p * (len(xs) - 1)))] if xs else None
        return {"p50": q(.5), "p90": q(.9), "p99": q(.99),
                "max": xs[-1] if xs else None}

    n_excl = len(exclusions)
    receipt = {
        "historical_rows": len(src_rows),
        "paired_eligible": n_elig,
        "excluded_engine_irreproducible": n_excl,
        "blocker_rows": len(src_rows) - n_elig - n_excl,
        "exclusion_rows": [
            {"cur_sha": sha(c), "nxt": n, "class": cl,
             "reason": why} for (c, n, cl, why) in exclusions],
        "blocker_detail": [
            {"cur_sha": sha(c), "nxt": n, "classes": cls}
            for (c, n, cls) in blockers][:20],
        "retry_by_rule": dict(Counter(
            programs[(c, n)].get("rule", "?")
            for (c, n), r in programs.items()
            if "err" not in r and r.get("attempt", 0) > 0)),
        "retry_seed_hist": dict(Counter(
            r["successful_hash_seed"]
            for r in programs.values()
            if "err" not in r and r.get("attempt", 0) > 0)),
        "per_source_all": dict(per_src_all),
        "per_source_excluded": dict(per_src_excl),
        "per_rule": dict(per_rule),
        "param_kind": dict(per_pk),
        "overlap_counts": dict(ov),
        "token_dist": {"state_view": dist(st_lens),
                       "program_view": dist(pg_lens),
                       "note": "descriptive only"},
        "literal_lt_rows": n_lt,
        "tok_roundtrip_fail": tok_fail,
        "ladder": [0] + LADDER,
        "artifacts": {
            "master": {"path": str(MASTER), "sha256": fsha(MASTER),
                       "bytes": MASTER.stat().st_size},
            "paired": {"path": str(PAIRED), "sha256": fsha(PAIRED),
                       "bytes": PAIRED.stat().st_size}},
        "pins": pins, "manifest_rows": man["rows"],
        "bars": {
            "MASTER_POPULATION": per_src_all.total() == 73324,
            "ACCOUNTING": n_elig + n_excl == 73324
                and len(src_rows) - n_elig - n_excl == 0,
            "FINAL_REPLAY": all(
                "err" not in programs[(c, n)]
                for (_, _, _, _, c, n) in src_rows
                if (c, n) not in excl_of),
            "EXCLUSION_SOUNDNESS": all(
                cl == "engine_irreproducible_emission"
                for (_, _, cl, _) in exclusions)
                and not blockers,
            "TOK_ROUNDTRIP": tok_fail == 0 and n_lt == 0,
            "NO_COLLISION": not any(
                cls == ["collision"] for (_, _, cls) in blockers),
            # PDC multiplicity accounting: per-source historical
            # counts equal PDC's booked unique_program dict; the
            # eligible view differs only by the named exclusions.
            "PDC_MULTIPLICITY": dict(per_src_all) == {
                f: v.get("unique_program", 0) for f, v in
                json.load(open("logs/mathworld1/pdc_verdict.json")
                          )["by_source_file"].items()},
            # derived (same law as ACTION-FINAL/AX-FIXTURE): no
            # child-key-sorted path generates any semantic field;
            # holds iff replay + exclusion-soundness bars hold.
            "NO_OPAQUE": all(
                "err" not in programs[(c, n)]
                for (_, _, _, _, c, n) in src_rows
                if (c, n) not in excl_of)
                and not blockers,
        },
        "start": START, "completion_commit": completion_commit()}
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ("start", "pins",
                                   "per_source_all")},
                     indent=1)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
