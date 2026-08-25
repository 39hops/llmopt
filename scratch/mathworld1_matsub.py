"""MATH-CYBER-1 MATCHED-SUBSET-MATERIALIZE-0 — the ONE neutral
paired artifact for the PDC matched population.

Emits data/matsub_pairs.jsonl (UNTRACKED, file-handoff): every
PDC unique_program historical row (expected 73,324 across the 13
theta0 birth-manifest files, source multiplicity preserved, rows
in historical source order) carrying cur, state_target =
historical nxt, the FINAL canonical ActionProgram structured
fields, program_text (ActionGCTok vocab 332), and overlap
annotation flags. Receipt logs/mathworld1/matsub_receipt.json.

Every emitted row reconstructs its program by the ACTION-FINAL
decode laws (u_choice map / parity-gated term map / deterministic
single-accepted-child; first-preorder site, first matching site)
in the visible-string domain and must replay
decode(cur, program) == nxt exactly. Failed parents get ONE cold
retry pass; survivors are blockers, never dropped.

Representation-neutral: no arm files, no target-length sorting,
no batching/shuffle/splits, no model anywhere.

SMOKE: MATSUB_SMOKE=1 processes the first 40 matched parents and
writes only *_smoke paths.

    .venv/bin/python scratch/mathworld1_matsub.py             (Mac)
"""
import hashlib
import json
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

import llmopt.search.derivation as derivation  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.search.derivation import State, successors  # noqa: E402
from scratch.mathworld1_actionsem import (RULE_KIND,  # noqa: E402
                                          apply_at, iparts_children,
                                          sites_preorder)
from scratch.mathworld1_actionfinal import (  # noqa: E402
    unprod_term_children)
from scratch.mathworld1_actiontok import ActionGCTok  # noqa: E402
from scratch.mathworld1_axfixture import serialize  # noqa: E402
from scratch.mathworld1_srepr_export import srepr_inverse  # noqa: E402

SMOKE = os.environ.get("MATSUB_SMOKE") == "1"
SUF = "_smoke" if SMOKE else ""
MANIFEST = "checkpoints/mathnative_19m_mw1_theta0.json"
ARTIFACT = Path(f"data/matsub_pairs{SUF}.jsonl")
RECEIPT = Path(f"logs/mathworld1/matsub_receipt{SUF}.json")
PINS = ["logs/mathworld1/pdc_relabel.jsonl",
        "logs/mathworld1/pdc_verdict.json",
        "logs/mathworld1/actionfinal_qual.json",
        "logs/mathworld1/axfixture/axfixture_manifest.json"]

TOK = ActionGCTok()


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def vis(key: str) -> str:
    return sp.sstr(srepr_inverse(key))


def derive_programs(item):
    """One parent: (cur, {nxt: rule}) -> per-nxt canonical program
    + replay verdict. Returns dict nxt -> row-or-error."""
    cur, want = item
    out = {}
    try:
        parent = sp.sympify(cur)
        if sp.sstr(parent) != cur:
            return {n: {"err": "parent_nonroundtrip"} for n in want}
        derivation._RULE_CACHE.clear()
        gen = list(successors(State(parent)))
        accepted = defaultdict(set)
        for name, ch in gen:
            rule = name.split("@", 1)[0] if "@" in name else name
            accepted[rule].add(ch.key())
        vcache = {}

        def kv(k):
            if k not in vcache:
                vcache[k] = vis(k)
            return vcache[k]

        bandh = sha(sp.srepr(parent))
        for nxt, rule in want.items():
            kind = RULE_KIND.get(rule)
            if rule not in accepted:
                out[nxt] = {"err": "rule_absent"}
                continue
            acc_vis = {k: kv(k) for k in accepted[rule]}
            if kind is None:
                site, node = -1, None
            else:
                hits = []
                for i, cand in enumerate(
                        sites_preorder(parent, kind)):
                    ck, _ = apply_at(parent, rule, cand)
                    if any(acc_vis.get(k) == nxt
                           for k in set(ck) & accepted[rule]):
                        hits.append((i, cand))
                        break
                if not hits:
                    out[nxt] = {"err": "unaddressable"}
                    continue
                site, node = hits[0]
            if rule == "i_parts":
                uc_map, _ = iparts_children(parent, node)
                m = [u for u, k in uc_map.items() if kv(k) == nxt]
                if len(m) != 1:
                    out[nxt] = {"err": "u_ambiguous"}
                    continue
                pkind, pindex = "u_choice", m[0]
                dk = uc_map.get(m[0])
                dec = {kv(dk)} if dk else set()
            elif rule == "i_unprod":
                tmap, parity = unprod_term_children(parent, node)
                if not parity:
                    out[nxt] = {"err": "parity_fail"}
                    continue
                m = [t for t, ks in tmap.items()
                     if any(kv(k) == nxt for k in ks)]
                if len(m) != 1:
                    out[nxt] = {"err": "term_ambiguous"}
                    continue
                pkind, pindex = "term_index", m[0]
                dec = {kv(k) for k in tmap[m[0]]}
            else:
                if kind is None:
                    dset = accepted[rule]
                else:
                    ck, _ = apply_at(parent, rule, node)
                    dset = set(ck) & accepted[rule]
                if len(dset) != 1:
                    out[nxt] = {"err": "det_ambiguous"}
                    continue
                pkind, pindex = "none", -1
                dec = {kv(next(iter(dset)))}
            if dec != {nxt}:
                out[nxt] = {"err": "replay_mismatch"}
                continue
            text = serialize(rule, kind, site, pkind, pindex)
            if TOK.decode(TOK.encode(text)) != text:
                out[nxt] = {"err": "tok_roundtrip"}
                continue
            out[nxt] = {
                "rule": rule, "site_kind": kind if kind else "W",
                "site_ordinal": site, "param_kind": pkind,
                "param_index": pindex, "program_text": text,
                "band_hash": bandh}
    except Exception as e:  # timebox/oracle failures book, never drop
        for n in want:
            out.setdefault(n, {"err": f"exception:{type(e).__name__}"})
    return out


def load_population():
    up = defaultdict(dict)
    for l in open("logs/mathworld1/pdc_relabel.jsonl"):
        r = json.loads(l)
        for nxt, cl in r.get("rows", {}).items():
            if isinstance(cl, dict) \
                    and cl.get("class") == "unique_program":
                up[r["cur_sha"]][nxt] = cl["rule"]
    return up


def overlap_sets():
    corpus_states = set()
    for l in open("logs/mathworld1/states.jsonl"):
        corpus_states.add(json.loads(l)["state_before"])
    pvis = {}
    for l in open("logs/mathworld1/axfixture/parents.jsonl"):
        r = json.loads(l)
        pvis[r["parent_id"]] = r["state_visible_v1"]
    corpus_edges = set()
    for l in open("logs/mathworld1/axfixture/actions.jsonl"):
        r = json.loads(l)
        corpus_edges.add((pvis[r["parent_id"]], r["child_sstr"]))
    bands = defaultdict(set)
    for l in open("logs/mathworld1/terminal_census.jsonl"):
        r = json.loads(l)
        if r.get("row") == "state" and "state_hash" in r:
            bands[r["band"]].add(r["state_hash"])
    for l in open("logs/mathworld1/active_pair.jsonl"):
        r = json.loads(l)
        if "state_hash" in r and "stage" in r:
            bands[r["stage"]].add(r["state_hash"])
    return corpus_states, corpus_edges, dict(bands)


def main():
    for p in (ARTIFACT, RECEIPT):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    START = start_provenance(
        ["scratch/mathworld1_matsub.py",
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
    booked = json.load(
        open("logs/mathworld1/pdc_verdict.json"))
    exp_by_src = {f: v.get("unique_program", 0) for f, v in
                  booked["by_source_file"].items()}

    # source scan: matched rows in historical order
    src_rows = []          # (source_file, line, level, meta, cur, nxt)
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
    need = defaultdict(dict)
    for (_, _, _, _, cur, nxt) in src_rows:
        need[cur][nxt] = up[sha(cur)][nxt]
    items = sorted(need.items())
    if SMOKE:
        keep = {c for c, _ in items[:40]}
        items = [it for it in items if it[0] in keep]
        src_rows = [r for r in src_rows if r[4] in keep]
    print(f"[msm] matched rows={len(src_rows)} parents={len(items)}",
          flush=True)

    ctx = mp.get_context("fork")
    programs = {}
    errs = Counter()
    with ctx.Pool(9) as pool:
        for (cur, _), res in zip(
                items, pool.imap(derive_programs, items,
                                 chunksize=16)):
            for nxt, row in res.items():
                programs[(cur, nxt)] = row
                if "err" in row:
                    errs[row["err"]] += 1
    # Registered retry law (AMENDMENT -RETRY): failed parents get
    # up to 8 FRESH-SUBPROCESS attempts — a new interpreter per
    # attempt, so each samples a new hash seed (fork workers
    # inherit the launcher's; i_heurisch emission is
    # hash-seed-nondeterministic between equal spellings).
    import subprocess
    retry_hist = Counter()
    retry = sorted({cur for (cur, nxt), r in programs.items()
                    if "err" in r})
    print(f"[msm] retry parents={len(retry)} errs={dict(errs)}",
          flush=True)
    for cur in retry:
        for attempt in range(1, 9):
            p = subprocess.run(
                [sys.executable, __file__, "--pair"],
                input=json.dumps([cur, need[cur]]),
                capture_output=True, text=True)
            if p.returncode != 0:
                continue
            res = json.loads(p.stdout)
            if all("err" not in r for r in res.values()):
                for nxt, row in res.items():
                    programs[(cur, nxt)] = row
                retry_hist[attempt] += 1
                break
        else:
            retry_hist["exhausted"] += 1
    errs = Counter(r["err"] for r in programs.values()
                   if "err" in r)
    print(f"[msm] post-retry errs={dict(errs)} "
          f"hist={dict(retry_hist)}", flush=True)

    corpus_states, corpus_edges, bands = overlap_sets()
    per_src = Counter()
    per_rule = Counter()
    per_pk = Counter()
    ov = Counter()
    st_lens, pg_lens = [], []
    collide = {}
    n_lt = 0
    emitted = 0
    with open(ARTIFACT, "w") as fo:
        for (f, i, level, meta, cur, nxt) in src_rows:
            pr = programs[(cur, nxt)]
            if "err" in pr:
                continue  # bar 2 adjudicates; emission stays honest
            ckey = (cur, pr["program_text"])
            if ckey in collide and collide[ckey] != nxt:
                errs["collision"] += 1
                continue
            collide[ckey] = nxt
            if "<" in cur or "<" in nxt:
                n_lt += 1
            flags = {
                "overlap_corpus_state": cur in corpus_states,
                "overlap_corpus_edge": (cur, nxt) in corpus_edges}
            for b, hs in bands.items():
                flags[f"overlap_band_{b}"] = pr["band_hash"] in hs
            row = {
                "row_id": sha(f"{f}:{i}:{cur}:{nxt}"),
                "source_file": f, "source_line": i,
                "level": level, **({"meta": meta} if meta else {}),
                "cur": cur, "state_target": nxt,
                "rule": pr["rule"], "site_kind": pr["site_kind"],
                "site_ordinal": pr["site_ordinal"],
                "param_kind": pr["param_kind"],
                "param_index": pr["param_index"],
                "program_text": pr["program_text"], **flags}
            fo.write(json.dumps(row) + "\n")
            emitted += 1
            per_src[f] += 1
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

    uniq_pairs = len({(c, n) for (_, _, _, _, c, n) in src_rows
                      if "err" not in programs[(c, n)]})
    uniq_cur = len({c for (_, _, _, _, c, n) in src_rows
                    if "err" not in programs[(c, n)]})
    exp_rows = 40 if SMOKE else 73324
    receipt = {
        "smoke": SMOKE,
        "rows_emitted": emitted,
        "unique_pairs": uniq_pairs, "unique_cur": uniq_cur,
        "decode_errors": dict(errs),
        "retry_hist": dict(retry_hist),
        "per_source": dict(per_src),
        "per_rule": dict(per_rule),
        "param_kind": dict(per_pk),
        "overlap_counts": dict(ov),
        "train_band_note": "TRAIN band has no state hashes in any "
            "frozen receipt (bins-only census); train overlap NOT "
            "annotatable here",
        "token_dist": {"state_view": dist(st_lens),
                       "program_view": dist(pg_lens),
                       "note": "descriptive only"},
        "literal_lt_rows": n_lt,
        "artifact": {"path": str(ARTIFACT),
                     "sha256": fsha(ARTIFACT),
                     "bytes": ARTIFACT.stat().st_size},
        "pins": pins, "manifest_rows": man["rows"],
        "bars": {} if SMOKE else {
            "POPULATION_EXACT": emitted == 73324
                and dict(per_src) == exp_by_src
                and uniq_pairs == 59587 and uniq_cur == 58988,
            "FINAL_REPLAY": sum(errs.values()) == 0
                and emitted == 73324,
            "TOK_ROUNDTRIP": errs.get("tok_roundtrip", 0) == 0
                and n_lt == 0,
            "NO_COLLISION": errs.get("collision", 0) == 0,
        },
        "start": START, "completion_commit": completion_commit()}
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ("start", "pins", "per_source")},
                     indent=1), flush=True)
    print(f"[msm] expected rows {exp_rows} (smoke informational)",
          flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--pair":
        cur, want = json.loads(sys.stdin.read())
        print(json.dumps(derive_programs((cur, want))))
        sys.exit(0)
    sys.exit(main())
