"""MATH-CYBER-1 SVP-BIJECTION-GEOMETRY-DESK-0 — ONE static
code-geometry census over the frozen world, per the definitions
FROZEN at the desk prereg (commit 178db461) BEFORE any value was
computed. POSTHOC hypothesis generation only. Zero model
inference, zero checkpoints, zero torch, zero new permutations,
zero training, zero fresh population. Arms: FACTOR / HASH-P1
(stored codes) / HASH-P2 (derived from the immutable realization,
re-derivation gated == 952f332d...).

Families (definitions verbatim from the frozen prereg):
  A per-arm autoregressive target support from the 74,860
    training target codewords WITH multiplicity; prefix counts;
    positional/bigram/trigram counts; log P_emp with LAMBDA=1.
  B sibling-relative support per heldout state (target v legal
    rivals; pessimistic static rank; candidate-0 census;
    rank-1 fraction = the ONLY bar-carrying statistic).
  C covered-witness bridge geometry, witnesses selected
    mechanically from semantic tuples: same-term other-site
    (ordinal 0, same t) + same-site covered terms (t0, t1,
    reported separately in that frozen order); Hamming / LCP /
    per-position equality / position-aligned shared bigrams and
    trigrams; the same v legal rivals as density control.
  D local training-code density v the DISTINCT training
    codeword set (min Hamming; radius counts r=1,2,3; max
    common prefix + attaining count), beside rival min/max.
  E Feistel-internal overlap for P1/P2 only (SECONDARY; the
    model never observes internals): (round, input-half) pairs
    traversed by heldout actions v the training-traversed set.

FROZEN BAR (computed mechanically at the end): GEOMETRY-LEAD iff
Family-B target-rank-1 fraction (of 96) differs P1 v P2 by
>= .25 absolute with P1 above P2 AND the same sign in >= 3 of 4
frozen cells; MIXED-GEOMETRY iff >= .25-scale separations exist
with conflicting directions (adjudicated in the booking prose,
not here — this instrument emits the census and the Family-B
bar components only); NO-STATIC-LEAD otherwise. No p-values.

Outputs under logs/mathworld1/svpgeodesk/ (refuse-if-exists):
census.json, svpgeodesk_receipt.json.

    .venv/bin/python scratch/mathworld1_svpgeodesk.py        (Mac)
"""
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpcode import (ORD_MAX,  # noqa: E402
                                        factor_decode,
                                        factor_symbols,
                                        hash_decode,
                                        hash_symbols)
from scratch.mathworld1_actiontok import OPCODE_ORDER  # noqa: E402
from scratch.mathworld1_svpp2qual import (feistel2,  # noqa: E402
                                          hash2_decode,
                                          hash2_symbols)
from scratch.mathworld1_svpcode import (feistel,  # noqa: E402
                                        sym_to_int)

PAIRED = "data/matsub_paired.jsonl"
AUG = "logs/mathworld1/svpdiet/balanced_grid_train.jsonl"
HELD = "logs/mathworld1/svpdiet3/heldout_test16.jsonl"
PINS = {
    PAIRED:
        "a943ba7fc581db743b07192e5d951fadddd2ba19bca3225b75"
        "d8402351d468e8",
    AUG:
        "0ef3d8a880a7e07712d8de757bc1670df12701e487b856"
        "b44c97f8db16cb3759",
    HELD:
        "a3f6103b3733d909281849dcb3fd6ba9fba3891f2014bec13881"
        "b4509df46ddb",
}
P2_REALIZATION_SHA = (
    "952f332da4e25961b2dd52c786902e74ba4b33bbf8413f88496a"
    "0df952450ba9")
LAMBDA = 1.0
ARMS = ["FACTOR", "HASH_P1", "HASH_P2"]
CELLS = [(2, "IN"), (2, "OUT"), (3, "IN"), (3, "OUT")]
OUTDIR = Path("logs/mathworld1/svpgeodesk")


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ctup(c):
    return (c["rule"], c["site_kind"], c["site_ordinal"],
            c["param_kind"], c["param_index"])


def code_of(arm, tup):
    if arm == "FACTOR":
        return factor_symbols(*tup)
    if arm == "HASH_P1":
        return hash_symbols(*tup)
    return hash2_symbols(*tup)


def rederive_p2():
    stream = hashlib.sha256()
    for r in OPCODE_ORDER:
        for sk in ("W", "I"):
            for so in range(-1, ORD_MAX + 1):
                for pk in ("none", "u_choice", "term_index"):
                    for pi in range(-1, ORD_MAX + 1):
                        stream.update(bytes(
                            hash2_symbols(r, sk, so, pk, pi)))
    return stream.hexdigest()


def feistel_trace(tup, which):
    """(round, input-half) pairs traversed under forward encode."""
    v = sym_to_int(factor_symbols(*tup))
    fn = feistel if which == "HASH_P1" else feistel2
    # replicate the forward walk to capture round inputs
    L, R = v >> 12, v & 0xFFF
    pairs = []
    from scratch.mathworld1_svpcode import _round as r1
    from scratch.mathworld1_svpp2qual import _round2 as r2
    rf = r1 if which == "HASH_P1" else r2
    for i in range(4):
        pairs.append((i, R))
        L, R = R, L ^ rf(i, R)
    # sanity: the walk reproduces the shipped feistel
    gate((L << 12) | R == fn(v), "FEISTEL TRACE MISMATCH")
    return pairs


def stat5(vals):
    if not vals:
        return None
    sv = sorted(vals)
    n = len(sv)
    return {"median": sv[n // 2],
            "p10": sv[max(0, int(n * 0.10))],
            "p90": sv[min(n - 1, int(n * 0.90))]}


def pair_geo(a, b):
    ham = sum(1 for x, y in zip(a, b) if x != y)
    lcp = 0
    for x, y in zip(a, b):
        if x != y:
            break
        lcp += 1
    eq = [int(x == y) for x, y in zip(a, b)]
    bi = sum(1 for k in range(7)
             if a[k] == b[k] and a[k + 1] == b[k + 1])
    tri = sum(1 for k in range(6)
              if a[k:k + 3] == b[k:k + 3])
    return {"hamming": ham, "lcp": lcp, "pos_eq": eq,
            "shared_bigrams": bi, "shared_trigrams": tri}


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    import subprocess
    fr = subprocess.run(["git", "rev-parse", "178db461"],
                        capture_output=True, text=True)
    gate(fr.returncode == 0, "FREEZE COMMIT NOT FOUND")
    subj = subprocess.run(
        ["git", "log", "-1", "--format=%s", "178db461"],
        capture_output=True, text=True).stdout
    gate("BIJECTION-GEOMETRY-DESK-0" in subj,
         f"FREEZE COMMIT SUBJECT: {subj!r}")
    freeze_commit = fr.stdout.strip()
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN MISMATCH {p}")
    p2_derived = rederive_p2()
    gate(p2_derived == P2_REALIZATION_SHA, "P2 DRIFT")
    START = start_provenance(
        ["scratch/mathworld1_svpgeodesk.py",
         "scratch/mathworld1_svpp2qual.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_svpbirth.py",
         "llmopt/lab/provenance.py"])
    t0 = time.time()

    # training target actions (with multiplicity)
    train_tups = []
    for src in (PAIRED, AUG):
        for l in open(src):
            r = json.loads(l)
            train_tups.append((r["rule"], r["site_kind"],
                               r["site_ordinal"],
                               r["param_kind"],
                               r["param_index"]))
    gate(len(train_tups) == 74860, "TRAIN ROWS")

    # per-arm training codebooks
    train_codes = {a: [code_of(a, t) for t in train_tups]
                   for a in ARMS}
    # verify decode roundtrips on a stride sample (cheap)
    for a, dec in (("FACTOR", factor_decode),
                   ("HASH_P1", hash_decode),
                   ("HASH_P2", hash2_decode)):
        for i in range(0, 74860, 9973):
            gate(dec(train_codes[a][i]) == train_tups[i],
                 f"TRAIN RT {a}")

    # Family A count structures per arm
    fam_a = {}
    for a in ARMS:
        pref = Counter()
        pos = [Counter() for _ in range(8)]
        big = Counter()
        tri = Counter()
        for z in train_codes[a]:
            tz = tuple(z)
            for k in range(1, 9):
                pref[tz[:k]] += 1
            for k in range(8):
                pos[k][z[k]] += 1
            for k in range(7):
                big[(k, z[k], z[k + 1])] += 1
            for k in range(6):
                tri[(k, z[k], z[k + 1], z[k + 2])] += 1
        fam_a[a] = {"pref": pref, "pos": pos,
                    "big": big, "tri": tri}
    distinct = {a: set(map(tuple, train_codes[a]))
                for a in ARMS}

    def logp(a, z):
        tz = tuple(z)
        s = 0.0
        prev = 74860.0
        pref = fam_a[a]["pref"]
        for k in range(1, 9):
            c = pref.get(tz[:k], 0)
            s += math.log((c + LAMBDA) / (prev + 8 * LAMBDA))
            prev = float(c)
        return s

    # heldout primary states
    allrows = [json.loads(l) for l in open(HELD)]
    pri = [r for r in allrows
           if r["site_role"] == "heldout-I1"]
    gate(len(pri) == 96, "PRIMARY ROWS")
    for r in pri:
        for c in r["candidates"]:
            t = ctup(c)
            gate(factor_decode(c["factor_code"]) == t, "F RT")
            gate(hash_decode(c["hash_code"]) == t, "P1 RT")
            c["_codes"] = {"FACTOR": c["factor_code"],
                           "HASH_P1": c["hash_code"],
                           "HASH_P2": hash2_symbols(*t)}
            gate(hash2_decode(c["_codes"]["HASH_P2"]) == t,
                 "P2 RT")

    # training-traversed Feistel sets (Family E)
    trav = {}
    for which in ("HASH_P1", "HASH_P2"):
        s = set()
        for t in train_tups:
            s.update(feistel_trace(t, which))
        trav[which] = s

    # per-state census, families B/C/D/E
    states = []
    rank1 = {a: 0 for a in ARMS}
    rank1_cell = {a: {f"t{t}-{rg}": 0 for t, rg in CELLS}
                  for a in ARMS}
    cand0_rank1 = {a: 0 for a in ARMS}
    margins = {a: [] for a in ARMS}
    margins_cell = {a: {f"t{t}-{rg}": [] for t, rg in CELLS}
                    for a in ARMS}
    for r in pri:
        cands = r["candidates"]
        li = [i for i, c in enumerate(cands)
              if c["is_label"]][0]
        tgt_tup = ctup(cands[li])
        cell = f"t{r['term_cell']}-{r['regime']}"
        row = {"block_id": r["block_id"], "cell": cell,
               "n_candidates": len(cands),
               "label_index": li}
        for a in ARMS:
            scores = [logp(a, c["_codes"][a]) for c in cands]
            tgt = scores[li]
            rivals = [s for i, s in enumerate(scores)
                      if i != li]
            mx = max(rivals)
            # pessimistic static rank: ties count against
            rank = 1 + sum(1 for s in rivals if s >= tgt)
            margin = tgt - mx
            c0 = scores[0]
            c0_top = (len(scores) > 1
                      and scores[0] > max(scores[1:]))
            if rank == 1:
                rank1[a] += 1
                rank1_cell[a][cell] += 1
            if c0_top:
                cand0_rank1[a] += 1
            margins[a].append(margin)
            margins_cell[a][cell].append(margin)
            tz = cands[li]["_codes"][a]
            # Family A components for the target
            prefv = [fam_a[a]["pref"].get(tuple(tz[:k]), 0)
                     for k in range(1, 9)]
            # Family C: mechanical witnesses
            (ru, sk, so, pk, t_idx) = tgt_tup
            wit = {
                "same_term_other_site":
                    code_of(a, (ru, sk, 0, pk, t_idx)),
                "same_site_t0": code_of(a, (ru, sk, so, pk, 0)),
                "same_site_t1": code_of(a, (ru, sk, so, pk, 1))}
            witgeo = {k: pair_geo(tz, w)
                      for k, w in wit.items()}
            rivgeo = [pair_geo(tz, c["_codes"][a])
                      for i, c in enumerate(cands) if i != li]

            def density(z):
                dists = [sum(1 for x, y in zip(z, w)
                             if x != y) for w in distinct[a]]
                mh = min(dists)
                rad = {f"r{r_}": sum(1 for d_ in dists
                                     if d_ <= r_)
                       for r_ in (1, 2, 3)}
                lcs = []
                for w in distinct[a]:
                    lc = 0
                    for x, y in zip(z, w):
                        if x != y:
                            break
                        lc += 1
                    lcs.append(lc)
                ml = max(lcs)
                return {"min_hamming_train": mh, **rad,
                        "max_train_lcp": ml,
                        "n_at_max_lcp": sum(
                            1 for x in lcs if x == ml)}
            riv_density = [density(c["_codes"][a])
                           for i, c in enumerate(cands)
                           if i != li]
            # Family D (target + rival control via density())
            tden = density(tz)
            row[a] = {
                "target_logp": round(tgt, 4),
                "max_rival_logp": round(mx, 4),
                "margin": round(margin, 4),
                "static_rank": rank,
                "cand0_logp": round(c0, 4),
                "target_minus_cand0": round(tgt - c0, 4),
                "cand0_static_top": c0_top,
                "target_prefix_counts": prefv,
                "witness_geometry": witgeo,
                "rival_geometry_full": rivgeo,
                "density": tden,
                "rival_density": {
                    k: {"min": min(d_[k] for d_ in riv_density),
                        "max": max(d_[k] for d_ in riv_density)}
                    for k in ("min_hamming_train", "r1", "r2",
                              "r3", "max_train_lcp")}}
            if a in ("HASH_P1", "HASH_P2"):
                tp = feistel_trace(tgt_tup, a)
                tr = sum(1 for p_ in tp if p_ in trav[a])
                rr = [sum(1 for p_ in feistel_trace(
                          ctup(c), a) if p_ in trav[a])
                      for i, c in enumerate(cands) if i != li]
                row[a]["feistel_reuse"] = {
                    "target_count": tr,
                    "target_fraction": tr / 4.0,
                    "rival_counts": rr}
        states.append(row)

    # Family B summaries + the FROZEN BAR components
    n = len(pri)
    summary = {
        "rank1_fraction": {a: round(rank1[a] / n, 4)
                           for a in ARMS},
        "rank1_by_cell": {a: {c: rank1_cell[a][c]
                              for c in rank1_cell[a]}
                          for a in ARMS},
        "margin_stats": {a: {k: round(v, 4) for k, v in
                             (stat5(margins[a]) or {}).items()}
                         for a in ARMS},
        "margin_stats_by_cell": {
            a: {c2: {k: round(v, 4) for k, v in
                     (stat5(margins_cell[a][c2])
                      or {}).items()}
                for c2 in margins_cell[a]} for a in ARMS},
        "cand0_static_rank1_census": cand0_rank1}
    d = (summary["rank1_fraction"]["HASH_P1"]
         - summary["rank1_fraction"]["HASH_P2"])
    cells_p1_above = sum(
        1 for t, rg in CELLS
        if rank1_cell["HASH_P1"][f"t{t}-{rg}"]
        > rank1_cell["HASH_P2"][f"t{t}-{rg}"])
    bar = {
        "p1_minus_p2_rank1_fraction": round(d, 4),
        "bar_abs_threshold": 0.25,
        "cells_p1_above_p2": cells_p1_above,
        "bar_component_met":
            d >= 0.25 and cells_p1_above >= 3}

    census = {"summary_family_B": summary,
              "frozen_bar_components": bar,
              "positional_counts": {
                  a: [dict(sorted(
                      (str(k), v) for k, v in c.items()))
                      for c in fam_a[a]["pos"]]
                  for a in ARMS},
              "train_bigram_counts": {
                  a: {str(k): v for k, v in sorted(
                      fam_a[a]["big"].items())}
                  for a in ARMS},
              "train_trigram_counts": {
                  a: {str(k): v for k, v in sorted(
                      fam_a[a]["tri"].items())}
                  for a in ARMS},
              "distinct_train_codewords": {
                  a: len(distinct[a]) for a in ARMS},
              "feistel_training_traversal": {
                  w: len(trav[w]) for w in trav},
              "states": states,
              "lambda": LAMBDA,
              "note": ("posthoc descriptive census; no "
                       "p-values; the bar components feed the "
                       "booking's categorical label only")}
    OUTDIR.mkdir(parents=True)
    (OUTDIR / "census.json").write_text(
        json.dumps(census, indent=1))
    receipt = {
        "prereg": "MATH-CYBER-1-SVP-BIJECTION-GEOMETRY-DESK-0",
        "definitions_frozen_at_commit": freeze_commit,
        "n_train": len(train_tups),
        "n_primary_states": n,
        "census_sha": fsha(OUTDIR / "census.json"),
        "p2_realization_rederived": p2_derived,
        "wall_s": round(time.time() - t0, 1),
        "pins": {p: fsha(p) for p in PINS},
        "start": START,
        "completion_commit": completion_commit()}
    for p, h in PINS.items():
        gate(fsha(p) == h, f"POST PIN {p}")
    (OUTDIR / "svpgeodesk_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({"summary_family_B": summary,
                      "frozen_bar_components": bar,
                      "distinct_train_codewords":
                          census["distinct_train_codewords"],
                      "feistel_training_traversal":
                          census["feistel_training_traversal"],
                      "wall_s": receipt["wall_s"]},
                     indent=1), flush=True)
    print("[svpgeodesk] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
