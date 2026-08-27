"""MATH-CYBER-1 SVP-STRICT-GRID-CHALLENGE-DESIGN-0 — materialize
+ qualify the FINAL prospective strict-recombination challenge
under the frozen law (prereg committed with this driver BEFORE
any byte is generated). Zero model/checkpoint access, zero
inference; scoring happens under the NEXT GO only.

Design (frozen; full text in the RESULTS prereg): matched
site-pair BLOCKS over fresh disjoint CH-F3/CH-F4 banks (240 base
targets x 3 distractor variants = 720 qualification parents,
every one attempted and accounted); block qualifies iff all
three parents pass the full svpeval3-mirrored candidate law with
teacher label i_unprod, identical term_index, and measured
ordinals 1/1/0 (small-D/small-D/sin(sin(x))); selection = first
N=12 per {family x term cell 1|2|3} by SHA256(base_signature),
D_before alternating x**x / 1/(x+log(x)) in hash order (6/6 by
construction); any short stratum => CHALLENGE NO-FIRE. Novelty:
zero exact cur overlap with training, bands 1-3, and all 566
burned pilot parents. Semantic gates recomputed in-run.
Codeword-anatomy rider persisted (outcome-independent).

Outputs under logs/mathworld1/svpchal/ (refuse-if-exists):
blocks.jsonl (every base, pass or fail), decisions.jsonl
(selected states, svpeval3-compatible rows + block metadata),
svpchal_receipt.json.

    .venv/bin/python scratch/mathworld1_svpchal.py            (Mac)
"""
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.search.derivation import State, hce  # noqa: E402
from scratch.mathworld1_actiontok import ActionGCTok  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpcode import (factor_decode,  # noqa: E402
                                        factor_symbols,
                                        hash_decode, hash_symbols,
                                        in_domain)
from scratch.mathworld1_svpeval import (derive_program,  # noqa: E402
                                        stable_legal_set)

X = sp.Symbol("x")
CTX = 4096
N_PER_STRATUM = 12
TOK = ActionGCTok()
OUTDIR = Path("logs/mathworld1/svpchal")
PAIRED = Path("data/matsub_paired.jsonl")
PINS = {
    "data/matsub_paired.jsonl":
        "a943ba7fc581db743b07192e5d951fadddd2ba19bca3225b75"
        "d8402351d468e8",
    "logs/mathworld1/svpeval/decisions.jsonl":
        "f63100a62f3091d544750d679483009a473261c587f316524140"
        "6a86253858c6",
    "logs/mathworld1/svpeval2/decisions.jsonl":
        "89efbe0ea447ee937c0c130d5419112921a2dd6c2159c6c2112c"
        "fd5e92f79315",
    "logs/mathworld1/svpeval3/decisions.jsonl":
        "2ff5433249622df9d421cf8014131b3907092a943040bb7b20f4"
        "6f1afffb7efa",
    "logs/mathworld1/svpgriddesk_receipt.json":
        "ec9cb9b870d2515e7959025f7f3cbfcee7309a6dc90b880d3176"
        "d3e2ccf72edc",
    "logs/mathworld1/svpgriddesk2_receipt.json":
        "f0184b01c36017bcb93ed4b715e41e015999075c4da7976e11ad"
        "c5cae28e6977",
    "logs/mathworld1/svpgriddesk3_receipt.json":
        "26389ebb8d68f45447d9676b9c188ea86ab9dd70ec71329fe6ae"
        "268cfc34080f",
    "logs/mathworld1/svpgriddesk4_receipt.json":
        "8439fc636fcf5c6e18a7dd75a76642cfe088e7eeeaf1e9f1243c"
        "5bb0cf08610f",
}
BAND_FILES = ["logs/mathworld1/svpeval/decisions.jsonl",
              "logs/mathworld1/svpeval2/decisions.jsonl",
              "logs/mathworld1/svpeval3/decisions.jsonl"]
PILOT_RECEIPTS = ["logs/mathworld1/svpgriddesk_receipt.json",
                  "logs/mathworld1/svpgriddesk2_receipt.json",
                  "logs/mathworld1/svpgriddesk3_receipt.json",
                  "logs/mathworld1/svpgriddesk4_receipt.json"]
SMALL_D = (X**X, 1 / (X + sp.log(X)))
AFTER_D = sp.sin(sp.sin(X))
HELD_OUT_TERMS = (2, 3)


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_horizon():
    space = []
    polys = (2 * X, 5 * X**2)
    for P in (X**4, X**3 + X**2, X**4 + X, X**5, X**3 + X,
              X**4 + X**2):
        for T in (sp.sin, sp.cos):
            for c in (4, 5):
                for w in (sp.exp(X) / X, sp.sin(X) / X):
                    for k in (0, 1, 2):
                        f = (sp.expand(sp.diff(P * T(c * X), X))
                             + sp.Integral(w, X)
                             + sp.Add(*polys[:k]))
                        sig = (f"CH-F3|P={P}|T={T.__name__}|"
                               f"c={c}|w={w}|k={k}")
                        space.append(("CH-F3", sig, f))
    for (P1, P2) in ((X**4 + X**2, X), (X**3 + X**2, X),
                     (X**4, X**2),
                     (X**5, X), (X**4, X**3), (X**2 + X, X**3),
                     (X**5, X**2), (X**3 + X, X**2),
                     (X**4 + X, X), (X**5, X**3),
                     (X**3 + X**2, X**2), (X**4, X**2 + X)):
        for (a, b) in ((4, 5), (5, 4), (5, 3), (3, 4)):
            for w in (sp.exp(X) / X, sp.sin(X) / X):
                f = (sp.expand(sp.diff(P1 * sp.sin(a * X), X))
                     + sp.expand(sp.diff(P2 * sp.cos(b * X), X))
                     + sp.Integral(w, X))
                sig = (f"CH-F4|P1={P1}|P2={P2}|a={a}|b={b}|"
                       f"w={w}")
                space.append(("CH-F4", sig, f))
    return space


def qualify_parent(f_t, D):
    """svpeval3-mirrored candidate law on one constructed parent.
    Returns (row, fail_reason). row carries the full candidate
    list; label is the TEACHER argmin, never constructor intent."""
    node = sp.Integral(f_t, X)
    parent = sp.Add(node, sp.Integral(D, X))
    cur = sp.sstr(parent)
    st = State(parent)
    acts, stable = stable_legal_set(st)
    if not stable:
        return {"cur": cur}, "legal_set_unstable"
    if not acts:
        return {"cur": cur}, "no_legal_actions"
    accepted = defaultdict(set)
    for n, c in acts:
        accepted[n.split("@", 1)[0] if "@" in n else n].add(
            c.key())
    scored = [(hce(c), n, c.key()) for n, c in acts]
    mn = min(scored)
    ties = sum(1 for s in scored if s[0] == mn[0])
    cands = []
    collide = {}
    chosen = None
    for n, c in sorted(acts, key=lambda nc: (nc[0],
                                             nc[1].key())):
        rule = n.split("@", 1)[0] if "@" in n else n
        prog, why = derive_program(parent, rule, c.key(),
                                   accepted)
        if prog is None:
            return {"cur": cur}, f"program_{why}"
        text = prog["program_text"]
        if TOK.decode(TOK.encode(text)) != text:
            return {"cur": cur}, "tok_roundtrip"
        ck = sp.sstr(c.expr)
        if text in collide and collide[text] != ck:
            return {"cur": cur}, "program_collision"
        collide[text] = ck
        pre = len(TOK.encode(
            f"Current: {cur}\nHints: none\nStep: "))
        stl = pre + len(TOK.encode(ck + "\n")) + 1
        pgl = pre + len(TOK.encode(text)) + 1
        if stl > CTX or pgl > CTX:
            return {"cur": cur}, "context_overflow"
        tup = (prog["rule"], prog["site_kind"],
               prog["site_ordinal"], prog["param_kind"],
               prog["param_index"])
        if not in_domain(*tup):
            return {"cur": cur}, "code_domain"
        fs = factor_symbols(*tup)
        hs = hash_symbols(*tup)
        if factor_decode(fs) != tup or hash_decode(hs) != tup:
            return {"cur": cur}, "code_roundtrip"
        is_label = (n, c.key()) == (mn[1], mn[2])
        if is_label:
            chosen = tup
        cands.append({
            "child_sstr": ck, "child_srepr": sp.srepr(c.expr),
            **prog, "factor_code": fs, "hash_code": hs,
            "is_label": is_label, "state_seq_tokens": stl,
            "program_seq_tokens": pgl})
    if sum(c["is_label"] for c in cands) != 1:
        return {"cur": cur}, "label_not_unique"
    return {"cur": cur, "parent_srepr_sha":
            hashlib.sha256(sp.srepr(parent).encode()
                           ).hexdigest()[:16],
            "n_candidates": len(cands), "min_hce_ties": ties,
            "chosen_rule": chosen[0], "chosen_ordinal": chosen[2],
            "chosen_site_kind": chosen[1],
            "chosen_param_kind": chosen[3],
            "chosen_term": chosen[4], "candidates": cands}, None


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN MISMATCH {p}")
    START = start_provenance(
        ["scratch/mathworld1_svpchal.py",
         "scratch/mathworld1_svpeval.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_actionsem.py",
         "scratch/mathworld1_actionfinal.py",
         "scratch/mathworld1_unprodsem.py",
         "scratch/mathworld0.py",
         "llmopt/search/derivation.py", "llmopt/search/rules.py",
         "llmopt/lab/provenance.py"])
    # novelty reference sets + semantic support
    train_cur, seen, rs, rp = set(), set(), set(), set()
    for l in open(PAIRED):
        r = json.loads(l)
        train_cur.add(r["cur"])
        t = (r["rule"], r["site_kind"], r["site_ordinal"],
             r["param_kind"], r["param_index"])
        seen.add(t)
        rs.add((t[0], (t[1], t[2])))
        rp.add((t[0], (t[3], t[4])))
    band_cur = set()
    for bf in BAND_FILES:
        for l in open(bf):
            r = json.loads(l)
            if r.get("cur"):
                band_cur.add(r["cur"])
    pilot_cur = set()
    for pr in PILOT_RECEIPTS:
        for a in json.loads(Path(pr).read_text())["attempts"]:
            pilot_cur.add(a["parent_sstr"])
    gate(len(pilot_cur) == 566, f"PILOT PARENTS {len(pilot_cur)}")
    # semantic gates
    for t in HELD_OUT_TERMS:
        cell = ("i_unprod", "I", 1, "term_index", t)
        gate(cell not in seen, f"HELD-OUT IN TRAINING {cell}")
        gate(("i_unprod", ("I", 1)) in rs, "RS NOT COVERED")
        gate(("i_unprod", ("term_index", t)) in rp,
             f"RP NOT COVERED t{t}")
    for (o, t) in ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0),
                   (1, 1)):
        gate(("i_unprod", "I", o, "term_index", t) in seen,
             f"COVERED CELL ABSENT I{o}t{t}")

    horizon = build_horizon()
    gate(len(horizon) == 240, f"HORIZON {len(horizon)}")
    gate(len({s for _, s, _ in horizon}) == 240, "SIG DUP")
    OUTDIR.mkdir(parents=True)
    blocks = []
    t_all = time.monotonic()
    for i, (fam, sig, f_t) in enumerate(horizon):
        variants = []
        fail = None
        for tag, D in (("smallA", SMALL_D[0]),
                       ("smallB", SMALL_D[1]),
                       ("after", AFTER_D)):
            row, why = qualify_parent(f_t, D)
            row["variant"] = tag
            row["distractor"] = sp.sstr(D)
            if why:
                fail = f"{tag}:{why}"
                variants.append(row)
                break
            variants.append(row)
        blk = {"family": fam, "base_signature": sig,
               "target_integrand": sp.sstr(f_t),
               "sig_sha": hashlib.sha256(
                   sig.encode()).hexdigest(),
               "variants_pass": fail is None, "fail": fail}
        if fail is None:
            rules = {v["chosen_rule"] for v in variants}
            kinds = {(v["chosen_site_kind"],
                      v["chosen_param_kind"]) for v in variants}
            terms = {v["chosen_term"] for v in variants}
            ords = [v["chosen_ordinal"] for v in variants]
            if kinds != {("I", "term_index")}:
                blk["fail"] = f"teacher_kind:{sorted(kinds)}"
            elif rules != {"i_unprod"}:
                blk["fail"] = f"teacher_rule:{sorted(rules)}"
            elif len(terms) != 1:
                blk["fail"] = f"term_mismatch:{sorted(terms)}"
            elif ords != [1, 1, 0]:
                blk["fail"] = f"ordinals:{ords}"
            else:
                blk["term"] = terms.pop()
            blk["variants_pass"] = blk["fail"] is None
        blk["_variants"] = variants
        blocks.append(blk)
        print(f"[{i+1}/240] {fam} "
              f"{'OK t' + str(blk.get('term')) if blk['fail'] is None else blk['fail']}",
              flush=True)
    # strata + selection
    strata = defaultdict(list)
    for b in blocks:
        if b["fail"] is None and b["term"] in (1, 2, 3):
            strata[(b["family"], b["term"])].append(b)
    for k in strata:
        strata[k].sort(key=lambda b: b["sig_sha"])
    stratum_counts = {f"{f}|t{t}": len(strata.get((f, t), []))
                      for f in ("CH-F3", "CH-F4")
                      for t in (1, 2, 3)}
    short = [k for k, v in stratum_counts.items()
             if v < N_PER_STRATUM]
    receipt = {
        "prereg": "MATH-CYBER-1-SVP-STRICT-GRID-CHALLENGE-"
                  "DESIGN-0",
        "n_horizon": len(horizon),
        "stratum_qualified_counts": stratum_counts,
        "block_fail_census": dict(Counter(
            b["fail"] for b in blocks if b["fail"])),
        "qualified_terms_census": dict(Counter(
            b.get("term") for b in blocks
            if b["fail"] is None)),
        "wall_s": round(time.monotonic() - t_all, 1),
        "pins": {p: fsha(p) for p in PINS} | {
            p: fsha(p) for p in PILOT_RECEIPTS},
    }
    with open(OUTDIR / "blocks.jsonl", "w") as fh:
        for b in blocks:
            fh.write(json.dumps(
                {k: v for k, v in b.items()
                 if k != "_variants"}) + "\n")
    if short:
        receipt["verdict"] = "CHALLENGE NO-FIRE"
        receipt["short_strata"] = short
        receipt["start"] = START
        receipt["completion_commit"] = completion_commit()
        (OUTDIR / "svpchal_receipt.json").write_text(
            json.dumps(receipt, indent=1))
        print(json.dumps({k: v for k, v in receipt.items()
                          if k not in ("start", "pins")},
                         indent=1), flush=True)
        return 0
    # mechanical selection + D_before assignment
    dec_rows = []
    sel_census = Counter()
    all_curs = []
    dupsig = set()
    for f in ("CH-F3", "CH-F4"):
        for t in (1, 2, 3):
            sel = strata[(f, t)][:N_PER_STRATUM]
            for j, b in enumerate(sel):
                gate(b["base_signature"] not in dupsig,
                     "DUP SIG SELECTED")
                dupsig.add(b["base_signature"])
                primary_small = "smallA" if j % 2 == 0 \
                    else "smallB"
                robust_small = "smallB" if j % 2 == 0 \
                    else "smallA"
                by = {v["variant"]: v for v in b["_variants"]}
                role_i1 = ("control-I1" if t == 1
                           else "heldout-I1")
                role_i0 = ("control-I0" if t == 1
                           else "covered-I0")
                for role, v in (
                        (role_i1, by[primary_small]),
                        (role_i0, by["after"]),
                        ("robustness-I1", by[robust_small])):
                    dec_rows.append({
                        "block_id": f"{f}-t{t}-{j:02d}",
                        "family": f, "term_cell": t,
                        "site_role": role,
                        "primary": role != "robustness-I1",
                        "distractor": v["distractor"],
                        "block_d_before": by[primary_small][
                            "distractor"],
                        "base_signature": b["base_signature"],
                        "cur": v["cur"],
                        "parent_srepr_sha":
                            v["parent_srepr_sha"],
                        "n_candidates": v["n_candidates"],
                        "min_hce_ties": v["min_hce_ties"],
                        "chosen_ordinal": v["chosen_ordinal"],
                        "chosen_term": v["chosen_term"],
                        "candidates": v["candidates"]})
                    all_curs.append(v["cur"])
                sel_census[f"{f}|t{t}|"
                           f"{by[primary_small]['distractor']}"
                           ] += 1
    def refuted(reason):
        receipt["verdict"] = "CHALLENGE GATE-REFUTED"
        receipt["refuted_reason"] = reason
        receipt["start"] = START
        receipt["completion_commit"] = completion_commit()
        (OUTDIR / "svpchal_receipt.json").write_text(
            json.dumps(receipt, indent=1))
        raise SystemExit(f"GATE-REFUTED: {reason}")

    # balance assertion (exact 6/6 by construction)
    for f in ("CH-F3", "CH-F4"):
        for t in (1, 2, 3):
            a = sel_census.get(f"{f}|t{t}|x**x", 0)
            bd = sel_census.get(f"{f}|t{t}|1/(x + log(x))", 0)
            if not (a == 6 and bd == 6):
                refuted(f"D BALANCE {f} t{t} {a}/{bd}")
    # novelty gates on every emitted parent
    if len(all_curs) != len(set(all_curs)):
        refuted("DUP PARENT")
    for cur in all_curs:
        if cur in train_cur:
            refuted("TRAIN CUR OVERLAP")
        if cur in band_cur:
            refuted("BAND CUR OVERLAP")
        if cur in pilot_cur:
            refuted("PILOT CUR OVERLAP")
    # target-integrand overlap (report only, never a gate):
    # exact substring occurrence of the selected target
    # integrand's sstr inside any training/band visible cur
    tset = sorted({b["target_integrand"] for b in blocks
                   if b["base_signature"] in dupsig})
    tgt_overlap = {
        "training": sum(1 for ti in tset if any(
            ti in c for c in train_cur)),
        "bands": sum(1 for ti in tset if any(
            ti in c for c in band_cur))}
    with open(OUTDIR / "decisions.jsonl", "w") as fh:
        for r in dec_rows:
            fh.write(json.dumps(r) + "\n")
    # codeword anatomy rider (outcome-independent)
    train_rows = Counter()
    for l in open(PAIRED):
        r = json.loads(l)
        train_rows[(r["rule"], r["site_kind"],
                    r["site_ordinal"], r["param_kind"],
                    r["param_index"])] += 1
    tcodes = {}
    for tp, cnt in train_rows.items():
        tcodes[tp] = {"factor": factor_symbols(*tp),
                      "hash": hash_symbols(*tp), "count": cnt}
    rider = {}
    for t in HELD_OUT_TERMS:
        cell = ("i_unprod", "I", 1, "term_index", t)
        for kind, code in (("factor", factor_symbols(*cell)),
                           ("hash", hash_symbols(*cell))):
            dists = []
            for tp, tc in tcodes.items():
                d = sum(1 for a2, b2 in zip(code, tc[kind])
                        if a2 != b2)
                pfx = 0
                for a2, b2 in zip(code, tc[kind]):
                    if a2 != b2:
                        break
                    pfx += 1
                dists.append((d, pfx, " ".join(map(str, tp))))
            dists.sort()
            mind = dists[0][0]
            pos_support = []
            for i2, sym in enumerate(code):
                cs = Counter()
                for tp, tc in tcodes.items():
                    cs[tc[kind][i2]] += tc["count"]
                pos_support.append({
                    "position": i2, "symbol": sym,
                    "training_rows_with_symbol": cs.get(sym, 0),
                    "distinct_symbols_in_training": len(cs)})
            rider[f"I1t{t}:{kind}"] = {
                "code": code,
                "nearest_hamming": mind,
                "nearest_codewords": [x[2] for x in dists
                                      if x[0] == mind],
                "longest_shared_prefix": max(
                    p for _, p, _ in dists),
                "per_position_support": pos_support}
    receipt |= {
        "verdict": "CHALLENGE MATERIALIZED",
        "n_blocks_selected": len(dupsig),
        "n_states": len(dec_rows),
        "n_primary_states": sum(r["primary"] for r in dec_rows),
        "site_role_census": dict(Counter(
            r["site_role"] for r in dec_rows)),
        "d_before_census": dict(sel_census),
        "legal_set_size_census": dict(Counter(
            r["n_candidates"] for r in dec_rows)),
        "min_hce_ties_census": dict(Counter(
            r["min_hce_ties"] for r in dec_rows)),
        "target_integrand_overlap": tgt_overlap,
        "codeword_anatomy": rider,
        "decisions_sha": fsha(OUTDIR / "decisions.jsonl"),
        "blocks_sha": fsha(OUTDIR / "blocks.jsonl"),
        "start": START,
        "completion_commit": completion_commit()}
    (OUTDIR / "svpchal_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    out = {k: v for k, v in receipt.items()
           if k not in ("start", "pins", "codeword_anatomy",
                        "legal_set_size_census")}
    print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
