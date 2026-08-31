"""MATH-CYBER-1 SVP-DECODE-CONSTRAINT-LADDER — LEGAL-TRIE
GREEDY (L) instrument, the ONE new measurement of the frozen
U -> L -> R ladder (prereg DECODE-CONSTRAINT-LADDER-PREREG-0,
commit 3b5f1742). U and R are FROZEN AUTHORITIES, never rerun.

  SVPDCL_QUAL=1  qualification ONLY, no checkpoint touched:
                 all pins; per-state x arm trie construction
                 with LEAF COUNT == n_candidates, unique
                 candidate<->leaf bijection, teacher leaf,
                 642/642 candidate roundtrips per arm (321
                 covered + 321 heldout; the prereg registers
                 the law, not a count), semantic
                 candidate identity across arms; U legality
                 gate (reproduce 2/96 legal per heldout cell
                 from the frozen FREE-ACTION-1 raw + illegal
                 anatomy); R vectors reproduced from frozen
                 artifacts (85/78/80/87 heldout, 94/93/85/91
                 calibration); support-matrix recount from
                 corpus bytes.
  SVPDCL_RUN=1   ONE joint L run (requires green qual): four
                 checkpoints x two populations x 96 states;
                 at each position allowed = tokens extending
                 the reached prefix to >=1 complete frozen
                 legal serialization (EOS only at position 9);
                 select highest-logit ALLOWED token, exact
                 ties -> lowest token ID; no lookahead, no
                 beam, no teacher preference, no retry; full
                 per-position receipt (unmasked top-1, allowed
                 set, ranks, teacher token quantities,
                 surviving leaves before/after); U-coherence
                 census (unmasked top-1 v frozen U tokens on
                 byte-identical prefixes, >5% mismatch =
                 INSTRUMENT FAILURE for Contrast 1); ONE
                 teacher-forced forward per heldout state per
                 checkpoint at the withheld digit (softmax
                 rider); raw-first; frozen endpoints (8
                 applications, min-likelihood McNemar).

Outputs under logs/mathworld1/svpdcl/ (refuse-if-exists per
file): qual.json, raw_trie_decode.jsonl, receipt.json,
riders.json. NO SMOKE MODE: read-only scorer over frozen
bytes + four pinned checkpoints; refuse-if-exists is the sole
guard (disclosed).

    SVPDCL_QUAL=1 .venv/bin/python scratch/mathworld1_svpdcl.py
    SVPDCL_RUN=1  .venv/bin/python scratch/mathworld1_svpdcl.py
"""
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from scratch.mathworld1_actiontok import ActionGCTok  # noqa: E402
from scratch.mathworld1_svpcode import (factor_decode,  # noqa: E402
                                        in_domain)
from scratch.mathworld1_svpfoheld import binom_minlik_p  # noqa: E402
from scratch.mathworld1_svpforder import (INV, PERM,  # noqa: E402
                                          pf_decode)

CAL = "logs/mathworld1/svpdiet3/covered_calibration.jsonl"
HELD = "logs/mathworld1/svpdiet3/heldout_test16.jsonl"
UGEN = "logs/mathworld1/svpfreeact/raw_generations.jsonl"
PINS = {
    CAL: ("af1a4aa1df7bf3224745e91a90e1a77c36e5c54f7ff9b085"
          "09794d0fb7978db3"),
    HELD: ("a3f6103b3733d909281849dcb3fd6ba9fba3891f2014bec1"
           "3881b4509df46ddb"),
    UGEN: ("a04f13ba456da7e7c41a5e38f83bb15a6d7406513bc18ca8"
           "a83c8303bdfdd7bc"),
    "logs/mathworld1/svpfoheld/raw_token_scores.jsonl":
        ("ee0319f4c63396f7998cdffa5dee5b3d8c3bcf4b35866a3dcf"
         "499efb9c673111"),
    "logs/mathworld1/svpfoheld20/raw_token_scores.jsonl":
        ("18f27b666ce54c5402abff746e696f9b8c35c94235d096a060"
         "f97cf015aae17f"),
    "logs/mathworld1/svpfocal/scores.jsonl":
        ("deb7f59fe837423f9561b190451f770cc574d3f9486a4294ae"
         "6587233796a879"),
    "logs/mathworld1/svpfocal20/scores.jsonl":
        ("5df6ee83ff684920395d634a3f6f8ba9c6b7c96f5805540c8b"
         "8cefc25f968ddf"),
    "data/matsub_paired.jsonl":
        ("a943ba7fc581db743b07192e5d951fadddd2ba19bca3225b75"
         "d8402351d468e8"),
    "logs/mathworld1/svpdiet/combined_train_manifest.jsonl":
        ("897c8bf8fd2b6d39e361ed541d3e14c53c1c1302eeed560f77"
         "ee8fb2f2477bdd"),
    "logs/mathworld1/svpdiet/balanced_grid_train.jsonl":
        ("0ef3d8a880a7e07712d8de757bc1670df12701e487b856b44c"
         "97f8db16cb3759"),
}
CKPTS = {
    (19001, "CANONICAL"):
        ("checkpoints/svp_forder_canonical_s19001.pt",
         "ae0a86e027d8b0ca1cd7a97a83a6927d326da5bd34258910b1"
         "b81d3492322e1d"),
    (19001, "PARAM_FIRST"):
        ("checkpoints/svp_forder_paramfirst_s19001.pt",
         "0fe38f785f68165e868c54fff482844ea4b2476c737f2e4af5"
         "0990ece6df390f"),
    (20001, "CANONICAL"):
        ("checkpoints/svp_forder_canonical_s20001.pt",
         "0a841a5f2a43b6f64b0dac8259c26fd79961e6ab91359a54be"
         "9c2582815b3e34"),
    (20001, "PARAM_FIRST"):
        ("checkpoints/svp_forder_paramfirst_s20001.pt",
         "b7198ff2e7b903ab5ed075fe947cb29142c5790ec84831434c"
         "53a598e466c322"),
}
RANK_RAW = {19001: "logs/mathworld1/svpfoheld/"
                   "raw_token_scores.jsonl",
            20001: "logs/mathworld1/svpfoheld20/"
                   "raw_token_scores.jsonl"}
CAL_SCORES = {19001: "logs/mathworld1/svpfocal/scores.jsonl",
              20001: "logs/mathworld1/svpfocal20/scores.jsonl"}
RANK_EXPECT_HELD = {(19001, "CANONICAL"): 85,
                    (19001, "PARAM_FIRST"): 78,
                    (20001, "CANONICAL"): 80,
                    (20001, "PARAM_FIRST"): 87}
RANK_EXPECT_CAL = {(19001, "CANONICAL"): 94,
                   (19001, "PARAM_FIRST"): 93,
                   (20001, "CANONICAL"): 85,
                   (20001, "PARAM_FIRST"): 91}
SEEDS = [19001, 20001]
ARMS = ["CANONICAL", "PARAM_FIRST"]
VOCAB = 340
CODE_BASE = 332
WITHHELD = {"CANONICAL": 8, "PARAM_FIRST": 3}
OUTDIR = Path("logs/mathworld1/svpdcl")
QUAL = OUTDIR / "qual.json"
RAW = OUTDIR / "raw_trie_decode.jsonl"
RECEIPT = OUTDIR / "receipt.json"
RIDERS = OUTDIR / "riders.json"
TOK = ActionGCTok()
EOS = TOK.eos_id
SRC = ["scratch/mathworld1_svpdcl.py",
       "scratch/mathworld1_svpfreeact.py",
       "scratch/mathworld1_svpfoheld.py",
       "scratch/mathworld1_svpforder.py",
       "scratch/mathworld1_svpcode.py",
       "scratch/mathworld1_actiontok.py",
       "llmopt/train/mathnative.py",
       "llmopt/lab/provenance.py"]


def gate(cond, msg):
    if not cond:
        raise SystemExit(f"GATE FAILED: {msg}")


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ctup(c):
    return (c["rule"], c["site_kind"], c["site_ordinal"],
            c["param_kind"], c["param_index"])


def serialize(arm, fc):
    """Candidate serialization from the stored factor_code."""
    syms = (fc if arm == "CANONICAL"
            else [fc[PERM[i]] for i in range(8)])
    return [CODE_BASE + s for s in syms] + [EOS]


def load_pop(which):
    rows = [json.loads(l) for l in open(
        CAL if which == "A" else HELD)]
    if which == "A":
        gate(len(rows) == 96 and all(
            r["site_role"] == "covered-I0" for r in rows),
            "POP A")
    else:
        rows = [r for r in rows
                if r["site_role"] == "heldout-I1"]
        gate(len(rows) == 96, "POP B")
    for r in rows:
        labs = [c for c in r["candidates"] if c["is_label"]]
        gate(len(labs) == 1, "LABEL COUNT")
        r["_teacher"] = ctup(labs[0])
        gate(len(r["candidates"]) == r["n_candidates"],
             "CAND COUNT")
        # semantic candidate identity is arm-independent by
        # construction (one candidates list); serialize + gate
        r["_tries"] = {}
        for arm in ARMS:
            leaves = {}
            for c in r["candidates"]:
                fc = c["factor_code"]
                seq = serialize(arm, fc)
                gate(len(seq) == 9, "LEAF LEN")
                t = ctup(c)
                syms = [x - CODE_BASE for x in seq[:8]]
                dec = (factor_decode(syms)
                       if arm == "CANONICAL"
                       else pf_decode(syms))
                gate(dec == t and in_domain(*t),
                     f"CAND RT {arm}")
                key = tuple(seq)
                gate(key not in leaves, f"DUP LEAF {arm}")
                leaves[key] = t
            gate(len(leaves) == r["n_candidates"],
                 f"LEAF COUNT {arm}")
            tseq = tuple(serialize(
                arm, labs[0]["factor_code"]))
            gate(tseq in leaves, f"TEACHER LEAF {arm}")
            r["_tries"][arm] = leaves
    return rows


def rank_vectors():
    rv_h, rv_c = {}, {}
    for seed in SEEDS:
        for line in open(RANK_RAW[seed]):
            row = json.loads(line)
            cums = []
            for v in row["token_lps"]:
                s = 0.0
                cc = []
                for x in v:
                    s += x
                    cc.append(s)
                cums.append(cc)
            li = row["label_index"]
            g = cums[li][8]
            rv_h[(seed, row["arm"], row["block_id"])] = all(
                cums[j][8] < g for j in range(len(cums))
                if j != li)
        for line in open(CAL_SCORES[seed]):
            row = json.loads(line)
            li = row["label_index"]
            for arm in ARMS:
                m = row[arm]["mean_lp"]
                rv_c[(seed, arm, row["block_id"])] = all(
                    m[j] < m[li] for j in range(len(m))
                    if j != li)
    for (sd, a), exp in RANK_EXPECT_HELD.items():
        tot = sum(1 for (s, ar, b), v in rv_h.items()
                  if s == sd and ar == a and v)
        gate(tot == exp, f"RANK HELD {sd} {a} {tot}!={exp}")
    for (sd, a), exp in RANK_EXPECT_CAL.items():
        tot = sum(1 for (s, ar, b), v in rv_c.items()
                  if s == sd and ar == a and v)
        gate(tot == exp, f"RANK CAL {sd} {a} {tot}!={exp}")
    return rv_h, rv_c


def load_u():
    """Frozen U authority: per (seed, arm, pop, block)."""
    u = {}
    for line in open(UGEN):
        r = json.loads(line)
        u[(r["seed"], r["arm"], r["pop"], r["block_id"])] = r
    gate(len(u) == 768, "U ROWS")
    return u


def u_exclusion_gate(u, popB):
    out = {}
    for (seed, arm) in CKPTS:
        legal = ill_pi_only = ill_other = 0
        covered_cell = 0
        for r in popB:
            row = u[(seed, arm, "B", r["block_id"])]
            if row["legal_action"]:
                legal += 1
                continue
            dt = row["decoded_tuple"]
            tt = row["teacher_tuple"]
            if dt is not None and dt[:4] == tt[:4] \
                    and dt[4] != tt[4]:
                ill_pi_only += 1
                if dt[4] in (0, 1):
                    covered_cell += 1
            else:
                ill_other += 1
        gate(legal == 2, f"U LEGAL {seed} {arm} {legal}!=2")
        out[f"{seed}|{arm}"] = {
            "U_legal": legal,
            "U_illegal_param_index_only": ill_pi_only,
            "U_illegal_other": ill_other,
            "U_illegal_pi_in_trained_cell": covered_cell}
    return out


def support_matrix():
    cnt = Counter()
    for p in ("data/matsub_paired.jsonl",
              "logs/mathworld1/svpdiet/"
              "balanced_grid_train.jsonl"):
        for line in open(p):
            r = json.loads(line)
            if r["rule"] == "i_unprod" \
                    and r["site_kind"] == "I":
                cnt[(r["site_ordinal"], r["param_kind"],
                     r["param_index"])] += 1
    so1 = {str(pi): cnt.get((1, "term_index", pi), 0)
           for pi in (0, 1, 2, 3)}
    gate(so1["2"] == 0 and so1["3"] == 0
         and so1["0"] > 0 and so1["1"] > 0,
         f"SUPPORT FACT {so1}")
    return {"i_unprod_I_site1_term_index": so1,
            "i_unprod_I_site0_term_index": {
                str(pi): cnt.get((0, "term_index", pi), 0)
                for pi in (0, 1, 2, 3)}}


def qual_main():
    if QUAL.exists():
        raise SystemExit(f"REFUSING: {QUAL} exists")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN {p}")
    for k, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"CKPT PIN {k}")
    START = start_provenance(SRC)
    gate(TOK.vocab_size == CODE_BASE == 332 and EOS == 1,
         "ALPHABET")
    gate(all(INV[PERM[i]] == i for i in range(8)), "INV LAW")
    gate(TOK.encode("Current: x\nHints: none\nStep: ")
         == TOK.encode("Current: x\nHints: none\nStep: "),
         "PROMPT DET")
    pops = {"A": load_pop("A"), "B": load_pop("B")}
    n_cands = sum(r["n_candidates"] for rows in pops.values()
                  for r in rows)
    gate(n_cands == 642, f"CAND TOTAL {n_cands}")
    # 642 = 321 covered + 321 heldout candidates per arm,
    # derived from the pinned population bytes; the prereg
    # registers the roundtrip LAW without a count.
    rank_vectors()
    u = load_u()
    excl = u_exclusion_gate(u, pops["B"])
    sup = support_matrix()
    receipt = {
        "prereg": "MATH-CYBER-1-SVP-DECODE-CONSTRAINT-"
                  "LADDER-PREREG-0",
        "prereg_commit": "3b5f1742",
        "verdict": "QUALIFIED",
        "gates": {
            "pins": len(PINS) + len(CKPTS),
            "candidate_roundtrips_per_arm": n_cands,
            "trie_states_gated": 192,
            "rank_vectors_reproduced": True,
            "u_rows": 768},
        "u_exclusion": excl,
        "support_matrix": sup,
        "start": START,
        "completion_commit": completion_commit()}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    QUAL.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in
                      ("verdict", "gates", "u_exclusion",
                       "support_matrix")}, indent=1),
          flush=True)
    print("[svpdcl:qual] QUALIFIED", flush=True)
    return 0


def run_main():
    import torch
    from llmopt.train.mathnative import build_model
    for p in (RAW, RECEIPT, RIDERS):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    gate(QUAL.exists(), "QUAL NOT RUN")
    qr = json.loads(QUAL.read_text())
    gate(qr["verdict"] == "QUALIFIED", "NOT QUALIFIED")
    for pth, h in qr["start"]["file_sha256"].items():
        gate(fsha(pth) == h, f"QUAL STALE {pth}")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN {p}")
    for k, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"CKPT PIN {k}")
    START = start_provenance(SRC)
    pops = {"A": load_pop("A"), "B": load_pop("B")}
    rv_h, rv_c = rank_vectors()
    u = load_u()
    gate(torch.backends.mps.is_available(), "MPS")
    dev = torch.device("mps")

    t0 = time.time()
    recs = []
    tf_rows = []
    coh = {"A": [0, 0], "B": [0, 0]}
    for (seed, arm), (path, csha) in CKPTS.items():
        m = build_model(VOCAB, ctx=4096)
        m.load_state_dict(torch.load(path, weights_only=True))
        gate(sum(p.numel() for p in m.parameters())
             == 19142016, "PARAMS")
        m.eval()
        m = m.to(dev)

        def logits_at(ids):
            x = torch.tensor([ids], device=dev)
            with torch.no_grad():
                return m(x)[0, -1].float().cpu()

        for pop, rows in pops.items():
            for r in rows:
                pre = TOK.encode(f"Current: {r['cur']}\n"
                                 f"Hints: none\nStep: ")
                gate(hashlib.sha256(
                    json.dumps(pre).encode()).hexdigest()
                    == u[(seed, arm, pop, r["block_id"])]
                    ["prompt_sha"], "PROMPT SHA v U")
                leaves = r["_tries"][arm]
                surviving = list(leaves)
                ids = list(pre)
                gen = []
                steps = []
                urow = u[(seed, arm, pop, r["block_id"])]
                tseq = [x for x in serialize(
                    arm, [c["factor_code"]
                          for c in r["candidates"]
                          if c["is_label"]][0])]
                for t in range(9):
                    logits = logits_at(ids)
                    allowed = sorted({leaf[t]
                                      for leaf in surviving})
                    mx_full = logits.max()
                    full_top_ids = (logits == mx_full).nonzero(
                        as_tuple=True)[0].tolist()
                    full_top = min(full_top_ids)
                    al = torch.tensor(allowed)
                    av = logits[al]
                    mx_a = av.max()
                    tie_ids = [allowed[i] for i in
                               (av == mx_a).nonzero(
                                   as_tuple=True)[0].tolist()]
                    chosen = min(tie_ids)
                    lp = torch.log_softmax(logits, -1)
                    order = torch.argsort(
                        logits, descending=True).tolist()
                    sel_rank_full = order.index(chosen) + 1
                    teacher_tok = tseq[t]
                    t_allowed = teacher_tok in allowed
                    t_rank_allowed = (sorted(
                        allowed,
                        key=lambda a: (-float(logits[a]), a))
                        .index(teacher_tok) + 1
                        if t_allowed else None)
                    before = len(surviving)
                    surviving = [leaf for leaf in surviving
                                 if leaf[t] == chosen]
                    after = len(surviving)
                    gate(after >= 1, "TRIE DEAD END")
                    # U coherence: prefix byte-identity
                    u_pref = urow["gen_tokens"][:t]
                    if gen == u_pref \
                            and t < len(urow["gen_tokens"]):
                        coh[pop][0] += 1
                        if full_top != urow["gen_tokens"][t]:
                            coh[pop][1] += 1
                    steps.append({
                        "t": t + 1,
                        "full_top": int(full_top),
                        "full_top_lp": float(lp[full_top]),
                        "allowed": [int(a) for a in allowed],
                        "allowed_size": len(allowed),
                        "selected": int(chosen),
                        "sel_rank_full": sel_rank_full,
                        "sel_lp": float(lp[chosen]),
                        "masked_out_top": full_top
                            not in allowed,
                        "teacher_tok": int(teacher_tok),
                        "teacher_lp": float(lp[teacher_tok]),
                        "teacher_allowed": t_allowed,
                        "teacher_rank_allowed":
                            t_rank_allowed,
                        "teacher_unique_allowed":
                            t_allowed and len(allowed) == 1,
                        "leaves_before": before,
                        "leaves_after": after,
                        "n_ties": len(tie_ids)})
                    gen.append(int(chosen))
                    ids.append(int(chosen))
                gate(len(gen) == 9 and gen[8] == EOS,
                     "L TERMINAL")
                key = tuple(gen)
                gate(key in leaves, "L NOT A LEAF")
                dt = leaves[key]
                legal_derived = (in_domain(*dt) and dt in
                                 {ctup(c) for c
                                  in r["candidates"]})
                teach = dt == r["_teacher"]
                recs.append({
                    "seed": seed, "arm": arm, "pop": pop,
                    "block_id": r["block_id"],
                    "ckpt_sha": csha,
                    "gen_tokens": gen,
                    "decoded_tuple": list(dt),
                    "teacher_tuple": list(r["_teacher"]),
                    "legal_action": legal_derived,
                    "teacher_match": teach,
                    "steps": steps})
                # teacher-forced softmax rider (heldout only)
                if pop == "B":
                    wpos = WITHHELD[arm]
                    tf_ids = list(pre) + tseq[:wpos - 1]
                    logits = logits_at(tf_ids)
                    lp = torch.log_softmax(logits, -1)
                    probs = torch.softmax(logits, -1)
                    order = torch.argsort(
                        logits, descending=True).tolist()
                    ttok = tseq[wpos - 1]
                    atoms = list(range(CODE_BASE,
                                       CODE_BASE + 8))
                    arank = sorted(
                        atoms,
                        key=lambda a: (-float(logits[a]), a)
                    ).index(ttok) + 1
                    comp = min(int(x) for x in
                               (logits == logits.max())
                               .nonzero(as_tuple=True)[0]
                               .tolist())
                    # covered v heldout param LOW-DIGIT mass:
                    # sym values for x low digit (pi+1)%8
                    cov = [CODE_BASE + ((pi + 1) % 8)
                           for pi in (0, 1)]
                    hel = [CODE_BASE + ((pi + 1) % 8)
                           for pi in (2, 3)]
                    tf_rows.append({
                        "seed": seed, "arm": arm,
                        "block_id": r["block_id"],
                        "withheld_pos": wpos,
                        "teacher_tok": int(ttok),
                        "teacher_rank_full":
                            order.index(ttok) + 1,
                        "teacher_rank_atoms": arank,
                        "teacher_lp": float(lp[ttok]),
                        "top_tok": comp,
                        "top_lp": float(lp[comp]),
                        "mass_covered_pi01":
                            float(sum(probs[c] for c in cov)),
                        "mass_heldout_pi23":
                            float(sum(probs[h] for h in hel))})
        del m
        print(f"[svpdcl] {seed} {arm} done "
              f"({time.time() - t0:.0f}s)", flush=True)

    with open(RAW, "w") as fo:
        for rec in recs:
            fo.write(json.dumps(rec) + "\n")
        for row in tf_rows:
            fo.write(json.dumps({"kind": "teacher_forced",
                                 **row}) + "\n")
    raw_sha = fsha(RAW)
    gate(len(recs) == 768 and len(tf_rows) == 384,
         "RAW COUNTS")
    coh_frac_b = (coh["B"][1] / coh["B"][0]
                  if coh["B"][0] else 0.0)
    gate(coh_frac_b <= 0.05,
         f"U COHERENCE HELDOUT {coh['B'][1]}/{coh['B'][0]}")

    def sel(seed, arm, pop):
        return [r for r in recs if r["seed"] == seed
                and r["arm"] == arm and r["pop"] == pop]

    # L legal hard gate
    for (seed, arm) in CKPTS:
        for pop in ("A", "B"):
            rs = sel(seed, arm, pop)
            gate(len(rs) == 96 and all(
                r["legal_action"] for r in rs),
                f"L LEGAL 96/96 {seed} {arm} {pop}")

    # CONTRAST 1 + 2 (strict heldout, 8 applications)
    c1, c2 = {}, {}
    for (seed, arm) in CKPTS:
        rs = sel(seed, arm, "B")
        Lv = {r["block_id"]: r["teacher_match"] for r in rs}
        Uv = {r["block_id"]:
              bool(u[(seed, arm, "B", r["block_id"])]
                   ["teacher_match"]) for r in rs}
        Rv = {r["block_id"]: rv_h[(seed, arm, r["block_id"])]
              for r in rs}
        u_only = sum(1 for b in Lv if Uv[b] and not Lv[b])
        l_only = sum(1 for b in Lv if Lv[b] and not Uv[b])
        gate(u_only == 0, f"U-ONLY NONZERO {seed} {arm}")
        p1 = binom_minlik_p(u_only, u_only + l_only)
        if l_only > u_only and p1 < 0.05:
            v1 = "LEGAL-CONSTRAINT-HELPS"
        elif u_only > l_only and p1 < 0.05:
            v1 = "LEGAL-CONSTRAINT-HURTS"
        else:
            v1 = "NO-DIRECTIONAL-SEPARATION"
        c1[f"{seed}|{arm}"] = {
            "verdict": v1,
            "U_96": sum(Uv.values()),
            "L_96": sum(Lv.values()),
            "U_only": u_only, "L_only": l_only,
            "both_correct": sum(1 for b in Lv
                                if Uv[b] and Lv[b]),
            "both_wrong": sum(1 for b in Lv
                              if not Uv[b] and not Lv[b]),
            "mcnemar_p_two_sided": p1}
        l_only2 = sum(1 for b in Lv if Lv[b] and not Rv[b])
        r_only2 = sum(1 for b in Lv if Rv[b] and not Lv[b])
        p2 = binom_minlik_p(l_only2, l_only2 + r_only2)
        if r_only2 > l_only2 and p2 < 0.05:
            v2 = "GLOBAL-RANKING-ADVANTAGE"
        elif l_only2 > r_only2 and p2 < 0.05:
            v2 = "LOCAL-GREEDY-ADVANTAGE"
        else:
            v2 = "NO-DIRECTIONAL-SEPARATION"
        c2[f"{seed}|{arm}"] = {
            "verdict": v2,
            "L_96": sum(Lv.values()),
            "R_96": sum(Rv.values()),
            "L_only": l_only2, "R_only": r_only2,
            "both_correct": sum(1 for b in Lv
                                if Lv[b] and Rv[b]),
            "both_wrong": sum(1 for b in Lv
                              if not Lv[b] and not Rv[b]),
            "mcnemar_p_two_sided": p2}

    # frozen map per checkpoint
    cells = {}
    for k in c1:
        helps = c1[k]["verdict"] == "LEGAL-CONSTRAINT-HELPS"
        gra = c2[k]["verdict"] == "GLOBAL-RANKING-ADVANTAGE"
        lga = c2[k]["verdict"] == "LOCAL-GREEDY-ADVANTAGE"
        if helps and not gra:
            cells[k] = "A: LEGALITY-SUPPORT DOMINANT"
        elif helps and gra:
            cells[k] = "B: TWO-STAGE GAP"
        elif not helps and gra:
            cells[k] = "C: LOCAL-v-GLOBAL DOMINANT"
        else:
            cells[k] = ("D: COHERENCE PROBLEM "
                        "(U ~ L ~ R v the booked record)")
        if lga:
            cells[k] += " + LOCAL-GREEDY-ADVANTAGE (named)"

    receipt = {
        "prereg": "MATH-CYBER-1-SVP-DECODE-CONSTRAINT-"
                  "LADDER-PREREG-0",
        "prereg_commit": "3b5f1742",
        "raw_sha": raw_sha,
        "n_rows": len(recs), "n_teacher_forced": len(tf_rows),
        "u_coherence": {
            pop: {"comparable": coh[pop][0],
                  "mismatches": coh[pop][1],
                  "fraction": round(
                      coh[pop][1] / coh[pop][0], 6)
                  if coh[pop][0] else 0.0}
            for pop in ("A", "B")},
        "contrast1_U_v_L": c1,
        "contrast2_L_v_R": c2,
        "interpretive_cells": cells,
        "qual_sha": fsha(QUAL),
        "wall_s": round(time.time() - t0, 1),
        "device": "mps", "torch": torch.__version__,
        "ckpt_pins": {f"{s}|{a}": fsha(p)
                      for (s, a), (p, h) in CKPTS.items()},
        "pins": {p: fsha(p) for p in PINS},
        "start": START,
        "completion_commit": completion_commit()}
    for k, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"POST CKPT {k}")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"POST PIN {p}")
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({"contrast1": c1, "contrast2": c2,
                      "cells": cells,
                      "u_coherence": receipt["u_coherence"]},
                     indent=1), flush=True)

    # riders
    def med(vals):
        sv = sorted(vals)
        return sv[len(sv) // 2] if sv else None

    cal_ladder = {}
    for (seed, arm) in CKPTS:
        rs = sel(seed, arm, "A")
        udesc = sum(
            bool(u[(seed, arm, "A", r["block_id"])]
                 ["teacher_match"]) for r in rs)
        first_div = []
        tuniq = 0
        for r in rs:
            for st in r["steps"]:
                if st["masked_out_top"]:
                    first_div.append(st["allowed_size"])
                    break
            tuniq += sum(1 for st in r["steps"]
                         if st["teacher_unique_allowed"])
        cal_ladder[f"{seed}|{arm}"] = {
            "U": udesc,
            "L": sum(r["teacher_match"] for r in rs),
            "R": sum(1 for r in rs
                     if rv_c[(seed, arm, r["block_id"])]),
            "L_legal": sum(r["legal_action"] for r in rs),
            "teacher_unique_steps": tuniq,
            "median_allowed_at_first_divergence":
                med(first_div)}

    anatomy = {}
    recovery = {}
    for (seed, arm) in CKPTS:
        rs = sel(seed, arm, "B")
        wpos = WITHHELD[arm]
        by_t = defaultdict(list)
        popB = {r["block_id"]: r for r in pops["B"]}
        for r in rs:
            by_t[popB[r["block_id"]]["term_cell"]].append(r)
        atab = {}
        for tcell, rows in sorted(by_t.items()):
            st_at = [r["steps"][wpos - 1] for r in rows]
            # prefix divergence v teacher serialization
            ndiv = 0
            for r in rows:
                tr = popB[r["block_id"]]
                tseq = serialize(
                    arm, [c["factor_code"]
                          for c in tr["candidates"]
                          if c["is_label"]][0])
                if r["gen_tokens"][:wpos - 1] \
                        != tseq[:wpos - 1]:
                    ndiv += 1
            atab[f"t{tcell}"] = {
                "N": len(rows),
                "n_prefix_divergent": ndiv,
                "unmasked_top_allowed": sum(
                    1 for s in st_at
                    if not s["masked_out_top"]),
                "allowed_size_hist": dict(Counter(
                    s["allowed_size"] for s in st_at)),
                "teacher_rank_allowed_hist": dict(Counter(
                    str(s["teacher_rank_allowed"])
                    for s in st_at)),
                "teacher_unique": sum(
                    1 for s in st_at
                    if s["teacher_unique_allowed"]),
                "selected_eq_teacher": sum(
                    1 for s in st_at
                    if s["selected"] == s["teacher_tok"]),
                "leaves_before_hist": dict(Counter(
                    s["leaves_before"] for s in st_at)),
                "leaves_after_hist": dict(Counter(
                    s["leaves_after"] for s in st_at))}
        anatomy[f"{seed}|{arm}"] = atab
        # recovery classification (teacher-match successes)
        cls = Counter()
        for r in rs:
            if not r["teacher_match"]:
                continue
            st = r["steps"][wpos - 1]
            if st["leaves_before"] == 1:
                cls["PREVIOUS_PREFIX_DECIDED"] += 1
            elif st["allowed_size"] == 1:
                cls["FORCED_TEACHER"] += 1
            elif st["selected"] == st["teacher_tok"]:
                cls["CHOSE_TEACHER"] += 1
            else:
                cls["OTHER"] += 1
        recovery[f"{seed}|{arm}"] = dict(cls)

    # registered gap source: ranking raws' gold rows
    gold_lp = {}
    for seed in SEEDS:
        for line in open(RANK_RAW[seed]):
            row = json.loads(line)
            gold_lp[(seed, row["arm"], row["block_id"])] = \
                row["token_lps"][row["label_index"]]
    tf_by = defaultdict(list)
    for row in tf_rows:
        tf_by[(row["seed"], row["arm"])].append(row)
    tfr = {}
    for (seed, arm), rows in tf_by.items():
        ug = []
        ug_tf = []
        for row in rows:
            urow = u[(seed, arm, "B", row["block_id"])]
            wpos = WITHHELD[arm]
            ug.append(urow["chosen_lp"][wpos - 1]
                      - gold_lp[(seed, arm,
                                 row["block_id"])][wpos - 1])
            ug_tf.append(urow["chosen_lp"][wpos - 1]
                         - row["teacher_lp"])
        tfr[f"{seed}|{arm}"] = {
            "teacher_rank_full_hist": dict(Counter(
                r["teacher_rank_full"] for r in rows)),
            "teacher_rank_atoms_hist": dict(Counter(
                r["teacher_rank_atoms"] for r in rows)),
            "median_teacher_lp": round(med(
                [r["teacher_lp"] for r in rows]), 4),
            "median_top_lp": round(med(
                [r["top_lp"] for r in rows]), 4),
            "median_u_chosen_minus_teacher_approx":
                round(med(ug), 4),
            "median_u_chosen_minus_teacher_tf_secondary":
                round(med(ug_tf), 4),
            "median_mass_covered_pi01": round(med(
                [r["mass_covered_pi01"] for r in rows]), 6),
            "median_mass_heldout_pi23": round(med(
                [r["mass_heldout_pi23"] for r in rows]), 6)}

    RIDERS.write_text(json.dumps(
        {"calibration_ladder_descriptive": cal_ladder,
         "withheld_digit_anatomy": anatomy,
         "recovery_classification": recovery,
         "teacher_forced_softmax": tfr,
         "u_exclusion": qr["u_exclusion"],
         "support_matrix": qr["support_matrix"]},
        indent=1))
    print(json.dumps({"recovery": recovery,
                      "cal_ladder": cal_ladder}, indent=1),
          flush=True)
    print("[svpdcl:run] DONE", flush=True)
    return 0


def main():
    q = os.environ.get("SVPDCL_QUAL") == "1"
    r = os.environ.get("SVPDCL_RUN") == "1"
    gate(q != r, "EXACTLY ONE MODE")
    return qual_main() if q else run_main()


if __name__ == "__main__":
    sys.exit(main())
