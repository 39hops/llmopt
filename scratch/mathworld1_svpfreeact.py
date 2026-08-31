"""MATH-CYBER-1 SVP-FREE-ACTION-1 — ONE-STEP greedy free-
generation transport instrument (prereg FREE-ACTION-1-PREREG-0,
commit e87471f6). Two modes:

  SVPFA_QUAL=1   qualification ONLY — no checkpoint touched:
                 all pins; population structure; teacher tuples/
                 legality/serialization roundtrips both arms;
                 teacher REPLAY (engine resolution reproduces the
                 labeled child srepr on all 192 states); ranking
                 reference vectors reproduced from the booked raw
                 artifacts WITHOUT inference (85/78/80/87 heldout,
                 94/93/85/91 calibration); rescue-group
                 memberships RE-DERIVED from the pinned raws under
                 the desk's verbatim famD definitions with EXACT
                 count reproduction v the desk receipt (overlap
                 preserved; groups do not partition 96). Writes
                 qual.json + groups.json.
  SVPFA_RUN=1    ONE joint generation run (requires green qual):
                 four frozen checkpoints x two populations x 96
                 states; greedy argmax over the FULL 340 vocab,
                 exact-tie -> lowest token ID (ties counted);
                 stop at EOS or 9 tokens, EOS never forced; no
                 masks/beam/sampling/retry/repair. Raw per-state
                 generation rows persisted + hashed BEFORE any
                 summary. Funnel: EOS-WELL-FORMED -> SYNTAX-VALID
                 -> SEMANTIC-DECODE (CAUGHT predicate — decode
                 SystemExit/exception records 0, run continues)
                 -> LEGAL-ACTION; TEACHER-MATCH (primary, needs
                 level 3); REPLAY-VALID (engine resolution;
                 descriptive). Endpoints: per-checkpoint paired
                 ranking-v-generation exact McNemar on heldout
                 (RANKING-ADVANTAGE / GENERATION-ADVANTAGE /
                 NO-DIRECTIONAL-SEPARATION); calibration funnel
                 rider (descriptive, no verdict); field-order
                 rider per seed x population; first_error
                 histograms; rescue-anatomy rider (population B
                 only).

Outputs under logs/mathworld1/svpfreeact/ (refuse-if-exists per
file): qual.json, groups.json, raw_generations.jsonl,
receipt.json, riders.json.

    SVPFA_QUAL=1 .venv/bin/python scratch/mathworld1_svpfreeact.py
    SVPFA_RUN=1  .venv/bin/python scratch/mathworld1_svpfreeact.py
"""
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

import llmopt.search.derivation as derivation  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.search.derivation import State, successors  # noqa: E402
from scratch.mathworld1_actionfinal import (  # noqa: E402
    unprod_term_children)
from scratch.mathworld1_actionsem import (apply_at,  # noqa: E402
                                          iparts_children,
                                          sites_preorder)
from scratch.mathworld1_actiontok import ActionGCTok  # noqa: E402
from scratch.mathworld1_svpadj import mcnemar_exact  # noqa: E402
from scratch.mathworld1_svpcode import (factor_decode,  # noqa: E402
                                        factor_symbols, in_domain)
from scratch.mathworld1_svpforder import (INV, PERM,  # noqa: E402
                                          pf_decode, pf_encode)

CAL = "logs/mathworld1/svpdiet3/covered_calibration.jsonl"
HELD = "logs/mathworld1/svpdiet3/heldout_test16.jsonl"
PINS = {
    CAL: ("af1a4aa1df7bf3224745e91a90e1a77c36e5c54f7ff9b085"
          "09794d0fb7978db3"),
    HELD: ("a3f6103b3733d909281849dcb3fd6ba9fba3891f2014bec1"
           "3881b4509df46ddb"),
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
    "logs/mathworld1/svpfbcredit/desk.json":
        ("d190cf0c3fb9fac1bcf40a614b15e46478c8d8ff911e10619e"
         "b381ad0a9ca916"),
    "logs/mathworld1/svpforder_census/census.json":
        ("7d8343b3f328b00a1147b675441181e897d3d96eaff33fe33d"
         "403aa482dd71d5"),
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
FIELDS = {"CANONICAL": {"RULE": (1, 2), "SITE": (3, 5),
                        "PARAM": (6, 8), "EOS": (9, 9)},
          "PARAM_FIRST": {"PARAM": (1, 3), "RULE": (4, 5),
                          "SITE": (6, 8), "EOS": (9, 9)}}
OUTDIR = Path("logs/mathworld1/svpfreeact")
QUAL = OUTDIR / "qual.json"
GROUPS = OUTDIR / "groups.json"
RAWGEN = OUTDIR / "raw_generations.jsonl"
RECEIPT = OUTDIR / "receipt.json"
RIDERS = OUTDIR / "riders.json"
TOK = ActionGCTok()
EOS = TOK.eos_id
SRC = ["scratch/mathworld1_svpfreeact.py",
       "scratch/mathworld1_svpforder.py",
       "scratch/mathworld1_svpcode.py",
       "scratch/mathworld1_svpadj.py",
       "scratch/mathworld1_actiontok.py",
       "scratch/mathworld1_actionsem.py",
       "scratch/mathworld1_actionfinal.py",
       "llmopt/search/derivation.py",
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


def teacher_tokens(arm, tup):
    syms = (factor_symbols(*tup) if arm == "CANONICAL"
            else pf_encode(tup))
    return [CODE_BASE + s for s in syms] + [EOS]


def safe_decode(arm, syms):
    """Caught-predicate semantic decode (prereg blocker law):
    any decode failure returns None, never aborts."""
    try:
        tup = (factor_decode(syms) if arm == "CANONICAL"
               else pf_decode(syms))
        if not in_domain(*tup):
            return None
        return tup
    except BaseException:
        return None


def load_pop(which):
    rows = [json.loads(l) for l in open(
        CAL if which == "A" else HELD)]
    if which == "A":
        gate(len(rows) == 96 and all(
            r["site_role"] == "covered-I0" for r in rows),
            "POP A STRUCTURE")
    else:
        rows = [r for r in rows
                if r["site_role"] == "heldout-I1"]
        gate(len(rows) == 96, "POP B STRUCTURE")
    for r in rows:
        labs = [c for c in r["candidates"] if c["is_label"]]
        gate(len(labs) == 1, "LABEL COUNT")
        r["_teacher"] = ctup(labs[0])
        r["_teacher_child_srepr"] = labs[0]["child_srepr"]
        r["_candset"] = {ctup(c) for c in r["candidates"]}
        r["_child_by_tup"] = {ctup(c): c["child_srepr"]
                              for c in r["candidates"]}
        gate(r["_teacher"] in r["_candset"], "TEACHER LEGAL")
        for arm in ARMS:
            tt = teacher_tokens(arm, r["_teacher"])
            gate(len(tt) == 9, "TEACHER T9")
            gate(safe_decode(
                arm, [t - CODE_BASE for t in tt[:8]])
                == r["_teacher"], f"TEACHER RT {arm}")
        cz = labs[0]["factor_code"]
        gate(factor_decode(cz) == r["_teacher"], "TEACHER CZ")
        gate(pf_encode(r["_teacher"])
             == [cz[PERM[i]] for i in range(8)], "TEACHER PERM")
    return rows


def rank_vectors_held():
    """Heldout k=9 pessimistic top-1 per (seed, arm, block) from
    the booked raw token-score artifacts. No inference."""
    out = {}
    for seed in SEEDS:
        rows = [json.loads(l) for l in open(RANK_RAW[seed])]
        gate(len(rows) == 192, f"RANK RAW ROWS {seed}")
        for row in rows:
            cums = []
            for v in row["token_lps"]:
                s = 0.0
                c = []
                for x in v:
                    s += x
                    c.append(s)
                cums.append(c)
            li = row["label_index"]
            g = cums[li][8]
            top = all(cums[j][8] < g for j in range(len(cums))
                      if j != li)
            out[(seed, row["arm"], row["block_id"])] = top
    for (seed, arm), exp in RANK_EXPECT_HELD.items():
        tot = sum(1 for (s, a, b), v in out.items()
                  if s == seed and a == arm and v)
        gate(tot == exp, f"RANK HELD {seed} {arm} {tot}!={exp}")
    return out


def rank_vectors_cal():
    out = {}
    for seed in SEEDS:
        for l in open(CAL_SCORES[seed]):
            row = json.loads(l)
            li = row["label_index"]
            for arm in ARMS:
                m = row[arm]["mean_lp"]
                top = all(m[j] < m[li] for j in range(len(m))
                          if j != li)
                gate(top == bool(row[arm]["top1"]),
                     "CAL TOP1 MISMATCH")
                out[(seed, arm, row["block_id"])] = top
    for (seed, arm), exp in RANK_EXPECT_CAL.items():
        tot = sum(1 for (s, a, b), v in out.items()
                  if s == seed and a == arm and v)
        gate(tot == exp, f"RANK CAL {seed} {arm} {tot}!={exp}")
    return out


def derive_groups():
    """Re-derive rescue-group memberships from the pinned raws
    under the desk's verbatim famD definitions; exact count
    reproduction v the desk receipt is a hard gate."""
    desk = json.loads(
        Path("logs/mathworld1/svpfbcredit/desk.json")
        .read_text())
    census = json.loads(
        Path("logs/mathworld1/svpforder_census/census.json")
        .read_text())
    ps = census["structural"]["CANONICAL"]["per_state"]
    early = sorted(b for b, k in ps.items() if k == 1)
    late = sorted(b for b, k in ps.items() if k == 8)
    gate(len(early) == 29 and len(late) == 67, "GROUP SIZES")
    blocks = {a: {n: (s, e) for n, s, e in desk["blocks"][a]}
              for a in ARMS}
    out = {"EARLY29": early, "RELOCATED67": late, "famD": {}}
    for seed in SEEDS:
        rows = [json.loads(l) for l in open(RANK_RAW[seed])]
        by = {a: {} for a in ARMS}
        for row in rows:
            cums = []
            for v in row["token_lps"]:
                s = 0.0
                c = []
                for x in v:
                    s += x
                    c.append(s)
                cums.append(c)
            li = row["label_index"]
            tp = {}
            for k in range(1, 10):
                g = cums[li][k - 1]
                tp[k] = all(cums[j][k - 1] < g
                            for j in range(len(cums))
                            if j != li)
            by[row["arm"]][row["block_id"]] = tp
        for a in ARMS:
            pe = {n: e for n, (s0, e) in blocks[a].items()}
            pstart = {n: s0 for n, (s0, e) in blocks[a].items()}
            g = defaultdict(list)
            for b, tp in sorted(by[a].items()):
                after_param = tp[pe["PARAM"]]
                before_ks = (range(1, pe["PARAM"])
                             if a == "PARAM_FIRST"
                             else range(1, pstart["PARAM"]))
                was_before = any(tp[k] for k in before_ks)
                if after_param:
                    g["PARAM_CORRECT"].append(b)
                if was_before and not after_param:
                    g["PARAM_DAMAGED"].append(b)
                if a == "PARAM_FIRST":
                    if not after_param and tp[pe["RULE"]]:
                        g["RULE_RESCUED"].append(b)
                    if after_param and not tp[pe["RULE"]]:
                        g["RULE_LOST"].append(b)
                if not after_param and tp[9]:
                    g["FINAL_RESCUED"].append(b)
            booked = desk["families"][str(seed)][a]["D"]["all"]
            derived_counts = {k: len(v) for k, v in g.items()
                              if v}
            gate(derived_counts == booked,
                 f"GROUP COUNTS {seed} {a}: "
                 f"{derived_counts} != {booked}")
            out["famD"][f"{seed}|{a}"] = {k: sorted(v)
                                          for k, v in g.items()}
    return out


def build_replay_cache(pops):
    """One successors() call per (population, state) — block_ids
    COLLIDE across the two populations (all 96 shared, different
    cur), so the cache key carries the population; teacher
    resolution gated on all 192 (pop, state) rows."""
    cache = {}
    checked = 0
    for which, rows in pops.items():
        for r in rows:
            b = (which, r["block_id"])
            if b in cache:
                continue
            parent = sp.sympify(r["cur"])
            derivation._RULE_CACHE.clear()
            gen = list(successors(State(parent)))
            accepted = defaultdict(set)
            for name, ch in gen:
                rule = (name.split("@", 1)[0] if "@" in name
                        else name)
                accepted[rule].add(ch.key())
            cache[b] = (parent, accepted)
            res = resolve(parent, accepted, r["_teacher"])
            gate(res is not None
                 and res == r["_teacher_child_srepr"],
                 f"TEACHER REPLAY {b}")
            checked += 1
    gate(checked == 192, f"TEACHER REPLAYS {checked} != 192")
    return cache, checked


def resolve(parent, accepted, tup):
    """replay_core resolution, verbatim logic; any failure ->
    None (caught; malformed actions are results)."""
    try:
        rule, sk, so, pk_, pi = tup
        node = (None if sk == "W"
                else sites_preorder(parent, sk)[so])
        if pk_ == "u_choice":
            dk = iparts_children(parent, node)[0].get(pi)
            dec = {dk} if dk else set()
        elif pk_ == "term_index":
            tmap, parity = unprod_term_children(parent, node)
            dec = (tmap.get(pi, set()) & accepted[rule]
                   if parity else set())
        else:
            dset = (accepted[rule] if node is None else
                    set(apply_at(parent, rule, node)[0])
                    & accepted[rule])
            dec = dset if len(dset) == 1 else set()
        return next(iter(dec)) if len(dec) == 1 else None
    except BaseException:
        return None


def qual_main():
    for p in (QUAL, GROUPS):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN {p}")
    for k, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"CKPT PIN {k}")
    START = start_provenance(SRC)
    gate(TOK.vocab_size == CODE_BASE == 332 and EOS == 1,
         "ALPHABET LAW")
    gate(all(INV[PERM[i]] == i for i in range(8)), "INV LAW")
    pops = {"A": load_pop("A"), "B": load_pop("B")}
    rv_h = rank_vectors_held()
    rv_c = rank_vectors_cal()
    groups = derive_groups()
    cache, teacher_replays = build_replay_cache(pops)
    receipt = {
        "prereg": "MATH-CYBER-1-SVP-FREE-ACTION-1-PREREG-0",
        "prereg_commit": "e87471f6",
        "verdict": "QUALIFIED",
        "gates": {
            "pins_checked": len(PINS) + len(CKPTS),
            "pop_A_rows": len(pops["A"]),
            "pop_B_rows": len(pops["B"]),
            "teacher_roundtrips_both_arms": 2 * 192,
            "teacher_legal": 192,
            "teacher_replay_valid": teacher_replays,
            "rank_held_reproduced": {f"{s}|{a}": v for (s, a),
                                     v in
                                     RANK_EXPECT_HELD.items()},
            "rank_cal_reproduced": {f"{s}|{a}": v for (s, a),
                                    v in
                                    RANK_EXPECT_CAL.items()},
            "group_counts_match_desk": True},
        "wall_s": None,
        "start": START,
        "completion_commit": completion_commit()}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    GROUPS.write_text(json.dumps(groups, indent=1))
    receipt["groups_sha"] = fsha(GROUPS)
    QUAL.write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt["gates"], indent=1), flush=True)
    print("[svpfreeact:qual] QUALIFIED", flush=True)
    return 0


def run_main():
    import torch
    from llmopt.train.mathnative import build_model
    for p in (RAWGEN, RECEIPT, RIDERS):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    gate(QUAL.exists() and GROUPS.exists(), "QUAL NOT RUN")
    qr = json.loads(QUAL.read_text())
    gate(qr["verdict"] == "QUALIFIED", "NOT QUALIFIED")
    for pth, h in qr["start"]["file_sha256"].items():
        gate(fsha(pth) == h, f"QUAL STALE {pth}")
    gate(fsha(GROUPS) == qr["groups_sha"], "GROUPS MUTATED")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN {p}")
    for k, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"CKPT PIN {k}")
    START = start_provenance(SRC)
    groups = json.loads(GROUPS.read_text())
    pops = {"A": load_pop("A"), "B": load_pop("B")}
    rv_h = rank_vectors_held()
    rv_c = rank_vectors_cal()
    cache, _ = build_replay_cache(pops)
    gate(torch.backends.mps.is_available(), "MPS")
    dev = torch.device("mps")

    t0 = time.time()
    recs = []
    for (seed, arm), (path, csha) in CKPTS.items():
        m = build_model(VOCAB, ctx=4096)
        m.load_state_dict(torch.load(path, weights_only=True))
        gate(sum(p.numel() for p in m.parameters())
             == 19142016, "PARAM COUNT")
        m.eval()
        m = m.to(dev)
        for pop, rows in pops.items():
            for r in rows:
                pre = TOK.encode(f"Current: {r['cur']}\n"
                                 f"Hints: none\nStep: ")
                ids = list(pre)
                gen, lps, tiepos = [], [], []
                for step in range(9):
                    x = torch.tensor([ids], device=dev)
                    with torch.no_grad():
                        logits = m(x)[0, -1].float().cpu()
                    mx = logits.max()
                    tie_ids = (logits == mx).nonzero(
                        as_tuple=True)[0].tolist()
                    tok = min(tie_ids)
                    if len(tie_ids) > 1:
                        tiepos.append(step + 1)
                    lp = torch.log_softmax(logits, -1)[tok]
                    gen.append(int(tok))
                    lps.append(float(lp))
                    ids.append(int(tok))
                    if tok == EOS:
                        break
                eos_pos = (gen.index(EOS) + 1
                           if EOS in gen else None)
                ewf = (len(gen) == 9 and gen[8] == EOS
                       and EOS not in gen[:8])
                syn = ewf and all(
                    CODE_BASE <= t < CODE_BASE + 8
                    for t in gen[:8])
                tup = None
                if syn:
                    tup = safe_decode(
                        arm, [t - CODE_BASE for t in gen[:8]])
                dec = tup is not None
                legal = dec and tup in r["_candset"]
                teach = dec and tup == r["_teacher"]
                replay_binds = False
                replay = False
                replay_child_match = None
                if dec:
                    parent, accepted = cache[(pop,
                                             r["block_id"])]
                    child = resolve(parent, accepted, tup)
                    replay_binds = child is not None
                    replay = replay_binds
                    if replay_binds and legal:
                        replay_child_match = (
                            child == r["_child_by_tup"][tup])
                        replay = replay_child_match
                tt = teacher_tokens(arm, r["_teacher"])
                fe = None
                for t in range(len(gen)):
                    if gen[t] != tt[t]:
                        fe = t + 1
                        break
                if fe is None and len(gen) < 9:
                    fe = len(gen) + 1   # unreachable by law
                recs.append({
                    "seed": seed, "arm": arm, "pop": pop,
                    "block_id": r["block_id"],
                    "ckpt_sha": csha,
                    "prompt_sha": hashlib.sha256(
                        json.dumps(pre).encode()).hexdigest(),
                    "gen_tokens": gen, "n_gen": len(gen),
                    "chosen_lp": [round(x, 6) for x in lps],
                    "tie_positions": tiepos,
                    "n_ties": len(tiepos),
                    "eos_pos": eos_pos,
                    "eos_well_formed": ewf,
                    "syntax_valid": syn,
                    "semantic_decode": dec,
                    "decoded_tuple": (list(tup) if dec
                                      else None),
                    "legal_action": legal,
                    "teacher_match": teach,
                    "legal_nonteacher": legal and not teach,
                    "replay_binds": replay_binds,
                    "replay_valid": replay,
                    "replay_child_match": replay_child_match,
                    "teacher_tuple": list(r["_teacher"]),
                    "first_error_position": fe})
        del m
        print(f"[svpfreeact] {seed} {arm} done "
              f"({time.time() - t0:.0f}s)", flush=True)

    with open(RAWGEN, "w") as fo:
        for rec in recs:
            fo.write(json.dumps(rec) + "\n")
    raw_sha = fsha(RAWGEN)
    gate(len(recs) == 4 * 2 * 96, "REC COUNT")

    def sel(seed, arm, pop):
        return [r for r in recs if r["seed"] == seed
                and r["arm"] == arm and r["pop"] == pop]

    funnel = {}
    ties_total = 0
    for (seed, arm) in CKPTS:
        for pop in ("A", "B"):
            rs = sel(seed, arm, pop)
            gate(len(rs) == 96, "SEL 96")
            funnel[f"{seed}|{arm}|{pop}"] = {
                "eos_well_formed": sum(
                    r["eos_well_formed"] for r in rs),
                "syntax_valid": sum(
                    r["syntax_valid"] for r in rs),
                "semantic_decode": sum(
                    r["semantic_decode"] for r in rs),
                "legal_action": sum(
                    r["legal_action"] for r in rs),
                "teacher_match": sum(
                    r["teacher_match"] for r in rs),
                "legal_nonteacher": sum(
                    r["legal_nonteacher"] for r in rs),
                "replay_valid": sum(
                    r["replay_valid"] for r in rs),
                "n_ties": sum(r["n_ties"] for r in rs)}
            ties_total += funnel[f"{seed}|{arm}|{pop}"][
                "n_ties"]

    # PRIMARY: heldout ranking v generation per checkpoint
    primary = {}
    for (seed, arm) in CKPTS:
        rs = sel(seed, arm, "B")
        cells = Counter()
        for r in rs:
            R = rv_h[(seed, arm, r["block_id"])]
            G = r["teacher_match"]
            cells[(R, G)] += 1
        r_only = cells[(True, False)]
        g_only = cells[(False, True)]
        p = mcnemar_exact(r_only, g_only)
        if r_only > g_only and p < 0.05:
            verdict = "RANKING-ADVANTAGE"
        elif g_only > r_only and p < 0.05:
            verdict = "GENERATION-ADVANTAGE"
        else:
            verdict = "NO-DIRECTIONAL-SEPARATION"
        primary[f"{seed}|{arm}"] = {
            "verdict": verdict,
            "ranking_96": sum(1 for r in rs if rv_h[
                (seed, arm, r["block_id"])]),
            "generation_96": sum(r["teacher_match"]
                                 for r in rs),
            "R_only": r_only, "G_only": g_only,
            "both_correct": cells[(True, True)],
            "both_wrong": cells[(False, False)],
            "mcnemar_p_two_sided": p}

    # calibration rider: descriptive R v G, no verdict label
    cal_rider = {}
    for (seed, arm) in CKPTS:
        rs = sel(seed, arm, "A")
        cells = Counter()
        for r in rs:
            R = rv_c[(seed, arm, r["block_id"])]
            cells[(R, r["teacher_match"])] += 1
        cal_rider[f"{seed}|{arm}"] = {
            "ranking_96": sum(1 for r in rs if rv_c[
                (seed, arm, r["block_id"])]),
            "generation_96": sum(r["teacher_match"]
                                 for r in rs),
            "R_only": cells[(True, False)],
            "G_only": cells[(False, True)]}

    # field-order rider per seed x pop
    forder = {}
    for seed in SEEDS:
        for pop in ("A", "B"):
            byb = {}
            for arm in ARMS:
                byb[arm] = {r["block_id"]: r["teacher_match"]
                            for r in sel(seed, arm, pop)}
            c_only = sum(1 for b in byb["CANONICAL"]
                         if byb["CANONICAL"][b]
                         and not byb["PARAM_FIRST"][b])
            p_only = sum(1 for b in byb["CANONICAL"]
                         if byb["PARAM_FIRST"][b]
                         and not byb["CANONICAL"][b])
            p = mcnemar_exact(c_only, p_only)
            if c_only > p_only and p < 0.05:
                v = "C-ADVANTAGE"
            elif p_only > c_only and p < 0.05:
                v = "PF-ADVANTAGE"
            else:
                v = "NO-DIRECTIONAL-SEPARATION"
            forder[f"{seed}|{pop}"] = {
                "verdict": v,
                "C": sum(byb["CANONICAL"].values()),
                "PF": sum(byb["PARAM_FIRST"].values()),
                "C_only": c_only, "PF_only": p_only,
                "mcnemar_p_two_sided": p}

    # first-error histograms
    fe_hist = {}
    for (seed, arm) in CKPTS:
        for pop in ("A", "B"):
            h = Counter(str(r["first_error_position"])
                        for r in sel(seed, arm, pop))
            fe_hist[f"{seed}|{arm}|{pop}"] = {
                k: h[k] for k in
                ["None"] + [str(i) for i in range(1, 10)]
                if h[k]}

    receipt = {
        "prereg": "MATH-CYBER-1-SVP-FREE-ACTION-1-PREREG-0",
        "prereg_commit": "e87471f6",
        "raw_generations_sha": raw_sha,
        "n_generations": len(recs),
        "total_tie_events": ties_total,
        "funnel": funnel,
        "primary_heldout_transport": primary,
        "calibration_rider_descriptive": cal_rider,
        "field_order_rider": forder,
        "first_error_histograms": fe_hist,
        "fields_map": FIELDS,
        "qual_sha": fsha(QUAL),
        "groups_sha": fsha(GROUPS),
        "wall_s": round(time.time() - t0, 1),
        "device": "mps", "torch": torch.__version__,
        "ckpt_pins": {f"{s}|{a}": fsha(p) for (s, a), (p, h)
                      in CKPTS.items()},
        "pins": {p: fsha(p) for p in PINS},
        "start": START,
        "completion_commit": completion_commit()}
    for k, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"POST CKPT {k}")
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({"primary": primary,
                      "field_order": forder}, indent=1),
          flush=True)

    # rescue-anatomy rider (population B only)
    by_state = {(r["seed"], r["arm"], r["block_id"]): r
                for r in recs if r["pop"] == "B"}
    rid = {}
    for seed in SEEDS:
        for arm in ARMS:
            gm = dict(groups["famD"][f"{seed}|{arm}"])
            gm["EARLY29"] = groups["EARLY29"]
            gm["RELOCATED67"] = groups["RELOCATED67"]
            tab = {}
            for gname, members in gm.items():
                rs = [by_state[(seed, arm, b)]
                      for b in members]
                if not rs:
                    continue
                tab[gname] = {
                    "N": len(rs),
                    "eos_well_formed": sum(
                        r["eos_well_formed"] for r in rs),
                    "syntax_valid": sum(
                        r["syntax_valid"] for r in rs),
                    "semantic_decode": sum(
                        r["semantic_decode"] for r in rs),
                    "legal_action": sum(
                        r["legal_action"] for r in rs),
                    "teacher_match": sum(
                        r["teacher_match"] for r in rs),
                    "legal_nonteacher": sum(
                        r["legal_nonteacher"] for r in rs),
                    "first_error_hist": dict(Counter(
                        str(r["first_error_position"])
                        for r in rs))}
            rid[f"{seed}|{arm}"] = tab
    # central motivated cell: 20001 PF RULE_RESCUED
    rr = groups["famD"]["20001|PARAM_FIRST"].get(
        "RULE_RESCUED", [])
    rrs = [by_state[(20001, "PARAM_FIRST", b)] for b in rr]
    pf_param_end = FIELDS["PARAM_FIRST"]["PARAM"][1]
    central = {
        "group_N": len(rrs),
        "greedy_teacher_match": sum(r["teacher_match"]
                                    for r in rrs),
        "greedy_legal_action": sum(r["legal_action"]
                                   for r in rrs),
        "first_error_in_PARAM_1_3": sum(
            1 for r in rrs
            if r["first_error_position"] is not None
            and r["first_error_position"] <= pf_param_end),
        "first_error_later": sum(
            1 for r in rrs
            if r["first_error_position"] is not None
            and r["first_error_position"] > pf_param_end),
        "no_error": sum(
            1 for r in rrs
            if r["first_error_position"] is None),
        "ranking_correct_and_off_teacher_before_RULE": sum(
            1 for r in rrs
            if rv_h[(20001, "PARAM_FIRST", r["block_id"])]
            and r["first_error_position"] is not None
            and r["first_error_position"] <= pf_param_end)}
    RIDERS.write_text(json.dumps(
        {"rescue_anatomy_popB": rid,
         "central_20001_PF_RULE_RESCUED": central},
        indent=1))
    print(json.dumps({"central_20001_PF_RULE_RESCUED":
                      central}, indent=1), flush=True)
    print("[svpfreeact:run] DONE", flush=True)
    return 0


def main():
    q = os.environ.get("SVPFA_QUAL") == "1"
    r = os.environ.get("SVPFA_RUN") == "1"
    gate(q != r, "EXACTLY ONE MODE")
    return qual_main() if q else run_main()


if __name__ == "__main__":
    sys.exit(main())
