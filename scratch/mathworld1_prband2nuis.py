"""MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-NUISANCE-COUNTERFACTUAL-ASSESSMENT-0
render-only view assessment (model-blind desk).

Reads ONLY the frozen N=96 primary artifact
(logs/mathworld1/prband2prod/primary.jsonl, sha-pinned), the source
candidate law (qualify_parent), the ActionGCTok tokenizer, and SymPy.
Never opens a checkpoint, never reads any score or verifier receipt
(torch arrives transitively through the tokenizer module's imports;
torch.load is replaced by a function that raises, so no weights can
be read by this process).

For each state the six integrand terms are labelled by SEMANTIC ROLE
(HI_D: degree-e_hi trig term; HI_L: degree e_hi-1; LO_D: degree 1;
LO_L: degree 0; K: polynomial; W: inner Integral) and by SOURCE
family (the two terms produced by d/dx[x^e sin(cx)] form the
SIN-source pair, those of d/dx[x^e cos(cx)] the COS-source pair).
One GLOBAL role permutation per view renders the integrand as a
string; the outer structure Integral(<f>, x) + Integral(D, x) is
kept verbatim. Every view is checked for: parse, srepr identity with
the frozen parent, six-term multiset identity, candidate-law
reproduction (qualify_parent on the reparsed object == frozen
cand_sig / teacher / child srepr list), tokenizer round trip, prompt
token count, leading-sign bit, minus position, first trig function,
degree of the first trig term (the exponent-position signature), and
the CTX bound on the view prompt. NEUTRAL geometry = sign AND length
constant across all 96 (not merely a 48/96 ceiling).

Model-blind ceilings: for each view and each surface feature the
best fixed-class mapping feature-value -> theta over the 96 states,
plus its both-correct pair count. Reversal check for the
CUE-FLIPPED candidate against RAW on all 48 matched pairs.

Usage:
    PRBAND2N_SMOKE=1 .venv/bin/python scratch/mathworld1_prband2nuis.py
    .venv/bin/python scratch/mathworld1_prband2nuis.py
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

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from scratch.mathworld1_actiontok import ActionGCTok  # noqa: E402
from scratch.mathworld1_prband import cand_sig, sha  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpchal import (AFTER_D, CTX, X, fsha,  # noqa: E402
                                        qualify_parent)
import torch  # noqa: E402  (transitive via GCTok; trapped below)


def _no_load(*a, **k):
    raise SystemExit("GATE FAILED: torch.load called in a model-blind desk")


torch.load = _no_load

SMOKE = os.environ.get("PRBAND2N_SMOKE") == "1"
ASSESS = "MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-NUISANCE-COUNTERFACTUAL-ASSESSMENT-0"
ADOPTED = {"verdict": "MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-SCORING-0",
           "commit": "1c0d25c338fb6f3f2baf2e7e106208abb88f749b"}
PRIMARY = "logs/mathworld1/prband2prod/primary.jsonl"
PINS = {PRIMARY:
        "209391ef3b2e5c87308571d6ef309bb5724a214160caa1b7857f4a31f9112c34"}
OUTDIR = Path("logs/mathworld1/prband2nuis_smoke" if SMOKE
              else "logs/mathworld1/prband2nuis")
TOK = ActionGCTok()
PROMPT = "Current: {cur}\nHints: none\nStep: "

ROLES = ("HI_D", "HI_L", "LO_D", "LO_L", "K", "W")
# One global role permutation per view. Within a pair the
# degree-e term (coefficient +-c) precedes the degree-(e-1) term
# (coefficient e), which is the order SymPy's printer emits.
VIEWS = {
    "HIGH_PAIR_FIRST": ("HI_D", "HI_L", "K", "LO_D", "LO_L", "W"),
    "HIGH_PAIR_FIRST_LDEG": ("HI_L", "HI_D", "K", "LO_L", "LO_D", "W"),
    "K_FIRST": ("K", "HI_D", "HI_L", "LO_D", "LO_L", "W"),
    "LOW_PAIR_FIRST": ("LO_D", "LO_L", "K", "HI_D", "HI_L", "W"),
    "W_FIRST": ("W", "HI_D", "HI_L", "K", "LO_D", "LO_L"),
}
# Family (source) orderings: the role slot is filled by the
# SIN-source or COS-source pair regardless of which is high.
FAMILY_VIEWS = {
    "COS_SOURCE_FIRST": ("COS_D", "COS_L", "K", "SIN_D", "SIN_L", "W"),
    "SIN_SOURCE_FIRST": ("SIN_D", "SIN_L", "K", "COS_D", "COS_L", "W"),
}


def classify_terms(f):
    """Six integrand terms -> {role: term}, plus family roles."""
    gate(isinstance(f, sp.Add) and len(f.args) == 6, "SIX TERMS")
    roles, fam = {}, {}
    trig = []
    for t in f.args:
        if isinstance(t, sp.Integral):
            gate("W" not in roles, "TWO W")
            roles["W"] = t
        elif not t.has(sp.sin, sp.cos):
            gate("K" not in roles, "TWO K")
            roles["K"] = t
        else:
            fs = list(t.atoms(sp.sin, sp.cos))
            gate(len(fs) == 1, "ONE TRIG")
            poly = sp.cancel(t / fs[0])
            gate(poly.is_polynomial(X), "POLY PART")
            deg = sp.degree(poly, X)
            coeff = sp.Poly(poly, X).LC()
            trig.append((int(deg), coeff, fs[0].func.__name__, t))
    gate(len(trig) == 4, "FOUR TRIG")
    trig.sort(key=lambda z: z[0])
    degs = [z[0] for z in trig]
    gate(degs[0] == 0 and degs[1] == 1 and degs[3] == degs[2] + 1
         and degs[2] >= 4, f"DEGREE BAND {degs}")
    roles["LO_L"], roles["LO_D"] = trig[0][3], trig[1][3]
    roles["HI_L"], roles["HI_D"] = trig[2][3], trig[3][3]
    # source family of a pair = function of its degree-(e-1) term
    # (d/dx[x^e fn(cx)] = e x^(e-1) fn(cx) + c x^e fn'(cx))
    lo_fam, hi_fam = trig[0][2], trig[2][2]
    gate({lo_fam, hi_fam} == {"sin", "cos"}, "FAMILY SPLIT")
    theta = "SIN_LOW" if lo_fam == "sin" else "COS_LOW"
    for pre, fn in (("LO", lo_fam), ("HI", hi_fam)):
        fam[f"{fn.upper()}_D"] = roles[f"{pre}_D"]
        fam[f"{fn.upper()}_L"] = roles[f"{pre}_L"]
    fam["K"], fam["W"] = roles["K"], roles["W"]
    signs = {r: int(sp.sign(sp.Poly(sp.cancel(
        t / list(t.atoms(sp.sin, sp.cos))[0]), X).LC()))
             for r, t in roles.items() if r not in ("K", "W")}
    return roles, fam, theta, signs, degs[3]


def render(terms):
    """Join term strings the way SymPy's printer does: unary minus
    on a leading negative term, binary ' - ' afterwards."""
    out = ""
    for i, t in enumerate(terms):
        s = sp.sstr(t)
        neg = s.startswith("-")
        if i == 0:
            out = s
        elif neg:
            out += " - " + s[1:]
        else:
            out += " + " + s
    return out


def view_string(order, table, D):
    body = render([table[r] for r in order])
    return f"Integral({body}, x) + Integral({sp.sstr(D)}, x)"


def surface(cur_v, order_terms):
    prompt = PROMPT.format(cur=cur_v)
    ids = TOK.encode(prompt)
    gate(TOK.decode(ids) == prompt, "TOK ROUNDTRIP")
    body = cur_v[len("Integral("):]
    lead = "-" if body.startswith("-") else "+"
    minus_pos = None
    first_trig = None
    for i, t in enumerate(order_terms):
        s = sp.sstr(t)
        if minus_pos is None and s.startswith("-"):
            minus_pos = i
        if first_trig is None and t.has(sp.sin, sp.cos) \
                and not isinstance(t, sp.Integral):
            first_trig = list(t.atoms(sp.sin, sp.cos))[0].func.__name__
    first_deg = None
    for t in order_terms:
        if t.has(sp.sin, sp.cos) and not isinstance(t, sp.Integral):
            first_deg = int(sp.degree(sp.cancel(
                t / list(t.atoms(sp.sin, sp.cos))[0]), X))
            break
    gate(len(ids) + 9 <= CTX, "VIEW CONTEXT")
    return {"prompt_tokens": len(ids), "lead_sign": lead,
            "minus_pos": minus_pos, "first_trig": first_trig,
            "first_trig_degree": first_deg}


def check_view(row, parent, f, cur_v):
    """Parse + identity + candidate-law reproduction for one view."""
    p2 = sp.sympify(cur_v)
    ok = {"parse": True,
          "srepr_identical": sp.srepr(p2) == sp.srepr(parent),
          "equiv_zero": sp.simplify(p2 - parent) == 0}
    f2 = None
    for a in p2.args:
        if isinstance(a, sp.Integral) and a.function != AFTER_D:
            f2 = a.function
    ok["six_term_multiset"] = (f2 is not None and isinstance(f2, sp.Add)
                               and Counter(map(sp.srepr, f2.args))
                               == Counter(map(sp.srepr, f.args)))
    q, why = qualify_parent(f2, AFTER_D)
    ok["law_reproduces"] = why is None
    if why is None:
        _js, sid = cand_sig(q["candidates"])
        ok["cand_sig_id_equal"] = sid == row["cand_sig_id"]
        ok["teacher_equal"] = [q["chosen_rule"], q["chosen_site_kind"],
                               q["chosen_ordinal"], q["chosen_param_kind"],
                               q["chosen_term"]] == row["teacher"]
        ok["child_srepr_set_equal"] = (
            sorted(c["child_srepr"] for c in q["candidates"])
            == sorted(c["child_srepr"] for c in row["candidates"]))
        ok["ties_equal"] = q["min_hce_ties"] == row["min_hce_ties"]
        ok["law_cur_is_raw"] = q["cur"] == row["cur"]
    else:
        ok["law_fail"] = why
    return ok


def ceiling(states, feat):
    """Best fixed-class mapping feature-value -> theta: top-1 count
    and both-correct pair count under that mapping."""
    by = defaultdict(Counter)
    for s in states:
        by[s[feat]][s["theta"]] += 1
    mapping = {v: c.most_common(1)[0][0] for v, c in by.items()}
    top1 = sum(c[mapping[v]] for v, c in by.items())
    pairs = defaultdict(dict)
    for s in states:
        pairs[s["pair_id"]][s["theta"]] = mapping[s[feat]] == s["theta"]
    both = sum(1 for p in pairs.values() if all(p.values()))
    return {"values": {str(v): dict(c) for v, c in by.items()},
            "top1_of_96": top1, "both_correct_of_48": both,
            "constant": len(by) == 1}


def main():
    START = start_provenance(
        ["scratch/mathworld1_prband2nuis.py", "scratch/mathworld1_svpchal.py",
         "scratch/mathworld1_prband.py", "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_svpbirth.py", "llmopt/lab/provenance.py"])
    gate(torch.load is _no_load, "TORCH LOAD TRAP")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN {p}")
    t0 = time.monotonic()
    P = [json.loads(l) for l in open(PRIMARY)]
    if SMOKE:
        P = P[:4]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    views_path = OUTDIR / "views.jsonl"
    gate(not views_path.exists(), f"REFUSE OVERWRITE {views_path}")
    all_views = {**VIEWS, **FAMILY_VIEWS}
    per_view = {v: [] for v in ("RAW", *all_views)}
    vf = open(views_path, "w")
    for r in P:
        parent = sp.sympify(r["cur"])
        gate(sp.sstr(parent) == r["cur"], "RAW IS SSTR")
        gate(hashlib.sha256(sp.srepr(parent).encode()).hexdigest()[:16]
             == r["parent_srepr_sha"], "PARENT SREPR PIN")
        f = [a for a in parent.args if isinstance(a, sp.Integral)
             and a.function != AFTER_D][0].function
        roles, fam, theta, signs, e_hi = classify_terms(f)
        gate(theta == r["theta"], "THETA FROM ANATOMY")
        gate(e_hi == r["e_hi"], "E_HI FROM ANATOMY")
        base = {"pair_id": r["pair_id"], "theta": theta,
                "raw_prompt_tokens": r["prompt_tokens"]}
        raw_order = [roles[k] for k in VIEWS["HIGH_PAIR_FIRST"]]
        s_raw = surface(r["cur"], raw_order)
        gate(s_raw["prompt_tokens"] == r["prompt_tokens"], "RAW TOKENS")
        per_view["RAW"].append({**base, **s_raw, "cur": r["cur"]})
        vf.write(json.dumps({**base, "view": "RAW", "cur": r["cur"],
                             **s_raw, "checks": {"raw": True},
                             "signs": signs}) + "\n")
        for vname, order in all_views.items():
            table = fam if vname in FAMILY_VIEWS else roles
            cur_v = view_string(order, table, AFTER_D)
            terms = [table[k] for k in order]
            chk = check_view(r, parent, f, cur_v)
            sv = surface(cur_v, terms)
            rowv = {**base, **sv, "cur": cur_v, "checks": chk,
                    "identical_to_raw": cur_v == r["cur"]}
            per_view[vname].append(rowv)
            vf.write(json.dumps({**rowv, "view": vname}) + "\n")
    vf.close()

    report = {}
    for vname, states in per_view.items():
        checks = Counter()
        for s in states:
            for k, v in s.get("checks", {}).items():
                checks[f"{k}={v}"] += 1
        rep = {"n": len(states), "checks": dict(checks),
               "identical_to_raw": sum(1 for s in states
                                       if s.get("identical_to_raw")),
               "ceilings": {feat: ceiling(states, feat) for feat in
                            ("prompt_tokens", "lead_sign", "minus_pos",
                             "first_trig", "first_trig_degree")}}
        pairs = defaultdict(dict)
        for s in states:
            pairs[s["pair_id"]][s["theta"]] = s
        raw_pairs = defaultdict(dict)
        for s in per_view["RAW"]:
            raw_pairs[s["pair_id"]][s["theta"]] = s
        rev_sign = rev_len = same_abs = 0
        same_sign = same_len = 0
        for pid, p in pairs.items():
            if len(p) != 2:
                continue
            rp = raw_pairs[pid]
            sv = (p["SIN_LOW"]["lead_sign"], p["COS_LOW"]["lead_sign"])
            sr = (rp["SIN_LOW"]["lead_sign"], rp["COS_LOW"]["lead_sign"])
            dv = p["COS_LOW"]["prompt_tokens"] - p["SIN_LOW"]["prompt_tokens"]
            dr = (rp["COS_LOW"]["prompt_tokens"]
                  - rp["SIN_LOW"]["prompt_tokens"])
            rev_sign += sv == sr[::-1] and sv[0] != sv[1]
            same_sign += sv == sr
            rev_len += dv == -dr and dr != 0
            same_len += dv == dr
            same_abs += abs(dv) == abs(dr)
        rep["pair_geometry_v_raw"] = {
            "pairs": sum(1 for p in pairs.values() if len(p) == 2),
            "sign_reversed": rev_sign, "sign_same": same_sign,
            "length_diff_reversed": rev_len, "length_diff_same": same_len,
            "length_abs_diff_equal": same_abs}
        report[vname] = rep

    def neutral_ok(rep):
        c = rep["ceilings"]
        return c["lead_sign"]["constant"] and c["prompt_tokens"]["constant"]

    def flipped_ok(rep):
        g = rep["pair_geometry_v_raw"]
        return (g["sign_reversed"] == g["pairs"]
                and g["length_diff_reversed"] == g["pairs"]
                and g["length_abs_diff_equal"] == g["pairs"])

    verdict = {v: {"all_checks_true": all(
        k.endswith("=True") for k in rep["checks"]) and rep["n"] > 0,
        "neutral_geometry": neutral_ok(rep),
        "cue_flipped_geometry": flipped_ok(rep),
        "minus_pos_constant": rep["ceilings"]["minus_pos"]["constant"],
        "first_trig_constant": rep["ceilings"]["first_trig"]["constant"],
        "first_trig_degree_ceiling": rep["ceilings"]["first_trig_degree"][
            "top1_of_96"]}
        for v, rep in report.items() if v != "RAW"}
    receipt = {"smoke": SMOKE, "assessment": ASSESS, "adopted": ADOPTED,
               "pins": PINS, "n_states": len(P),
               "views": {k: list(v) for k, v in all_views.items()},
               "prompt_law": PROMPT, "report": report, "verdict": verdict,
               "wall_s": round(time.monotonic() - t0, 2),
               "provenance": START,
               "completion_commit": completion_commit(),
               "views_sha": fsha(str(views_path))}
    (OUTDIR / "prband2nuis_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps(verdict, indent=1))
    for v, rep in report.items():
        c = rep["ceilings"]
        print(v, "tok", c["prompt_tokens"]["values"], "sign",
              c["lead_sign"]["values"], "minus", c["minus_pos"]["values"],
              "trig", c["first_trig"]["values"],
              "deg", c["first_trig_degree"]["values"],
              "geom", rep["pair_geometry_v_raw"], "checks", rep["checks"])


if __name__ == "__main__":
    main()
