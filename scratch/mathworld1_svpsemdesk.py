"""MATH-CYBER-1 SVP-SEMANTIC-SUPPORT-QUAL-0 — semantic-factor
support census amending the marginal-field convention of
SVP-SUPPORT-OPPORTUNITY-DESK-0. Support-analysis ontology only:
R = rule, S = (site_kind, site_ordinal), P = (param_kind,
param_index); the trained FACTOR encoding is untouched. No model
loaded, no scores consumed, no training, no new task generation.

Frozen exclusive first-cause precedence:
  WHOLE-COVERED > NEW-RULE > NEW-SITE > NEW-PARAMETER >
  STRICT-WITHIN-RULE-RECOMBINATION > CROSS-RULE-PARAM-RECOMBINATION
Nonexclusive flags rs_seen/rp_seen and the prior desk's marginal
five-field class carried alongside for provenance. Also
enumerates every latent challenge cell: whole tuple absent from
training with (R,S) covered AND (R,P) covered (algebraically
available support cells only — engine constructibility and
teacher selection are NOT determined here).

Census covers PRIMARY-ELIGIBLE decisions only (72/79/69),
matching the frozen n_primary of every booked verdict on these
bands.

Outputs logs/mathworld1/svpsemdesk_receipt.json (refuse-if-
exists).

    .venv/bin/python scratch/mathworld1_svpsemdesk.py         (Mac)
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from scratch.mathworld1_svpbirth import gate  # noqa: E402

PINS = {
    "data/matsub_paired.jsonl":
        "a943ba7fc581db743b07192e5d951fadddd2ba19bca3225b75"
        "d8402351d468e8",
    "logs/mathworld1/svpeval/decisions.jsonl":
        "f63100a62f3091d544750d679483009a473261c587f3165241"
        "406a86253858c6",
    "logs/mathworld1/svpeval2/decisions.jsonl":
        "89efbe0ea447ee937c0c130d5419112921a2dd6c2159c6c211"
        "2cfd5e92f79315",
    "logs/mathworld1/svpeval3/decisions.jsonl":
        "2ff5433249622df9d421cf8014131b3907092a943040bb7b20f4"
        "6f1afffb7efa",
}
BANDS = {
    "band1": ("logs/mathworld1/svpeval/decisions.jsonl", 72),
    "band2": ("logs/mathworld1/svpeval2/decisions.jsonl", 79),
    "band3": ("logs/mathworld1/svpeval3/decisions.jsonl", 69),
}
FIELDS = ("rule", "site_kind", "site_ordinal", "param_kind",
          "param_index")
CLASSES = ("WHOLE-COVERED", "NEW-RULE", "NEW-SITE",
           "NEW-PARAMETER", "STRICT-WITHIN-RULE-RECOMBINATION",
           "CROSS-RULE-PARAM-RECOMBINATION")
RECEIPT = Path("logs/mathworld1/svpsemdesk_receipt.json")


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tup(d):
    return tuple(d[f] for f in FIELDS)


def rsp(t):
    return t[0], (t[1], t[2]), (t[3], t[4])


class Support:
    def __init__(self):
        self.whole = set()
        self.rules = set()
        self.sites = set()
        self.params = set()
        self.rs = set()
        self.rp = set()
        # prior desk's marginal per-field value sets, for the
        # carried-alongside provenance class
        self.marg = {f: set() for f in FIELDS}

    def add(self, t):
        r, s, p = rsp(t)
        self.whole.add(t)
        self.rules.add(r)
        self.sites.add(s)
        self.params.add(p)
        self.rs.add((r, s))
        self.rp.add((r, p))
        for f, v in zip(FIELDS, t):
            self.marg[f].add(v)

    def semantic_class(self, t):
        r, s, p = rsp(t)
        if t in self.whole:
            return "WHOLE-COVERED"
        if r not in self.rules:
            return "NEW-RULE"
        if s not in self.sites:
            return "NEW-SITE"
        if p not in self.params:
            return "NEW-PARAMETER"
        if (r, s) in self.rs and (r, p) in self.rp:
            return "STRICT-WITHIN-RULE-RECOMBINATION"
        return "CROSS-RULE-PARAM-RECOMBINATION"

    def marginal_class(self, t):
        # the prior desk's frozen precedence, carried verbatim
        if t in self.whole:
            return "WHOLE-COVERED"
        if t[0] not in self.marg["rule"]:
            return "NEW-RULE"
        for f, v in zip(FIELDS[1:], t[1:]):
            if v not in self.marg[f]:
                return "NEW-FACTOR-VALUE"
        return "RECOMBINATION-OOV"

    def flags(self, t):
        r, s, p = rsp(t)
        return {"rs_seen": (r, s) in self.rs,
                "rp_seen": (r, p) in self.rp,
                "s_marginal_seen": s in self.sites,
                "p_marginal_seen": p in self.params}


def tkey(t):
    return " ".join(map(str, t))


def band_census(path, n_expected, sup):
    lab = Counter()
    riv = Counter()
    dec = {c: 0 for c in CLASSES}
    cross = Counter()
    strict_records = []
    noncovered = []
    n_prim = 0
    for l in open(path):
        r = json.loads(l)
        if not r.get("primary_eligible"):
            continue
        n_prim += 1
        seen_here = set()
        for c in r["candidates"]:
            t = tup(c)
            k = sup.semantic_class(t)
            seen_here.add(k)
            (lab if c["is_label"] else riv)[k] += 1
            cross[(k, sup.marginal_class(t))] += 1
            if k == "STRICT-WITHIN-RULE-RECOMBINATION":
                strict_records.append({
                    "episode_id": r["episode_id"],
                    "decision_index": r["decision_index"],
                    "is_label": c["is_label"],
                    "tuple": tkey(t)})
            if k != "WHOLE-COVERED":
                noncovered.append(
                    {"tuple": tkey(t), "class": k,
                     "is_label": c["is_label"],
                     **sup.flags(t)})
        for k in seen_here:
            dec[k] += 1
    gate(n_prim == n_expected, f"PRIMARY N {n_prim}")
    return {
        "n_primary": n_prim,
        "labels": dict(lab),
        "rivals": dict(riv),
        "decisions_containing": {k: v for k, v in dec.items()
                                 if v},
        "semantic_x_marginal_crosstab": {
            f"{a} | {b}": n for (a, b), n in cross.most_common()},
        "strict_recombination_records": strict_records,
        "noncovered_actions": noncovered}


def main():
    if RECEIPT.exists():
        raise SystemExit(f"REFUSING: {RECEIPT} exists")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN MISMATCH {p}")
    START = start_provenance(
        ["scratch/mathworld1_svpsemdesk.py",
         "scratch/mathworld1_svpbirth.py",
         "llmopt/lab/provenance.py"])
    sup = Support()
    n_rows = 0
    per_rule_sites = {}
    per_rule_params = {}
    for l in open("data/matsub_paired.jsonl"):
        r = json.loads(l)
        n_rows += 1
        t = tup(r)
        sup.add(t)
        rr, s, p = rsp(t)
        per_rule_sites.setdefault(rr, set()).add(s)
        per_rule_params.setdefault(rr, set()).add(p)
    gate(n_rows == 73324, f"TRAIN ROWS {n_rows}")
    # latent challenge cells: per-rule product grid minus seen
    latent = []
    for rr in sorted(per_rule_sites):
        for s in sorted(per_rule_sites[rr]):
            for p in sorted(per_rule_params[rr],
                            key=lambda x: (x[0], x[1])):
                t = (rr, s[0], s[1], p[0], p[1])
                if t not in sup.whole and (
                        rr, s) in sup.rs and (rr, p) in sup.rp:
                    latent.append(tkey(t))
    gate(all((sup.semantic_class((*x,)) ==
              "STRICT-WITHIN-RULE-RECOMBINATION")
             for x in [tuple(
                 int(v) if v.lstrip("-").isdigit() else v
                 for v in c.split()) for c in latent]),
         "LATENT CELLS NOT ALL STRICT-CLASS")
    receipt = {
        "frozen_precedence":
            "WHOLE-COVERED > NEW-RULE > NEW-SITE > NEW-PARAMETER "
            "> STRICT-WITHIN-RULE-RECOMBINATION > "
            "CROSS-RULE-PARAM-RECOMBINATION; R=rule, "
            "S=(site_kind,site_ordinal), P=(param_kind,"
            "param_index); support = the 73,324 training targets; "
            "prior marginal five-field class carried alongside "
            "as provenance; population = primary-eligible "
            "decisions only (72/79/69)",
        "training": {
            "rows": n_rows,
            "whole_support": sorted(tkey(t) for t in sup.whole),
            "n_whole": len(sup.whole),
            "n_rules": len(sup.rules),
            "n_sites": len(sup.sites),
            "n_params": len(sup.params),
            "n_rule_site_pairs": len(sup.rs),
            "n_rule_param_pairs": len(sup.rp),
            "per_rule_grid": {
                rr: {"sites": sorted(map(str,
                                         per_rule_sites[rr])),
                     "params": sorted(map(str,
                                          per_rule_params[rr])),
                     "grid": len(per_rule_sites[rr]) * len(
                         per_rule_params[rr]),
                     "seen": sum(1 for t in sup.whole
                                 if t[0] == rr)}
                for rr in sorted(per_rule_sites)},
            "latent_strict_cells": latent,
            "n_latent_strict_cells": len(latent)},
        "bands": {
            name: band_census(path, n, sup)
            for name, (path, n) in BANDS.items()},
        "pins": {p: fsha(p) for p in PINS},
        "start": START, "completion_commit": completion_commit()}
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    out = json.loads(json.dumps(receipt))
    out.pop("start"); out.pop("pins")
    out["training"].pop("whole_support")
    for b in out["bands"].values():
        b["noncovered_actions"] = (
            f"{len(b['noncovered_actions'])} rows (full in "
            "receipt)")
    print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
