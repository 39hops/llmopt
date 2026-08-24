"""MATH-CYBER-1 ACTION-OPCODE-QUAL-0 — ActionGCTok: a tokenizer
EXTENSION (never a mutation) of the frozen GCTok, plus its
qualification. Qualifies that the real tokenizer implements the
already-booked ACTION-BASIS-v2 opcode counterfactual without
legacy drift; the predicted lengths are NOT a new discovery.

EXTENSION LAW (frozen): legacy ids 0..295 keep byte-for-byte
identical meaning — base atoms 0..39 unchanged, byte fallback
stays ids 40..295 (n_base untouched). New RESERVED rule opcodes
"<r:{rule_name}>" append AFTER the 296-token legacy vocabulary,
ids 296+, in the standing engine order CORE_RULES + MACRO_RULES +
INT_RULES + LIM_RULES + ALGEBRA_MOVES. Reserved atoms match as
explicit literal strings (longest-first alongside base atoms);
raw rule-name substrings are NEVER retokenized.

v5 serialization = the frozen v2-desk serialization with the rule
name replaced by its reserved atom:
  algebra                      "<r:{rule}>\n"
  sited branch-deterministic   "<r:{rule}> {kind}{ordinal}\n"
  i_parts                      "<r:i_parts> {kind}{ordinal} u{k}\n"
  fenced i_unprod branch>0     "<r:i_unprod> {kind}{ordinal} b{k}\n"

BARS (frozen in the prereg): LEGACY-ID, LEGACY-ENCODING (old and
new tokenizers token-identical over the FULL theta0 birth diet
cur/nxt strings, shard shas pinned, plus every frozen MathWorld
parent/child string), OPCODE (each corpus rule = one unique new
token, zero collision/shadow), PROGRAM (all 725 qualified v4
programs serialize + tokenizer-round-trip exactly), and
COUNTERFACTUAL-REPRO (program med/p90/max == 5/8/8,
within-decision span p90 == 6, zero action-induced 512 overflow).

Receipt: logs/mathworld1/actiontok_qual.json (refuse-if-exists).
Zero model, zero training, zero fresh seeds, zero search.

    .venv/bin/python scratch/mathworld1_actiontok.py          (Mac)
"""
import glob
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

import llmopt.search.derivation as derivation  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.search.derivation import (ALGEBRA_MOVES,  # noqa: E402
                                      State, successors)
from llmopt.search.rules import (CORE_RULES, INT_RULES,  # noqa: E402
                                 LIM_RULES, MACRO_RULES)
from scratch.mathworld1_actionsem import (RULE_KIND,  # noqa: E402
                                          apply_at, iparts_children,
                                          sites_preorder)
from scratch.mathworld1_birth import GCTok  # noqa: E402
from scratch.mathworld1_srepr_export import srepr_inverse  # noqa: E402

OUT = Path("logs/mathworld1/actiontok_qual.json")

OPCODE_ORDER = ([n for n, _ in CORE_RULES]
                + [n for n, _ in MACRO_RULES]
                + [n for n, _ in INT_RULES]
                + [n for n, _ in LIM_RULES]
                + [n for n, _ in ALGEBRA_MOVES])


class ActionGCTok(GCTok):
    """GCTok + reserved rule opcodes appended after id 295."""

    def __init__(self):
        super().__init__()
        assert self.vocab_size == 296 and self.n_base == 40
        self.reserved = [f"<r:{n}>" for n in OPCODE_ORDER]
        self.res_id = {t: 296 + i
                       for i, t in enumerate(self.reserved)}
        self.vocab_size = 296 + len(self.reserved)
        # reserved atoms join the longest-first matcher; base-atom
        # ids and byte fallback (self.n_base + b) stay untouched
        self._by_len = sorted(
            self._by_len + self.reserved, key=len, reverse=True)

    def encode(self, s: str) -> list[int]:
        out, i = [], 0
        while i < len(s):
            for t in self._by_len:
                if s.startswith(t, i):
                    out.append(self.res_id.get(t, self.id.get(t)))
                    i += len(t)
                    break
            else:
                out.extend(self.n_base + b for b in s[i].encode())
                i += 1
        return out

    def decode(self, ids: list[int]) -> str:
        parts, buf = [], []
        inv = {v: k for k, v in self.res_id.items()}
        for i in ids:
            if 40 <= i < 296:
                buf.append(i - 40)
                continue
            if buf:
                parts.append(bytes(buf).decode())
                buf = []
            if i in inv:
                parts.append(inv[i])
            else:
                parts.append(self.atoms[i])
        if buf:
            parts.append(bytes(buf).decode())
        return "".join(parts)


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def fsha(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(
        ["scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_birth.py",
         "scratch/mathworld1_actionsem.py",
         "scratch/mathworld1_srepr_export.py",
         "llmopt/search/derivation.py", "llmopt/search/rules.py"])
    old, new = GCTok(), ActionGCTok()

    # BAR LEGACY-ID: ids 0..295 identical meaning
    legacy_id = (new.atoms[:40] == old.atoms
                 and new.n_base == old.n_base == 40
                 and all(v >= 296 for v in new.res_id.values())
                 and len(set(new.res_id.values()))
                 == len(new.res_id))

    # BAR LEGACY-ENCODING: token-identical over the full birth
    # diet + frozen MathWorld parent/child strings
    diet_files = sorted(
        glob.glob("data/micromodel_chains_shard*.jsonl"))
    diet_files.append("data/step_chains.jsonl")
    diet_shas = {f: fsha(f) for f in diet_files}
    n_diet_strings = 0
    legacy_mismatch = 0
    for f in diet_files:
        for line in open(f):
            r = json.loads(line)
            for s in (r["cur"], r["nxt"]):
                n_diet_strings += 1
                if old.encode(s) != new.encode(s):
                    legacy_mismatch += 1
    mw_strings = 0
    for l in open("logs/mathworld1/states.jsonl"):
        r = json.loads(l)
        for s in (r["state_before"], r["state_after"]):
            mw_strings += 1
            if old.encode(s) != new.encode(s):
                legacy_mismatch += 1
    for l in open("logs/mathworld1/actions.jsonl"):
        r = json.loads(l)
        mw_strings += 1
        if old.encode(r["child"]) != new.encode(r["child"]):
            legacy_mismatch += 1
    legacy_encoding = legacy_mismatch == 0

    # corpus programs: v4 encode, v5 serialization, round-trip,
    # lengths, spans, overflow
    states = {}
    for l in open("logs/mathworld1/states_srepr.jsonl"):
        r = json.loads(l)
        states[(r["episode_id"], r["step_id"])] = r
    v1states = {}
    for l in open("logs/mathworld1/states.jsonl"):
        r = json.loads(l)
        v1states[(r["episode_id"], r["step_id"])] = r
    acts = defaultdict(list)
    for l in open("logs/mathworld1/actions_srepr.jsonl"):
        r = json.loads(l)
        acts[(r["episode_id"], r["step_id"])].append(r)

    corpus_rules = sorted({a["rule"] for aa in acts.values()
                           for a in aa})
    opcode_ok = all(f"<r:{r}>" in new.res_id for r in corpus_rules)

    prog_lens, spans = [], []
    roundtrip_fail = serialize_fail = 0
    action_induced = []
    parent_only = 0
    n_prog = 0
    for key in sorted(acts):
        parent = srepr_inverse(states[key]["state_before"])
        if sha(sp.srepr(parent)) != states[key]["state_before_hash"]:
            raise SystemExit(f"BINDING state_hash {key}")
        derivation._RULE_CACHE.clear()
        gen = sorted(successors(State(parent)),
                     key=lambda nc: (nc[0], nc[1].key()))
        if (Counter(sha(c.key()) for _, c in gen)
                != Counter(a["child_hash"] for a in acts[key])):
            raise SystemExit(f"BINDING legal_set {key}")
        accepted = defaultdict(set)
        for name, c in gen:
            rule = name.split("@", 1)[0] if "@" in name else name
            accepted[rule].add(c.key())
        prefix = len(new.encode(
            f"Current: {v1states[key]['state_before']}"
            f"\nHints: none\nStep: "))
        drow = []
        for a in acts[key]:
            rule = a["rule"]
            kind = RULE_KIND[rule]
            op = f"<r:{rule}>"
            if kind is None:
                ser = f"{op}\n"
            else:
                s4 = sites_preorder(parent, kind)
                hits = []
                for i, cand in enumerate(s4):
                    ck, _ = apply_at(parent, rule, cand)
                    if any(sha(k) == a["child_hash"]
                           for k in set(ck) & accepted[rule]):
                        hits.append((i, cand))
                if not hits:
                    raise SystemExit(f"UNADDRESSABLE {key} {rule}")
                ordinal, node = hits[0]
                if rule == "i_parts":
                    uc_map, _ = iparts_children(parent, node)
                    uc = [u for u, k in uc_map.items()
                          if sha(k) == a["child_hash"]]
                    if len(uc) != 1:
                        raise SystemExit(f"U-AMBIG {key}")
                    ser = f"{op} {kind}{ordinal} u{uc[0]}\n"
                else:
                    bkeys = sorted(
                        set(apply_at(parent, rule, node)[0])
                        & accepted[rule])
                    branch = [i for i, k in enumerate(bkeys)
                              if sha(k) == a["child_hash"]]
                    if len(branch) != 1:
                        raise SystemExit(f"BRANCH {key} {rule}")
                    ser = (f"{op} {kind}{ordinal} b{branch[0]}\n"
                           if branch[0] > 0
                           else f"{op} {kind}{ordinal}\n")
            ids = new.encode(ser)
            if new.decode(ids) != ser:
                roundtrip_fail += 1
            if new.res_id[op] not in ids:
                serialize_fail += 1
            pl = len(ids)
            prog_lens.append(pl)
            drow.append(pl)
            n_prog += 1
        spans.append(max(drow) - min(drow))
        if prefix + 1 > 512:
            parent_only += 1
        elif any(prefix + p > 512 for p in drow):
            action_induced.append(list(key))

    med = statistics.median(prog_lens)
    p90 = pct(prog_lens, 0.9)
    mx = max(prog_lens)
    span90 = pct(spans, 0.9)
    bars = {
        "LEGACY_ID": legacy_id,
        "LEGACY_ENCODING": legacy_encoding,
        "OPCODE": opcode_ok,
        "PROGRAM": (n_prog == 725 and roundtrip_fail == 0
                    and serialize_fail == 0),
        "COUNTERFACTUAL_REPRO": (
            med == 5 and p90 == 8 and mx == 8 and span90 == 6
            and len(action_induced) == 0),
    }
    verdict = {
        "bars": bars,
        "vocab_size_new": new.vocab_size,
        "n_reserved": len(new.reserved),
        "opcode_order_source":
            "CORE+MACRO+INT+LIM+ALGEBRA standing order",
        "corpus_rules": corpus_rules,
        "n_diet_strings": n_diet_strings,
        "n_mathworld_strings": mw_strings,
        "legacy_mismatches": legacy_mismatch,
        "diet_file_sha256": diet_shas,
        "programs": n_prog,
        "roundtrip_fail": roundtrip_fail,
        "serialize_fail": serialize_fail,
        "program_len": {"med": med, "p90": p90, "max": mx},
        "decision_span_p90": span90,
        "action_induced_512": action_induced,
        "parent_only_512": parent_only,
        "start": START, "completion_commit": completion_commit()}
    OUT.write_text(json.dumps(verdict, indent=1))
    print(json.dumps({k: v for k, v in verdict.items()
                      if k not in ("start", "diet_file_sha256")},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
