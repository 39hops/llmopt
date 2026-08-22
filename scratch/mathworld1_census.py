"""MATH-CYBER-1 interface census (AMENDMENT MATH-CYBER-1-DESK-0
-INTERFACE): full L4-7 pass over every legal action recorded in the
frozen rung-0 ACTIVE receipts. For each of the 101 decisions the
walk re-enumerates the legal set (asserting legal_action_set_hash
equality row-by-row against logs/mathworld0/active.jsonl — a
binding check, abort on mismatch) and, for every (parent, child)
action, measures under the stock MathTokenizer (base ATOMS vocab,
strict mode):

  - encodability of the FULL scoring sequence
      "Current: {parent}\\nHints: none\\nStep: {child}\\n"
    (the registered static-probe prompt), with the failing char
    class recorded on failure;
  - the sequence token length when encodable;
  - conservative-fallback coverage: a decision is MODEL-CONTROLLED
    only when the parent and ALL K children encode; an episode is
    fully model-controlled when all its decisions are.

Read-only over frozen evidence; writes one receipt:
logs/mathworld1/census.jsonl (refuse-if-exists).

    .venv/bin/python scratch/mathworld1_census.py             (Mac)
"""
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.mathgen.problems import make_integrate  # noqa: E402
from llmopt.search.derivation import State, successors  # noqa: E402
from llmopt.train.mathnative import MathTokenizer  # noqa: E402

OUT = Path("logs/mathworld1/census.jsonl")
X = sp.Symbol("x")


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/mathworld1_census.py", "scratch/mathworld0.py",
         "llmopt/search/derivation.py", "llmopt/mathgen/problems.py",
         "llmopt/train/mathnative.py"])
    tok = MathTokenizer()

    def enc_len(s):
        try:
            return len(tok.encode(s)), None
        except ValueError as e:
            return None, str(e).split("'")[1]

    rows = [json.loads(line) for line in
            Path("logs/mathworld0/active.jsonl").read_text()
            .splitlines() if "meta" not in line]
    eps = {}
    for r in rows:
        eps.setdefault(r["episode_id"], []).append(r)

    badchars = Counter()
    pair_lens = []
    n_act = n_pair_ok = dec_total = dec_model_ok = ep_model_ok = 0
    t0 = time.monotonic()
    with OUT.open("a") as f:
        for eid, erows in eps.items():
            lv = int(eid[1])
            seed = int(eid.split("-s")[1])
            prob = make_integrate(lv, seed)
            st = State(sp.Integral(prob._expr, X))
            ep_ok, walked = True, False
            for r in erows:
                if not r["chosen_action"]:
                    break
                acts = sorted(successors(st),
                              key=lambda nc: (nc[0], nc[1].key()))
                ah = sha("\n".join(f"{n}|{c.key()}"
                                   for n, c in acts))
                if ah != r["legal_action_set_hash"]:
                    raise SystemExit(
                        f"BINDING MISMATCH {eid} step {r['step_id']}")
                walked = True
                dec_total += 1
                parent = str(st.expr)
                all_ok = enc_len(parent)[1] is None
                for name, c in acts:
                    n_act += 1
                    seq = (f"Current: {parent}\nHints: none\n"
                           f"Step: {str(c.expr)}\n")
                    length, bad = enc_len(seq)
                    if bad:
                        badchars[bad] += 1
                        all_ok = False
                    else:
                        pair_lens.append(length)
                        n_pair_ok += 1
                    f.write(json.dumps({
                        "episode_id": eid, "step_id": r["step_id"],
                        "action": f"{name}#{sha(c.key())}",
                        "pair_seq_tokens": length,
                        "unencodable_char": bad}) + "\n")
                if all_ok:
                    dec_model_ok += 1
                else:
                    ep_ok = False
                match = [nc for nc in acts
                         if f"{nc[0]}#{sha(nc[1].key())}"
                         == r["chosen_action"]]
                if not match:
                    raise SystemExit(
                        f"BINDING MISMATCH (chosen) {eid} "
                        f"step {r['step_id']}")
                st = match[0][1]
            if walked and ep_ok:
                ep_model_ok += 1
        pl = sorted(pair_lens)
        summary = {
            "actions": n_act, "pair_encodable": n_pair_ok,
            "unencodable_char_classes": dict(badchars),
            "pair_seq_tokens": {
                "med": pl[len(pl) // 2],
                "p90": pl[int(0.9 * len(pl))], "max": pl[-1],
                "over_512": sum(1 for x in pl if x > 512)},
            "decisions_model_controlled": dec_model_ok,
            "decisions_total": dec_total,
            "episodes_fully_model_controlled": ep_model_ok,
            "episodes_total": len(eps),
            "wall_s": round(time.monotonic() - t0, 1),
            "start": START,
            "completion_commit": completion_commit()}
        f.write(json.dumps({"meta": summary}) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
