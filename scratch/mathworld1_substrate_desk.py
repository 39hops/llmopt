"""MATH-CYBER-1 fresh-substrate desk (no model loaded, no vocab
shipped): price a GRAMMAR-CLOSED tokenizer — the existing ATOMS
multi-char vocabulary plus a deterministic single-byte fallback
token per uncovered character — against (a) the 725 calibration
actions of the frozen rung-0 trajectories and (b) the standing
MathNative base training diet.

The byte fallback closes the sstr language generically: any
identifier the corpus has never shown (fresnelc, erf, future
special functions) encodes without fixture-conditioned atom
additions. Token counts under the proposal are computed by the
same greedy longest-match walk with a one-token-per-char fallback
instead of a raise.

Census outputs:
  1. all 725 calibration actions: full pair-sequence
     ("Current: {parent}\\nHints: none\\nStep: {child}\\n") token
     lengths under the proposed tokenizer (100% encodable by
     construction);
  2. action-level and whole-decision-level fit at ctx in
     {512, 1024, 2048, 4096, 8192};
  3. the base training diet (data/micromodel_chains_shard*.jsonl
     + data/step_chains.jsonl, the default train_mathnative load):
     row counts and sequence-length distribution under STOCK
     strict encoding (rows skipped = out-of-language) v under the
     proposed tokenizer (newly admitted rows and their lengths).

Reads logs/mathworld1/census.jsonl (frozen) for the action rows?
No — action texts are not stored there; this desk re-walks the 40
frozen episodes with the same binding assertion (hash equality
against logs/mathworld0/active.jsonl, abort on mismatch).

Receipt: logs/mathworld1/substrate_desk.json (refuse-if-exists).

    .venv/bin/python scratch/mathworld1_substrate_desk.py     (Mac)
"""
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.mathgen.problems import make_integrate  # noqa: E402
from llmopt.search.derivation import State, successors  # noqa: E402
from llmopt.train.mathnative import MathTokenizer  # noqa: E402

OUT = Path("logs/mathworld1/substrate_desk.json")
X = sp.Symbol("x")
CTXS = [512, 1024, 2048, 4096, 8192]


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/mathworld1_substrate_desk.py",
         "scratch/mathworld0.py", "llmopt/search/derivation.py",
         "llmopt/mathgen/problems.py", "llmopt/train/mathnative.py"])
    tok = MathTokenizer()

    def prop_len(s: str) -> tuple[int, int]:
        """(tokens, fallback_bytes) under ATOMS + byte fallback."""
        n = fb = i = 0
        while i < len(s):
            for t in tok._by_len:
                if s.startswith(t, i):
                    n += 1
                    i += len(t)
                    break
            else:
                n += 1
                fb += len(s[i].encode())
                i += 1
        return n, fb

    def stock_len(s: str):
        try:
            return len(tok.encode(s))
        except ValueError:
            return None

    # --- (1)+(2) calibration actions ---
    rows = [json.loads(line) for line in
            Path("logs/mathworld0/active.jsonl").read_text()
            .splitlines() if "meta" not in line]
    eps = {}
    for r in rows:
        eps.setdefault(r["episode_id"], []).append(r)
    pair_lens = []
    dec_lens = {}          # (eid, sid) -> [lengths of all K seqs]
    t0 = time.monotonic()
    for eid, erows in eps.items():
        lv = int(eid[1])
        seed = int(eid.split("-s")[1])
        prob = make_integrate(lv, seed)
        st = State(sp.Integral(prob._expr, X))
        for r in erows:
            if not r["chosen_action"]:
                break
            acts = sorted(successors(st),
                          key=lambda nc: (nc[0], nc[1].key()))
            ah = sha("\n".join(f"{n}|{c.key()}" for n, c in acts))
            if ah != r["legal_action_set_hash"]:
                raise SystemExit(
                    f"BINDING MISMATCH {eid} step {r['step_id']}")
            parent = str(st.expr)
            lens = []
            for name, c in acts:
                seq = (f"Current: {parent}\nHints: none\n"
                       f"Step: {str(c.expr)}\n")
                n, _ = prop_len(seq)
                lens.append(n)
                pair_lens.append(n)
            dec_lens[(eid, r["step_id"])] = (eid, lens)
            match = [nc for nc in acts
                     if f"{nc[0]}#{sha(nc[1].key())}"
                     == r["chosen_action"]]
            if not match:
                raise SystemExit(f"BINDING MISMATCH (chosen) {eid}")
            st = match[0][1]
    pl = sorted(pair_lens)
    fit = {}
    for ctx in CTXS:
        a = sum(1 for x in pl if x <= ctx)
        dec_ok = ep_ok = 0
        ep_all = {}
        for (eid, sid), (e, lens) in dec_lens.items():
            ok = all(x <= ctx for x in lens)
            dec_ok += ok
            ep_all.setdefault(e, True)
            ep_all[e] &= ok
        ep_ok = sum(ep_all.values())
        fit[ctx] = {"actions": a, "decisions": dec_ok,
                    "episodes": ep_ok}
    # --- (3) training diet ---
    diet_rows = []
    for f in sorted(glob.glob("data/micromodel_chains_shard*.jsonl")):
        diet_rows += [json.loads(line) for line in open(f)]
    diet_rows += [json.loads(line)
                  for line in open("data/step_chains.jsonl")]
    stock_lens, prop_lens_admit = [], []
    newly_admitted = 0
    for r in diet_rows:
        t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
        s = stock_len(t)
        n, _ = prop_len(t)
        if s is not None:
            stock_lens.append(s)
        else:
            newly_admitted += 1
            prop_lens_admit.append(n)
    sl = sorted(stock_lens)
    na = sorted(prop_lens_admit)

    def dist(v):
        return {} if not v else {
            "n": len(v), "med": v[len(v) // 2],
            "p90": v[int(0.9 * len(v))], "max": v[-1],
            "le512": sum(1 for x in v if x <= 512)}

    receipt = {
        "proposed_tokenizer": "ATOMS greedy longest-match + "
                              "deterministic single-byte fallback "
                              "per uncovered char (grammar-closed "
                              "over sstr; no fixture-conditioned "
                              "atoms)",
        "calibration_actions": len(pair_lens),
        "pair_seq_tokens_proposed": {
            "med": pl[len(pl) // 2], "p90": pl[int(0.9 * len(pl))],
            "max": pl[-1]},
        "fit_by_ctx": fit,
        "decisions_total": len(dec_lens),
        "episodes_total": len(eps),
        "diet_rows_total": len(diet_rows),
        "diet_stock_encoded": dist(sl),
        "diet_newly_admitted": dist(na),
        "wall_s": round(time.monotonic() - t0, 1),
        "start": START, "completion_commit": completion_commit()}
    OUT.write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
