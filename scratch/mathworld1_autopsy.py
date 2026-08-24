"""LABEL-YIELD-0 spent-band autopsy (chat analysis only, no repo
mutation): exhaustive one-deviation rescue census of the four L4
failures under frozen TERMINAL-FIRST + theta_0 continuations.
Outputs JSON to the scratchpad. Spent seeds only, zero training."""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

REPO = Path("/Users/artin/code/llmopt")
sys.path.insert(0, str(REPO))
import os
os.chdir(REPO)

import sympy as sp  # noqa: E402
import torch  # noqa: E402

import llmopt.search.derivation as derivation  # noqa: E402
from llmopt.mathgen.problems import make_integrate  # noqa: E402
from llmopt.search.derivation import (State, is_solved,  # noqa: E402
                                      successors)
from llmopt.train.mathnative import build_model  # noqa: E402
from scratch.mathworld1_birth import GCTok  # noqa: E402

OUT = Path("/private/tmp/claude-501/-Users-artin-code-llmopt/"
           "5054de3f-0ab5-4351-9fa8-d8460dae3223/scratchpad/"
           "autopsy_out.json")
MAX_DECISIONS = 12
WALL_CAP_S = 60.0
CTX = 4096
X = sp.Symbol("x")
FAILS = [9503, 9504, 9507, 9518]


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()[:16]


class World:
    def __init__(self):
        self.cache = {}
        self.walls = {}

    def legal(self, st):
        k = st.key()
        if k not in self.cache:
            derivation._RULE_CACHE.clear()
            t0 = time.monotonic()
            acts = sorted(successors(st),
                          key=lambda nc: (nc[0], nc[1].key()))
            self.walls[k] = time.monotonic() - t0
            self.cache[k] = [(n, c.expr) for n, c in acts]
        acts = [(n, State(e, st.plies + 1, st.history + (n,)))
                for n, e in self.cache[k]]
        ah = sha("\n".join(f"{n}|{c.key()}" for n, c in acts))
        return acts, ah


class Scorer:
    def __init__(self, model, tok, dev):
        self.model, self.tok, self.dev = model, tok, dev

    def score(self, pre, cid):
        ids = torch.tensor([pre + cid], device=self.dev)
        with torch.no_grad():
            logits = self.model(ids)
        lp = torch.log_softmax(logits[0].float(), -1)
        s = sum(lp[len(pre) + i - 1, t].item()
                for i, t in enumerate(cid))
        assert math.isfinite(s)
        return s

    def rank(self, st, acts):
        pre = self.tok.encode(
            f"Current: {str(st.expr)}\nHints: none\nStep: ")
        cand = []
        for name, c in acts:
            cid = self.tok.encode(str(c.expr) + "\n")
            if len(pre) + len(cid) > CTX:
                return None
            cand.append((name, c, cid))
        scored = [(self.score(pre, cid), n, c)
                  for n, c, cid in cand]
        scored.sort(key=lambda t: (-t[0], t[1], t[2].key()))
        return scored


def tf_walk(st, world, scorer, budget):
    charged = 0.0
    paid = set()
    steps = []
    for d in range(budget):
        if is_solved(st):
            return {"outcome": "solved", "depth": d,
                    "steps": steps}
        if charged > WALL_CAP_S:
            return {"outcome": "wall_cap", "depth": d,
                    "steps": steps}
        acts, ah = world.legal(st)
        k = st.key()
        if k not in paid:
            charged += world.walls.get(k, 0.0)
            paid.add(k)
        t0 = time.monotonic()
        if not acts:
            return {"outcome": "dead_end", "depth": d,
                    "steps": steps}
        terms = [(n, c) for n, c in acts if is_solved(c)]
        if terms:
            name, child = min(terms,
                              key=lambda nc: (nc[0],
                                              nc[1].key()))
            charged += time.monotonic() - t0
            steps.append((f"{name}#{sha(child.key())}", False))
            st = child
            continue
        scored = scorer.rank(st, acts)
        charged += time.monotonic() - t0
        if scored is None:
            return {"outcome": "model_ctx_overflow", "depth": d,
                    "steps": steps}
        _, n0, c0 = scored[0]
        steps.append((f"{n0}#{sha(c0.key())}", True))
        st = c0
    if is_solved(st):
        return {"outcome": "solved", "depth": budget,
                "steps": steps}
    return {"outcome": "budget_exhausted", "depth": budget,
            "steps": steps}


def truncate(tr, h):
    if tr["outcome"] == "solved" and tr["depth"] <= h:
        return "solved", tr["depth"]
    if (tr["outcome"] in ("dead_end", "model_ctx_overflow",
                          "wall_cap") and tr["depth"] <= h):
        return tr["outcome"], tr["depth"]
    return "budget_exhausted", h


def main():
    rows = [json.loads(l) for l in
            open("logs/mathworld1/yield_census.jsonl")
            if l.strip()]
    tok = GCTok()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_model(tok.vocab_size, ctx=CTX).to(dev)
    model.load_state_dict(torch.load(
        "checkpoints/mathnative_19m_mw1_theta0.pt",
        map_location=dev))
    model.eval()
    scorer = Scorer(model, tok, dev)
    world = World()
    memo = {}
    out = {}
    for seed in FAILS:
        eid = f"L4-s{seed}"
        rec = sorted([r for r in rows
                      if r.get("row") == "decision"
                      and r["episode_id"] == eid],
                     key=lambda r: r["step"])
        st = State(sp.Integral(make_integrate(4, seed)._expr, X))
        sites = []
        binding_ok = True
        for r in rec:
            k = st.key()
            if sha(k) != r["state_hash"]:
                binding_ok = False
                break
            acts, ah = world.legal(st)
            if ah != r["legal_set_hash"]:
                binding_ok = False
                break
            scored = scorer.rank(st, acts)
            assert scored is not None
            ids = [f"{n}#{sha(c.key())}" for _, n, c in scored]
            argmax_match = ids[0] == r["chosen"]
            forks = []
            for rk, (s_alt, n, c) in enumerate(scored, 1):
                fid = f"{n}#{sha(c.key())}"
                if fid == r["chosen"]:
                    chosen_rank = rk
                    chosen_score = s_alt
                    continue
                if is_solved(c):
                    tr = {"outcome": "solved", "depth": 0}
                else:
                    ck = c.key()
                    if ck not in memo:
                        memo[ck] = tf_walk(
                            c, world, scorer,
                            MAX_DECISIONS - 1)
                    tr = memo[ck]
                h = MAX_DECISIONS - (r["step"] + 1)
                o, d = truncate(tr, h)
                forks.append({
                    "rank": rk, "forced": fid,
                    "score": round(s_alt, 4),
                    "native_h": h, "native_outcome": o,
                    "native_depth": d,
                    "full_outcome": tr["outcome"],
                    "full_depth": tr["depth"],
                    "child_hash": fid.rsplit("#", 1)[1]})
            sites.append({
                "step": r["step"],
                "state_hash": r["state_hash"],
                "n_legal": len(scored),
                "chosen": r["chosen"],
                "chosen_rank_rescored": chosen_rank,
                "chosen_score": round(chosen_score, 4),
                "argmax_match": argmax_match,
                "forks": forks})
            # advance by recorded chosen
            nxt = [c for _, n, c in scored
                   if f"{n}#{sha(c.key())}" == r["chosen"]]
            assert len(nxt) == 1
            st = nxt[0]
            print(f"[autopsy] {eid} step {r['step']} "
                  f"n={len(scored)}", flush=True)
        out[eid] = {"binding_ok": binding_ok, "sites": sites}
    OUT.write_text(json.dumps(out))
    print("[autopsy] done", flush=True)


if __name__ == "__main__":
    main()
