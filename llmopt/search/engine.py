"""The measured-best engines, as one import (integration of the
2026-07-06..08 racing results — see docs/RESULTS.md).

    from llmopt.search.engine import solve
    result = solve(sp.Integral(x * sp.cos(x), x))

Default = the zero-neural-network configuration that scored 316/360
(87.8%) on the held-out races: rule-bigram Markov prior (top-3
pruning), width-2 beam, sampled verification (winning paths always
fully verified). The 337/360 record config additionally needs the
LoRA'd 0.5B for entropy-gated adaptive k — pass its score_fn via
`llm_score_fn` to enable it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import sympy as sp

from llmopt.search.derivation import SearchResult, State, beam_search
from llmopt.search.proposer import entropy_k, make_scoring_proposer

_PRIOR_PATH = Path(__file__).resolve().parents[2] / "checkpoints" / "markov_prior.json"


class MarkovPrior:
    """Rule-bigram counts from verifier-approved winning paths. The
    measured result behind it: ties the fine-tuned 0.5B at top-3
    offline, beats it in-search (293 vs 288) at zero inference cost."""

    def __init__(self, unigram: dict, bigram: dict):
        self.unigram = unigram
        self.bigram = bigram

    @classmethod
    def load(cls, path: Path = _PRIOR_PATH) -> "MarkovPrior":
        d = json.loads(Path(path).read_text())
        return cls(d["unigram"], d["bigram"])

    @classmethod
    def from_rows(cls, rows: list[dict]) -> "MarkovPrior":
        unigram: dict = {}
        bigram: dict = {}
        prev = None
        for row in rows:
            chosen = row["moves"][row["answer"]].split("@")[0]
            unigram[chosen] = unigram.get(chosen, 0) + 1
            if prev is not None:
                bigram.setdefault(prev, {})
                bigram[prev][chosen] = bigram[prev].get(chosen, 0) + 1
            prev = chosen
        return cls(unigram, bigram)

    def proposer(self):
        # Unseen-rule trial mass: rules the prior has never seen score
        # 0.5*median OUTRIGHT (not 0.01*median). The 0.01 factor was
        # measured insufficient (2026-07-10 five-ply trace: i_log_power
        # fired every ply, scored 0.74 vs bigram 34+, never cracked
        # top-3). Full trial mass is safe because rules only fire on
        # matching structure — the fire itself is the evidence. Do NOT
        # fix this by re-mining the prior instead: measured twice to
        # regress (dilution; L5 89.6% -> 73.1%, and 478e269).
        med = sorted(self.unigram.values())[len(self.unigram) // 2] \
            if self.unigram else 1

        def prop(state: State, children):
            prev = (state.history[-1].split("@")[0]
                    if state.history else None)
            table = self.bigram.get(prev) if prev else None

            def s(name: str) -> float:
                r = name.split("@")[0]
                if r not in self.unigram:
                    return 0.5 * med
                return ((table.get(r, 0) if table else 0)
                        + 0.01 * self.unigram[r])

            return sorted(children, key=lambda c: -s(c[0]))

        return prop


def solve(expr: sp.Expr, *, budget: int = 200,
          prior: MarkovPrior | None = None,
          llm_score_fn: Callable | None = None,
          use_macros: bool = True, magic: bool = True) -> SearchResult:
    """Solve with the measured-best configuration.

    Without llm_score_fn: markov3 @ width 2 (334/360-class with the
    autopsy rules, zero NN). With it: hybrid markov-ranks +
    entropy-gated k at T=0.1 (349/360-class record config).
    magic=True prunes Risch-certified dead states (RESULTS: +1
    replicated, 71 cuts at int L4, provably zero false positives)."""
    state_filter = None
    if magic:
        from llmopt.search.magic import is_dead
        state_filter = lambda s: not is_dead(s)  # noqa: E731
    if llm_score_fn is not None:
        return beam_search(
            expr, width=2, max_plies=24, max_nodes=budget,
            proposer=make_scoring_proposer(llm_score_fn),
            propose_k=entropy_k(1, 3, temperature=0.1),
            use_macros=use_macros, verify_p=0.1,
            state_filter=state_filter)
    # width 3 since 2026-07-10 node-cost round 2: the verify fix
    # (doit(integrals=False)) made nodes cheap enough to widen —
    # raced L5 238/249 (95.6%, from 223) at 3.5x LESS wall; L3 +1,
    # L4 tied on 60-problem samples. Width 2 lost composition-fragile
    # chains whose every rule existed (9-problem residue -> 8 solved).
    prior = prior or MarkovPrior.load()
    return beam_search(
        expr, width=3, max_plies=24, max_nodes=budget,
        proposer=prior.proposer(), propose_k=3,
        use_macros=use_macros, verify_p=0.1,
        state_filter=state_filter)
